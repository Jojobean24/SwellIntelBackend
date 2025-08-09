import os
import math
import requests
from datetime import datetime, date, timedelta
from typing import Dict, Tuple, Optional

from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# --------- Env & constants ----------
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")  # set in Render → Environment
STATIC_DIR = "static"
AI_CACHE_TTL_SECONDS = 1800  # 30 minutes
os.makedirs(STATIC_DIR, exist_ok=True)

# --------- App ----------
app = FastAPI(title="Swell Intel Backend", version="1.7.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Simple cache for image responses keyed by rounded lat/lon
_ai_cache: Dict[Tuple[float, float], Dict] = {}

# --------- Helpers ----------
def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def _float_or_none(x: Optional[str]) -> Optional[float]:
    try:
        if x is None: return None
        if isinstance(x, str) and x.upper() == "MM": return None
        return float(x)
    except Exception:
        return None

def deg_to_cardinal(deg: float) -> str:
    dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
    ix = int((deg % 360) / 22.5 + 0.5) % 16
    return dirs[ix]

def is_offshore_east_coast(wdir: Optional[float]) -> bool:
    # Heuristic for US East Coast: W quadrant is roughly offshore (210–330)
    if wdir is None: return False
    return 210 <= (wdir % 360) <= 330

def get_noaa_stations():
    url = "https://www.ndbc.noaa.gov/activestations.xml"
    r = requests.get(url, timeout=12)
    r.raise_for_status()
    from xml.etree import ElementTree
    root = ElementTree.fromstring(r.content)
    stations = []
    for s in root.findall("station"):
        try:
            stations.append({
                "id": s.attrib["id"],
                "lat": float(s.attrib["lat"]),
                "lon": float(s.attrib["lon"]),
                "name": s.attrib.get("name", ""),
            })
        except Exception:
            continue
    return stations

def fetch_noaa_conditions(station_id: str) -> Optional[dict]:
    url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt"
    r = requests.get(url, timeout=12)
    if r.status_code != 200:
        return None
    lines = r.text.splitlines()
    if len(lines) < 3:
        return None

    header = lines[0].split()
    latest = lines[2].split()
    idx = {h: i for i, h in enumerate(header)}

    wvht_m = _float_or_none(latest[idx["WVHT"]]) if "WVHT" in idx else None
    apd = _float_or_none(latest[idx["APD"]]) if "APD" in idx else None
    dpd = _float_or_none(latest[idx["DPD"]]) if "DPD" in idx else None
    wspd_raw = _float_or_none(latest[idx["WSPD"]]) if "WSPD" in idx else None
    wdir_deg = _float_or_none(latest[idx["WDIR"]]) if "WDIR" in idx else None

    if wvht_m is None:
        return None

    wind_mph = None
    if wspd_raw is not None:
        # NDBC usually reports knots; convert to mph
        wind_mph = round(wspd_raw * 1.15078, 1)

    period_s = apd if apd is not None else dpd

    return {
        "wave_height_ft": round(wvht_m * 3.281, 1),
        "wave_period_s": round(period_s, 1) if period_s is not None else None,
        "wind_speed_mph": wind_mph,
        "wind_dir_deg": wdir_deg,
        "wind_dir_txt": deg_to_cardinal(wdir_deg) if wdir_deg is not None else None,
    }

def find_nearest_station_with_waves(lat: float, lon: float, max_candidates: int = 300):
    stations = get_noaa_stations()
    stations.sort(key=lambda s: haversine(lat, lon, s["lat"], s["lon"]))

    numeric = [s for s in stations if s["id"].isdigit()]
    non_numeric = [s for s in stations if not s["id"].isdigit()]

    tried = 0
    for s in numeric:
        if tried >= max_candidates: break
        tried += 1
        cond = fetch_noaa_conditions(s["id"])
        if cond:
            return s, cond

    for s in non_numeric[:max_candidates]:
        cond = fetch_noaa_conditions(s["id"])
        if cond:
            return s, cond

    return stations[0], None

def make_abs(request: Request, path: str) -> str:
    base = str(request.base_url).rstrip("/")
    return f"{base}{path if path.startswith('/') else '/' + path}"

# --------- Stability image ----------
def stability_ai_image(prompt: str) -> Optional[str]:
    """Generate an image with Stability if API key present; return relative /static path."""
    if not STABILITY_API_KEY:
        print("[stability] STABILITY_API_KEY not set; using placeholder.")
        return None

    url = "https://api.stability.ai/v2beta/stable-image/generate/core"
    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Accept": "image/*",  # Stability requires this
    }
    # IMPORTANT: use multipart/form-data
    files = {
        "prompt": (None, prompt),
        "model": (None, "sd3.5-large"),
        "output_format": (None, "png"),
        "aspect_ratio": (None, "16:9"),
    }
    try:
        r = requests.post(url, headers=headers, files=files, timeout=60)
        if r.status_code != 200:
            print("[stability] API error:", r.status_code, r.text[:400])
            return None
        fname = f"surf_{int(datetime.utcnow().timestamp())}.png"
        path = os.path.join(STATIC_DIR, fname)
        with open(path, "wb") as f:
            f.write(r.content)
        return f"/static/{fname}"
    except Exception as e:
        print("[stability] Exception:", e)
        return None

# --------- Tides (NOAA CO-OPS) ----------
def _noaa_get(url: str):
    try:
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("[tides] fetch error:", e)
        return None

def fetch_todays_hilo(tide_station: str):
    today = date.today().strftime("%Y%m%d")
    url = (
        "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?"
        f"product=predictions&application=swellintel&begin_date={today}&end_date={today}"
        f"&datum=MLLW&station={tide_station}&time_zone=lst_ldt&units=english&interval=hilo&format=json"
    )
    j = _noaa_get(url)
    return j.get("predictions") if j else None

def fetch_todays_hourly(tide_station: str):
    today = date.today().strftime("%Y%m%d")
    url = (
        "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?"
        f"product=predictions&application=swellintel&begin_date={today}&end_date={today}"
        f"&datum=MLLW&station={tide_station}&time_zone=lst_ldt&units=english&interval=h&format=json"
    )
    j = _noaa_get(url)
    return j.get("predictions") if j else None

def parse_low_to_rising_window_from_hilo(preds):
    if not preds: return None
    fmt = "%Y-%m-%d %H:%M"
    now = datetime.now()
    lows = [p for p in preds if p.get("type") == "L"]
    if not lows: return None
    next_low = None
    for p in lows:
        t = datetime.strptime(p["t"], fmt)
        if t >= now:
            next_low = t
            break
    if not next_low:
        next_low = datetime.strptime(lows[-1]["t"], fmt)
    return {"start": next_low.isoformat(), "end": (next_low + timedelta(hours=2)).isoformat()}

def parse_low_to_rising_window_from_hourly(preds):
    if not preds: return None
    fmt = "%Y-%m-%d %H:%M"
    now = datetime.now()
    series = []
    for p in preds:
        try:
            t = datetime.strptime(p["t"], fmt)
            v = float(p["v"])
            series.append((t, v))
        except:
            continue
    if len(series) < 3: return None

    mins = []
    for i in range(1, len(series) - 1):
        t0, v0 = series[i-1]
        t1, v1 = series[i]
        t2, v2 = series[i+1]
        if v1 <= v0 and v1 <= v2:
            mins.append(series[i][0])

    next_low = None
    for t in mins:
        if t >= now:
            next_low = t
            break
    if not next_low:
        next_low = mins[-1]

    return {"start": next_low.isoformat(), "end": (next_low + timedelta(hours=2)).isoformat()}

# --------- Routes ----------
@app.get("/health")
def health():
    return {"ok": True, "ts": datetime.utcnow().isoformat()}

@app.get("/summary")
def summary(lat: float, lon: float, station_id: Optional[str] = Query(None)):
    if station_id:
        cond = fetch_noaa_conditions(station_id)
        station = {"id": station_id, "name": f"NOAA {station_id}", "lat": lat, "lon": lon}
        if not cond:
            return {"summary": f"No live wave data on station {station_id}.", "station": station}
        parts = [f"{cond['wave_height_ft']} ft"]
        if cond.get("wave_period_s") is not None: parts.append(f"@ {cond['wave_period_s']} s")
        if cond.get("wind_speed_mph") is not None: parts.append(f"wind {cond['wind_speed_mph']} mph")
        text = ", ".join(parts) + f" — {station['name']} ({station['id']})"
        return {"summary": text, "station": station}

    station, cond = find_nearest_station_with_waves(lat, lon)
    if not cond:
        return {"summary": "No live wave data available nearby.", "station": station}

    parts = [f"{cond['wave_height_ft']} ft"]
    if cond.get("wave_period_s") is not None: parts.append(f"@ {cond['wave_period_s']} s")
    if cond.get("wind_speed_mph") is not None: parts.append(f"wind {cond['wind_speed_mph']} mph")
    text = ", ".join(parts) + f" — {station['name']} ({station['id']})"
    return {"summary": text, "station": station}

@app.get("/forecast-image")
def forecast_image(
    request: Request,
    lat: float,
    lon: float,
    station_id: Optional[str] = Query(None),
    force: int = Query(0)  # force=1 disables cache during dev
):
    cache_key = None
    now_ts = datetime.utcnow().timestamp()
    if not station_id:
        cache_key = (round(lat, 3), round(lon, 3))
        if not force and cache_key in _ai_cache and now_ts - _ai_cache[cache_key]["ts"] < AI_CACHE_TTL_SECONDS:
            return _ai_cache[cache_key]["data"]

    if station_id:
        cond = fetch_noaa_conditions(station_id)
        station = {"id": station_id, "name": f"NOAA {station_id}", "lat": lat, "lon": lon}
    else:
        station, cond = find_nearest_station_with_waves(lat, lon)

    if not cond:
        data = {
            "summary": "No live wave data available nearby.",
            "imageUrl": make_abs(request, "/static/forecast.jpg"),
            "station": station,
            "imageProvider": "placeholder"
        }
        if cache_key:
            _ai_cache[cache_key] = {"data": data, "ts": now_ts}
        return data

    parts = [f"{cond['wave_height_ft']} ft"]
    if cond.get("wave_period_s") is not None: parts.append(f"@ {cond['wave_period_s']} s")
    if cond.get("wind_speed_mph") is not None: parts.append(f"wind {cond['wind_speed_mph']} mph")
    summary_text = ", ".join(parts) + f" — {station['name']}"

    prompt = (
        f"Photorealistic surf scene near {station['name']}, waves {cond['wave_height_ft']} feet"
        + (f" at {cond['wave_period_s']} seconds" if cond.get('wave_period_s') is not None else "")
        + (f", wind {cond['wind_speed_mph']} mph" if cond.get('wind_speed_mph') is not None else "")
        + ". Natural colors, realistic ocean texture, late light, 16:9 composition, high detail."
    )

    image_rel = stability_ai_image(prompt)
    provider = "stability" if image_rel else "placeholder"
    if not image_rel:
        image_rel = "/static/forecast.jpg"

    data = {"summary": summary_text, "imageUrl": make_abs(request, image_rel), "station": station, "imageProvider": provider}
    if cache_key:
        _ai_cache[cache_key] = {"data": data, "ts": now_ts}
    return data

@app.get("/wind")
def wind(lat: float, lon: float):
    station, cond = find_nearest_station_with_waves(lat, lon)
    if not cond:
        raise HTTPException(404, "No nearby wave station.")
    return {
        "station": station,
        "wind_speed_mph": cond.get("wind_speed_mph"),
        "wind_dir_deg": cond.get("wind_dir_deg"),
        "wind_dir_txt": cond.get("wind_dir_txt"),
        "offshore": is_offshore_east_coast(cond.get("wind_dir_deg")),
    }

# ---------- Strike window ----------
def _noaa_get_json(url: str):
    try:
        r = requests.get(url, timeout=12); r.raise_for_status()
        return r.json()
    except Exception as e:
        print("[tides] fetch error:", e); return None

def fetch_todays_hilo(tide_station: str):
    today = date.today().strftime("%Y%m%d")
    url = ("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?"
           f"product=predictions&application=swellintel&begin_date={today}&end_date={today}"
           f"&datum=MLLW&station={tide_station}&time_zone=lst_ldt&units=english&interval=hilo&format=json")
    j = _noaa_get_json(url); return j.get("predictions") if j else None

def fetch_todays_hourly(tide_station: str):
    today = date.today().strftime("%Y%m%d")
    url = ("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?"
           f"product=predictions&application=swellintel&begin_date={today}&end_date={today}"
           f"&datum=MLLW&station={tide_station}&time_zone=lst_ldt&units=english&interval=h&format=json")
    j = _noaa_get_json(url); return j.get("predictions") if j else None

def parse_low_to_rising_window_from_hilo(preds):
    if not preds: return None
    fmt = "%Y-%m-%d %H:%M"; now = datetime.now()
    lows = [p for p in preds if p.get("type") == "L"]
    if not lows: return None
    next_low = None
    for p in lows:
        t = datetime.strptime(p["t"], fmt)
        if t >= now: next_low = t; break
    if not next_low: next_low = datetime.strptime(lows[-1]["t"], fmt)
    return {"start": next_low.isoformat(), "end": (next_low + timedelta(hours=2)).isoformat()}

def parse_low_to_rising_window_from_hourly(preds):
    if not preds: return None
    fmt = "%Y-%m-%d %H:%M"; now = datetime.now()
    series = []
    for p in preds:
        try:
            t = datetime.strptime(p["t"], fmt); v = float(p["v"])
            series.append((t, v))
        except: pass
    if len(series) < 3: return None
    mins = []
    for i in range(1, len(series)-1):
        _, v0 = series[i-1]; t1, v1 = series[i]; _, v2 = series[i+1]
        if v1 <= v0 and v1 <= v2: mins.append(t1)
    next_low = None
    for t in mins:
        if t >= now: next_low = t; break
    if not next_low: next_low = mins[-1]
    return {"start": next_low.isoformat(), "end": (next_low + timedelta(hours=2)).isoformat()}

@app.get("/optimal-window")
def optimal_window(lat: float, lon: float, tide_station: str = "8720291"):
    window = None
    hilo = fetch_todays_hilo(tide_station)
    window = parse_low_to_rising_window_from_hilo(hilo)
    if not window:
        hourly = fetch_todays_hourly(tide_station)
        window = parse_low_to_rising_window_from_hourly(hourly)
    if not window:
        window = {"start": None, "end": None}

    station, cond = find_nearest_station_with_waves(lat, lon)
    wind_dir = cond.get("wind_dir_deg") if cond else None
    wind_spd = cond.get("wind_speed_mph") if cond else None
    offshore = is_offshore_east_coast(wind_dir)

    # Decide the note
    note = None
    if window.get("start"):
        wave_height = cond.get("wave_height_ft") if cond else None
        if wave_height is not None and wave_height > 3:
            note = "Go shred, it's good!" if offshore else "Go to work."
        else:
            note = "Surf's small, maybe work."
    else:
        note = "No explicit low found; check tide station or try another nearby."

    return {
        "tide_station": tide_station,
        "window": window,
        "wind": {
            "dir_deg": wind_dir,
            "dir_txt": deg_to_cardinal(wind_dir) if wind_dir is not None else None,
            "speed_mph": wind_spd,
            "offshore": offshore,
        },
        "buoy_station": station,
        "note": note,
    }

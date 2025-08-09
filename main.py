import os
import math
import requests
from datetime import datetime, date, timedelta
from typing import Dict, Tuple, Optional, List

from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# =========================
# Config
# =========================
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")  # optional; if missing we use placeholder
STATIC_DIR = "static"
AI_CACHE_TTL_SECONDS = 1800  # 30 min
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI(title="Swell Intel Backend", version="2.5.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_ai_cache: Dict[Tuple[float, float], Dict] = {}

# =========================
# Utils
# =========================
def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def _float_or_none(x):
    try:
        if x is None: return None
        if isinstance(x, str) and x.upper() == "MM": return None
        return float(x)
    except:
        return None

def clamp(v, lo, hi): return max(lo, min(hi, v))

def deg_to_cardinal(deg: Optional[float]) -> Optional[str]:
    if deg is None: return None
    dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
    ix = int((deg % 360) / 22.5 + 0.5) % 16
    return dirs[ix]

def is_offshore_east_coast(wdir: Optional[float]) -> bool:
    if wdir is None: return False
    return 210 <= (wdir % 360) <= 330  # W quadrant ~ offshore

def is_onshore_east_coast(wdir: Optional[float]) -> bool:
    if wdir is None: return False
    return 30 <= (wdir % 360) <= 150   # E quadrant ~ onshore

def make_abs(request: Request, path: str) -> str:
    base = str(request.base_url).rstrip("/")
    return f"{base}{path if path.startswith('/') else '/' + path}"

# =========================
# NOAA Buoy (live)
# =========================
def get_noaa_stations():
    url = "https://www.ndbc.noaa.gov/activestations.xml"
    r = requests.get(url, timeout=12); r.raise_for_status()
    from xml.etree import ElementTree
    root = ElementTree.fromstring(r.content)
    out = []
    for s in root.findall("station"):
        try:
            out.append({
                "id": s.attrib["id"],
                "lat": float(s.attrib["lat"]),
                "lon": float(s.attrib["lon"]),
                "name": s.attrib.get("name",""),
            })
        except: pass
    return out

def fetch_noaa_conditions(station_id: str) -> Optional[dict]:
    url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt"
    r = requests.get(url, timeout=12)
    if r.status_code != 200: return None
    lines = r.text.splitlines()
    if len(lines) < 3: return None
    header = lines[0].split()
    latest = lines[2].split()
    idx = {h:i for i,h in enumerate(header)}

    wvht_m = _float_or_none(latest[idx["WVHT"]]) if "WVHT" in idx else None
    apd = _float_or_none(latest[idx["APD"]]) if "APD" in idx else None
    dpd = _float_or_none(latest[idx["DPD"]]) if "DPD" in idx else None
    wspd = _float_or_none(latest[idx["WSPD"]]) if "WSPD" in idx else None  # knots
    wdir = _float_or_none(latest[idx["WDIR"]]) if "WDIR" in idx else None

    if wvht_m is None: return None
    wind_mph = round((wspd or 0) * 1.15078, 1) if wspd is not None else None
    period_s = apd if apd is not None else dpd

    return {
        "wave_height_ft": round(wvht_m * 3.281, 1),
        "wave_period_s": round(period_s, 1) if period_s is not None else None,
        "wind_speed_mph": wind_mph,
        "wind_dir_deg": wdir,
        "wind_dir_txt": deg_to_cardinal(wdir) if wdir is not None else None,
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
        if cond: return s, cond
    for s in non_numeric[:max_candidates]:
        cond = fetch_noaa_conditions(s["id"])
        if cond: return s, cond
    return stations[0], None

# =========================
# Open‑Meteo (current + marine)
# =========================
def fetch_current_weather(lat: float, lon: float) -> Optional[dict]:
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        "&current=precipitation,cloud_cover,weather_code,wind_speed_10m,wind_direction_10m,is_day"
        "&timezone=auto"
    )
    try:
        r = requests.get(url, timeout=12); r.raise_for_status()
        cur = r.json().get("current", {})
        return {
            "wcode": cur.get("weather_code"),
            "cloud_cover": cur.get("cloud_cover"),
            "precip": cur.get("precipitation"),
            "wind10m": cur.get("wind_speed_10m"),
            "winddir10m": cur.get("wind_direction_10m"),
            "is_day": cur.get("is_day"),
        }
    except Exception as e:
        print("[weather] fetch error:", e)
        return None

def describe_weather(w: Optional[dict]) -> str:
    if not w: return "clear skies, daylight"
    code = w.get("wcode")
    clouds = w.get("cloud_cover") or 0
    precip = (w.get("precip") or 0)
    is_day = w.get("is_day", 1) == 1

    if code in (45,48): base = "foggy"
    elif code in (51,53,55): base = "light drizzle"
    elif code in (61,63): base = "light rain"
    elif code in (65,): base = "heavy rain"
    elif code in (80,81): base = "showers"
    elif code in (82,): base = "heavy showers"
    elif code in (71,73,75,77,85,86): base = "snow"
    elif code in (95,96,99): base = "thunderstorm"
    else:
        if clouds >= 85: base = "overcast"
        elif clouds >= 50: base = "mostly cloudy"
        elif clouds >= 20: base = "partly cloudy"
        else: base = "clear skies"
    if "rain" in base or "shower" in base or "drizzle" in base:
        if precip and precip >= 5: base = base.replace("light","moderate")
        if precip and precip >= 15: base = base.replace("moderate","heavy")
    return f"{base}, {'daylight' if is_day else 'night'}"

def fetch_weekly(lat: float, lon: float, days: int = 5):
    days = clamp(days, 1, 7)
    try:
        murl = (
            "https://marine-api.open-meteo.com/v1/marine?"
            f"latitude={lat}&longitude={lon}"
            "&daily=wave_height_max,wave_height_mean,wave_period_max"
            "&timezone=auto"
        )
        wurl = (
            "https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            "&daily=wind_speed_10m_max,wind_direction_10m_dominant"
            "&timezone=auto"
        )
        m = requests.get(murl, timeout=15).json()
        w = requests.get(wurl, timeout=15).json()
        dates = (m.get("daily", {}) or {}).get("time", [])[:days]
        hmean = (m.get("daily", {}) or {}).get("wave_height_mean", [])[:days]
        hmax  = (m.get("daily", {}) or {}).get("wave_height_max", [])[:days]
        pmax  = (m.get("daily", {}) or {}).get("wave_period_max", [])[:days]
        wspd  = (w.get("daily", {}) or {}).get("wind_speed_10m_max", [])[:days]
        wdir  = (w.get("daily", {}) or {}).get("wind_direction_10m_dominant", [])[:days]

        out = []
        if dates:
            for i, d in enumerate(dates):
                h_m = hmean[i] if i < len(hmean) and hmean[i] is not None else (hmax[i] if i < len(hmax) else 0.5)
                h_ft = round((h_m or 0) * 3.281, 1)
                h_ft_adj = max(h_ft - 1.0, 0.0)  # Jacksonville realism
                ws = wspd[i] if i < len(wspd) else None
                wd = wdir[i] if i < len(wdir) else None
                offshore = is_offshore_east_coast(wd)
                stars = 1
                if h_ft_adj > 1: stars = 2
                if h_ft_adj >= 3: stars = 3
                if offshore: stars += 1
                if h_ft_adj > 3 and offshore: stars = 5
                stars = clamp(stars, 1, 5)
                out.append({
                    "date": d,
                    "wave_height_ft_adj": round(h_ft_adj, 1),
                    "wave_period_s_max": (pmax[i] if i < len(pmax) else None),
                    "wind_speed_mph_max": round((ws or 0) * 0.621371, 1) if ws is not None else None,
                    "wind_dir_deg_dom": wd,
                    "wind_dir_txt_dom": deg_to_cardinal(wd) if wd is not None else None,
                    "offshore": offshore,
                    "stars": stars,
                    "summary": f"{round(h_ft_adj,1)} ft, wind {round((ws or 0)*0.621371,1) if ws is not None else '--'} mph {deg_to_cardinal(wd) or '--'}",
                })
            return out

        # Fallback if marine API empty
        w2 = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=wind_speed_10m_max,wind_direction_10m_dominant&timezone=auto",
            timeout=15
        ).json()
        dates2 = (w2.get("daily", {}) or {}).get("time", [])[:days]
        wspd2  = (w2.get("daily", {}) or {}).get("wind_speed_10m_max", [])[:days]
        wdir2  = (w2.get("daily", {}) or {}).get("wind_direction_10m_dominant", [])[:days]
        out2 = []
        for i, d in enumerate(dates2):
            wd = wdir2[i] if i < len(wdir2) else None
            ws = wspd2[i] if i < len(wspd2) else None
            base_ft = 2.0 if wd is not None else 1.5
            h_ft_adj = max(base_ft - 1.0, 0.5)
            offshore = is_offshore_east_coast(wd)
            stars = 1 + (1 if h_ft_adj > 1 else 0) + (1 if offshore else 0)
            stars = clamp(stars, 1, 5)
            out2.append({
                "date": d,
                "wave_height_ft_adj": round(h_ft_adj,1),
                "wave_period_s_max": None,
                "wind_speed_mph_max": round((ws or 0) * 0.621371, 1) if ws is not None else None,
                "wind_dir_deg_dom": wd,
                "wind_dir_txt_dom": deg_to_cardinal(wd) if wd is not None else None,
                "offshore": offshore,
                "stars": stars,
                "summary": f"{round(h_ft_adj,1)} ft est, wind {round((ws or 0)*0.621371,1) if ws is not None else '--'} mph {deg_to_cardinal(wd) or '--'}",
            })
        return out2

    except Exception as e:
        print("[weekly] error:", e)
        return []

# =========================
# Image (Stability or placeholder) — now biased small & messy
# =========================
def stability_ai_image(prompt: str):
    if not STABILITY_API_KEY:
        return None, "STABILITY_API_KEY not set"
    url = "https://api.stability.ai/v2beta/stable-image/generate/core"
    headers = {"Authorization": f"Bearer {STABILITY_API_KEY}", "Accept": "image/*"}
    files = {
        "prompt": (None, prompt),
        "model": (None, "sd3.5"),  # use sd3.5-large if you have it
        "output_format": (None, "png"),
        "aspect_ratio": (None, "16:9"),
    }
    try:
        r = requests.post(url, headers=headers, files=files, timeout=60)
        if r.status_code != 200:
            return None, f"API {r.status_code}: {r.text[:300]}"
        fname = f"surf_{int(datetime.utcnow().timestamp())}.png"
        path = os.path.join(STATIC_DIR, fname)
        with open(path, "wb") as f: f.write(r.content)
        return f"/static/{fname}", None
    except Exception as e:
        return None, f"Exception: {e}"

# =========================
# Tides (strike window)
# =========================
def _noaa_get(url: str):
    try:
        r = requests.get(url, timeout=12); r.raise_for_status(); return r.json()
    except Exception as e:
        print("[tides] fetch error:", e); return None

def fetch_todays_hilo(tide_station: str):
    today = date.today().strftime("%Y%m%d")
    url = ("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?"
           f"product=predictions&application=swellintel&begin_date={today}&end_date={today}"
           f"&datum=MLLW&station={tide_station}&time_zone=lst_ldt&units=english&interval=hilo&format=json")
    j = _noaa_get(url); return j.get("predictions") if j else None

def fetch_todays_hourly(tide_station: str):
    today = date.today().strftime("%Y%m%d")
    url = ("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?"
           f"product=predictions&application=swellintel&begin_date={today}&end_date={today}"
           f"&datum=MLLW&station={tide_station}&time_zone=lst_ldt&units=english&interval=h&format=json")
    j = _noaa_get(url); return j.get("predictions") if j else None

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
            t = datetime.strptime(p["t"], fmt); v = float(p["v"]); series.append((t, v))
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

# =========================
# Routes
# =========================
@app.get("/health")
def health(): return {"ok": True, "ts": datetime.utcnow().isoformat()}

@app.get("/summary")
def summary(lat: float, lon: float, station_id: Optional[str] = Query(None)):
    if station_id:
        cond = fetch_noaa_conditions(station_id)
        station = {"id": station_id, "name": f"NOAA {station_id}", "lat": lat, "lon": lon}
        if not cond: return {"summary": f"No live wave data on station {station_id}.", "station": station}
    else:
        station, cond = find_nearest_station_with_waves(lat, lon)
        if not cond: return {"summary": "No live wave data available nearby.", "station": station}

    # Jax realism: subtract 1.0 ft in text
    h_adj = max((cond.get("wave_height_ft") or 0) - 1.0, 0.0)
    parts = [f"{h_adj:.1f} ft (adj)"]
    if cond.get("wave_period_s") is not None: parts.append(f"@ {cond['wave_period_s']} s")
    if cond.get("wind_speed_mph") is not None: parts.append(f"wind {cond['wind_speed_mph']} mph {cond.get('wind_dir_txt') or ''}")
    text = ", ".join(parts) + f" — {station['name']} ({station['id']})"
    return {"summary": text, "station": station}

@app.get("/forecast-image")
def forecast_image(request: Request, lat: float, lon: float, station_id: Optional[str] = Query(None), force: int = Query(0)):
    cache_key = None
    now_ts = datetime.utcnow().timestamp()
    if not station_id:
        cache_key = (round(lat, 3), round(lon, 3))
        if not force and cache_key in _ai_cache and now_ts - _ai_cache[cache_key]["ts"] < AI_CACHE_TTL_SECONDS:
            return _ai_cache[cache_key]["data"]

    # Buoy + weather
    if station_id:
        cond = fetch_noaa_conditions(station_id)
        station = {"id": station_id, "name": f"NOAA {station_id}", "lat": lat, "lon": lon}
    else:
        station, cond = find_nearest_station_with_waves(lat, lon)
    weather = fetch_current_weather(lat, lon)
    weather_desc = describe_weather(weather)

    if not cond:
        data = {
            "summary": "No live wave data available nearby.",
            "imageUrl": make_abs(request, "/static/forecast.jpg"),
            "station": station,
            "imageProvider": "placeholder",
            "fallback_reason": "no_buoy_wave_data",
            "weather": weather, "weather_desc": weather_desc,
        }
        if cache_key: _ai_cache[cache_key] = {"data": data, "ts": now_ts}
        return data

    # Adjusted realism for Jax AI image: subtract 1.5 ft (looks smaller/softer) and always bias messy/choppy
    h_adj = max((cond.get("wave_height_ft") or 0) - 1.5, 0.0)
    wdir = cond.get("wind_dir_deg")
    wtxt = cond.get("wind_dir_txt") or ""
    ws = cond.get("wind_speed_mph") or 0

    parts = [f"{h_adj:.1f} ft (adj)"]
    if cond.get("wave_period_s") is not None: parts.append(f"@ {cond['wave_period_s']} s")
    if ws: parts.append(f"wind {ws} mph {wtxt}")
    base_summary = ", ".join(parts) + f" — {station['name']}"

    # Explicit messy surface — no surfer/glamour language
    surface = "wind‑blown chop, short‑period waves, disorganized peaks, whitewater, lumpy surface"

    prompt = (
        f"Realistic ocean photograph near {station['name']}; "
        f"{weather_desc}; "
        f"small to medium wave faces around {h_adj:.1f} feet"
        + (f" at {cond['wave_period_s']} seconds" if cond.get('wave_period_s') is not None else "")
        + (f"; wind {ws} mph {wtxt}" if ws else "")
        + f"; surface condition: {surface}. "
          "No people, no surfer, no boards. Unedited, no filters, neutral color, muted tones, true-to-life water, 16:9 frame."
    )

    img_rel, ai_err = stability_ai_image(prompt)
    provider = "stability" if img_rel else "placeholder"
    if not img_rel: img_rel = "/static/forecast.jpg"

    data = {
        "summary": base_summary,
        "imageUrl": make_abs(request, img_rel),
        "station": station,
        "imageProvider": provider,
        "fallback_reason": ai_err,
        "weather": weather, "weather_desc": weather_desc,
        "prompt_used": prompt
    }
    if cache_key: _ai_cache[cache_key] = {"data": data, "ts": now_ts}
    return data

@app.get("/wind")
def wind(lat: float, lon: float):
    station, cond = find_nearest_station_with_waves(lat, lon)
    # Fallback to Open‑Meteo if buoy missing wind
    ow = fetch_current_weather(lat, lon) or {}
    om_wspd = ow.get("wind10m")
    om_wdir = ow.get("winddir10m")
    if not cond:
        return {
            "station": station,
            "wind_speed_mph": round((om_wspd or 0), 1) if om_wspd is not None else None,
            "wind_dir_deg": om_wdir,
            "wind_dir_txt": deg_to_cardinal(om_wdir) if om_wdir is not None else None,
            "offshore": is_offshore_east_coast(om_wdir),
            "source": "open-meteo",
        }
    wspd = cond.get("wind_speed_mph")
    wdir = cond.get("wind_dir_deg")
    if wspd is None or wdir is None:
        wspd = round((om_wspd or 0), 1) if om_wspd is not None else None
        wdir = om_wdir
        src = "open-meteo"
    else:
        src = "noaa"
    return {
        "station": station,
        "wind_speed_mph": wspd,
        "wind_dir_deg": wdir,
        "wind_dir_txt": deg_to_cardinal(wdir) if wdir is not None else None,
        "offshore": is_offshore_east_coast(wdir),
        "source": src,
    }

@app.get("/optimal-window")
def optimal_window(lat: float, lon: float, tide_station: str = "8720291"):
    # Tide window helpers
    def _get(url: str):
        try:
            r = requests.get(url, timeout=12); r.raise_for_status(); return r.json()
        except Exception as e:
            print("[tides] fetch error:", e); return None
    def _hilo(ts: str):
        today = date.today().strftime("%Y%m%d")
        url = ("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?"
               f"product=predictions&application=swellintel&begin_date={today}&end_date={today}"
               f"&datum=MLLW&station={ts}&time_zone=lst_ldt&units=english&interval=hilo&format=json")
        j = _get(url); return j.get("predictions") if j else None
    def _hourly(ts: str):
        today = date.today().strftime("%Y%m%d")
        url = ("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?"
               f"product=predictions&application=swellintel&begin_date={today}&end_date={today}"
               f"&datum=MLLW&station={ts}&time_zone=lst_ldt&units=english&interval=h&format=json")
        j = _get(url); return j.get("predictions") if j else None
    def _from_hilo(preds):
        if not preds: return None
        fmt = "%Y-%m-%d %H:%M"; now = datetime.now()
        lows = [p for p in preds if p.get("type") == "L"]
        if not lows: return None
        nxt = None
        for p in lows:
            t = datetime.strptime(p["t"], fmt)
            if t >= now: nxt = t; break
        if not nxt: nxt = datetime.strptime(lows[-1]["t"], fmt)
        return {"start": nxt.isoformat(), "end": (nxt + timedelta(hours=2)).isoformat()}
    def _from_hourly(preds):
        if not preds: return None
        fmt = "%Y-%m-%d %H:%M"; now = datetime.now()
        series = []
        for p in preds:
            try: t = datetime.strptime(p["t"], fmt); v = float(p["v"]); series.append((t,v))
            except: pass
        if len(series) < 3: return None
        mins = []
        for i in range(1,len(series)-1):
            _, v0 = series[i-1]; t1, v1 = series[i]; _, v2 = series[i+1]
            if v1 <= v0 and v1 <= v2: mins.append(t1)
        nxt = None
        for t in mins:
            if t >= now: nxt = t; break
        if not nxt: nxt = mins[-1]
        return {"start": nxt.isoformat(), "end": (nxt + timedelta(hours=2)).isoformat()}

    window = _from_hilo(_hilo(tide_station)) or _from_hourly(_hourly(tide_station)) or {"start": None, "end": None}

    station, cond = find_nearest_station_with_waves(lat, lon)
    # Prefer buoy wind; fallback to Open‑Meteo so UI never blank
    ow = fetch_current_weather(lat, lon) or {}
    wspd = cond.get("wind_speed_mph") if cond else None
    wdir = cond.get("wind_dir_deg") if cond else None
    if wspd is None or wdir is None:
        wspd = round((ow.get("wind10m") or 0), 1) if ow.get("wind10m") is not None else None
        wdir = ow.get("winddir10m")
    offshore = is_offshore_east_coast(wdir)

    # Jax realism note logic (uses −1.0 ft for user-facing call)
    h_adj = None
    if cond and cond.get("wave_height_ft") is not None:
        h_adj = max(cond["wave_height_ft"] - 1.0, 0.0)
    if window.get("start"):
        if h_adj is not None and h_adj > 3:
            note = "Go shred, it's good!" if offshore else "Go to work."
        else:
            note = "Surf's small, maybe work."
    else:
        note = "No explicit low found; check tide station or try another nearby."

    return {
        "tide_station": tide_station,
        "window": window,
        "wind": {
            "dir_deg": wdir,
            "dir_txt": deg_to_cardinal(wdir) if wdir is not None else None,
            "speed_mph": wspd,
            "offshore": offshore,
        },
        "buoy_station": station,
        "note": note,
    }

@app.get("/weekly")
def weekly(lat: float, lon: float, days: int = 5):
    return {"days": fetch_weekly(lat, lon, days)}

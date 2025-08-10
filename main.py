# main.py — v3.2.0: adds POST JSON endpoints for strike/weekly to avoid query parsing issues
import os, io, math, requests
from datetime import datetime, date, timedelta, timezone
from typing import Optional, List, Dict
from fastapi import FastAPI, Query, Request, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image, ExifTags
from dateutil import parser as dtparse

# ---------- Config ----------
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")  # optional
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
BUCKET_PHOTOS = os.getenv("SUPABASE_BUCKET_PHOTOS", "photos")
BUCKET_PREVIEWS = os.getenv("SUPABASE_BUCKET_PREVIEWS", "previews")

app = FastAPI(title="Swell Intel Backend", version="3.2.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---------- Utils ----------
def clamp(v, lo, hi): return max(lo, min(hi, v))

def make_abs(request: Request, path: str) -> str:
    base = str(request.base_url).rstrip("/")
    return f"{base}{path if path.startswith('/') else '/' + path}"

def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def deg_to_cardinal(deg: Optional[float]) -> Optional[str]:
    if deg is None: return None
    dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
    ix = int((deg % 360) / 22.5 + 0.5) % 16
    return dirs[ix]

def is_offshore_east_coast(wdir: Optional[float]) -> bool:
    if wdir is None: return False
    return 210 <= (wdir % 360) <= 330  # WSW..NNW

def _f(x):
    try:
        if x is None: return None
        if isinstance(x, str) and x.upper() == "MM": return None
        return float(x)
    except:
        return None

# ---------- NOAA + Open-Meteo ----------
def get_noaa_stations():
    url = "https://www.ndbc.noaa.gov/activestations.xml"
    r = requests.get(url, timeout=12); r.raise_for_status()
    from xml.etree import ElementTree
    root = ElementTree.fromstring(r.content)
    out = []
    for s in root.findall("station"):
        try:
            out.append({"id": s.attrib["id"], "lat": float(s.attrib["lat"]), "lon": float(s.attrib["lon"]), "name": s.attrib.get("name","")})
        except: pass
    return out

def fetch_noaa_conditions(station_id: str) -> Optional[dict]:
    url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt"
    r = requests.get(url, timeout=12)
    if r.status_code != 200: return None
    lines = r.text.splitlines()
    if len(lines) < 3: return None
    header = lines[0].split(); latest = lines[2].split()
    idx = {h:i for i,h in enumerate(header)}
    wvht_m = _f(latest[idx["WVHT"]]) if "WVHT" in idx else None
    apd = _f(latest[idx["APD"]]) if "APD" in idx else None
    dpd = _f(latest[idx["DPD"]]) if "DPD" in idx else None
    wspd = _f(latest[idx["WSPD"]]) if "WSPD" in idx else None  # knots
    wdir = _f(latest[idx["WDIR"]]) if "WDIR" in idx else None
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

def find_nearest_station_with_waves(lat: float, lon: float):
    stations = get_noaa_stations()
    stations.sort(key=lambda s: haversine(lat, lon, s["lat"], s["lon"]))
    for s in stations:
        cond = fetch_noaa_conditions(s["id"])
        if cond: return s, cond
    return stations[0] if stations else None, None

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
        print("[weather] fetch error:", e); return None

# ---------- Weekly ----------
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
                h_ft_adj = max(h_ft - 1.0, 0.0)  # Jax bias
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
        return []
    except Exception as e:
        print("[weekly] error:", e); return []

# ---------- Strike Window ----------
def fetch_tides_hourly_and_hilo(station_id: str):
    today = date.today()
    y = today.year; m = f"{today.month:02d}"; d = f"{today.day:02d}"
    begin = f"{y}{m}{d}"; end = begin
    base = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    q = f"application=swellintel&datum=MLLW&station={station_id}&time_zone=lst_ldt&units=english&format=json"
    rH = requests.get(f"{base}?product=predictions&begin_date={begin}&end_date={end}&interval=h&{q}", timeout=12).json()
    rL = requests.get(f"{base}?product=predictions&begin_date={begin}&end_date={end}&interval=hilo&{q}", timeout=12).json()
    hourly = [{"t": dtparse.parse(p["t"]), "v": float(p["v"])} for p in rH.get("predictions", [])]
    hilo = [{"t": dtparse.parse(p["t"]), "v": float(p["v"]), "type": p["type"]} for p in rL.get("predictions", [])]
    return hourly, hilo

@app.get("/optimal-window")
def optimal_window(lat: float, lon: float, tide_station: str):
    return compute_optimal(lat, lon, tide_station)

class StrikeBody(BaseModel):
    lat: float
    lon: float
    tide_station: str

@app.post("/optimal-window-json")
def optimal_window_json(body: StrikeBody):
    return compute_optimal(body.lat, body.lon, body.tide_station)

def compute_optimal(lat: float, lon: float, tide_station: str):
    station, cond = find_nearest_station_with_waves(lat, lon)
    wind_deg = cond.get("wind_dir_deg") if cond else None
    wind_txt = cond.get("wind_dir_txt") if cond else None
    wind_mph = cond.get("wind_speed_mph") if cond else None
    offshore = is_offshore_east_coast(wind_deg)

    try:
        hourly, hilo = fetch_tides_hourly_and_hilo(tide_station)
        now = datetime.now().astimezone()
        next_low  = next((h for h in hilo if h["type"] == "L" and h["t"] >= now), None)
        next_high = next((h for h in hilo if h["type"] == "H" and (not next_low or h["t"] > next_low["t"]) ), None)
        window = None
        if next_low and next_high:
            start = next_low["t"]
            end   = min(next_low["t"] + timedelta(hours=2), next_high["t"])
            window = {"start": start.isoformat(), "end": end.isoformat()}
        note = None
        if offshore and wind_mph and wind_mph >= 5:
            note = "go shred — it’s good" if (cond and (cond.get("wave_height_ft") or 0) >= 3) else "promising: offshore winds"
        else:
            if cond and (cond.get("wave_height_ft") or 0) >= 3:
                note = "go to work"
            else:
                note = "watch for tide push"

        return {"window": window, "wind": {"dir_deg": wind_deg, "dir_txt": wind_txt, "speed_mph": wind_mph, "offshore": offshore}, "note": note}
    except Exception as e:
        print("[optimal] error:", e)
        return {"window": None, "wind": {"dir_deg": wind_deg, "dir_txt": wind_txt, "speed_mph": wind_mph, "offshore": offshore}, "note": "no strike window computed"}

@app.get("/weekly")
def weekly(lat: float, lon: float, days: int = 5):
    return {"days": fetch_weekly(lat, lon, days)}

class WeeklyBody(BaseModel):
    lat: float
    lon: float
    days: int = 5

@app.post("/weekly-json")
def weekly_json(body: WeeklyBody):
    return {"days": fetch_weekly(body.lat, body.lon, body.days)}

# ---------- Summary / Image ----------
@app.get("/summary")
def summary(lat: float, lon: float):
    station, cond = find_nearest_station_with_waves(lat, lon)
    if not cond:
        return {"summary": "No live wave data available nearby.", "station": station}
    h_adj = max((cond.get("wave_height_ft") or 0) - 1.0, 0.0)
    parts = [f"{h_adj:.1f} ft (adj)"]
    if cond.get("wave_period_s") is not None: parts.append(f"@ {cond['wave_period_s']} s")
    if cond.get("wind_speed_mph") is not None: parts.append(f"wind {cond['wind_speed_mph']} mph {cond.get('wind_dir_txt') or ''}")
    text = ", ".join(parts) + f" — {station['name']} ({station['id']})"
    return {"summary": text, "station": station}

@app.get("/forecast-image")
def forecast_image(request: Request, lat: float, lon: float):
    station, cond = find_nearest_station_with_waves(lat, lon)
    img = make_abs(request, "/static/forecast.jpg")
    base = "No live wave data available nearby." if not cond else f"~{max((cond['wave_height_ft'] or 0)-1.5,0):.1f} ft — {station['name']}"
    return {"summary": base, "imageUrl": img, "station": station, "imageProvider": "placeholder"}

@app.get("/health")
def health(): return {"ok": True, "ts": datetime.utcnow().isoformat()}

# ---------- Supabase upload/match (unchanged) ----------
# ... (the rest of the uploader, storage helpers, /upload page, /photos/upload, /matches/today) ...
# For brevity here, keep the same code you already have in your deployed main.py for these endpoints.

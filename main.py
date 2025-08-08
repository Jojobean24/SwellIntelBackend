import os
import math
from datetime import datetime
from typing import Dict, Tuple, Optional

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ====== Config ======
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")  # optional; we fallback to placeholder if missing
STATIC_DIR = "static"
AI_CACHE_TTL_SECONDS = 1800  # 30 minutes

os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI(title="Swell Intel Backend", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Simple in-memory cache for AI images keyed by rounded lat/lon
_ai_cache: Dict[Tuple[float, float], Dict] = {}


# ====== Helpers ======
def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0  # km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(
        math.radians(lat2)
    ) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def get_noaa_stations():
    """Fetch active NOAA stations (XML) with lat/lon/name/id."""
    url = "https://www.ndbc.noaa.gov/activestations.xml"
    r = requests.get(url, timeout=12)
    r.raise_for_status()
    from xml.etree import ElementTree

    root = ElementTree.fromstring(r.content)
    stations = []
    for s in root.findall("station"):
        try:
            stations.append(
                {
                    "id": s.attrib["id"],
                    "lat": float(s.attrib["lat"]),
                    "lon": float(s.attrib["lon"]),
                    "name": s.attrib.get("name", ""),
                }
            )
        except Exception:
            continue
    return stations


def fetch_noaa_conditions(station_id: str) -> Optional[dict]:
    """
    Parse NDBC realtime2 text using header names (robust to column shifts).
    Returns dict with wave_height_ft, wave_period_s, wind_speed_mph if present.
    """
    url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt"
    r = requests.get(url, timeout=12)
    if r.status_code != 200:
        return None

    lines = r.text.splitlines()
    if len(lines) < 3:
        return None

    header = lines[0].split()
    # Required columns for a 'valid' wave observation
    needed = {"WVHT", "APD", "WSPD"}  # wave height (m), avg period (s), wind speed
    if not needed.issubset(set(header)):
        return None  # this station doesn't report waves

    latest = lines[2].split()
    try:
        idx_wvht = header.index("WVHT")
        idx_apd = header.index("APD")
        idx_wspd = header.index("WSPD")

        wave_height_m = float(latest[idx_wvht])
        wave_period_s = float(latest[idx_apd])
        wind_speed_val = float(latest[idx_wspd])
    except Exception:
        return None

    # Convert wind speed heuristically: if it's big, assume knots; else assume m/s
    wind_mph = round(wind_speed_val * 1.15078, 1) if wind_speed_val > 40 else round(wind_speed_val * 2.23694, 1)

    return {
        "wave_height_ft": round(wave_height_m * 3.281, 1),
        "wave_period_s": round(wave_period_s, 1),
        "wind_speed_mph": wind_mph,
    }


def find_nearest_station_with_waves(lat: float, lon: float, max_candidates: int = 15):
    """
    Sort stations by distance and return (station, cond) where cond is parsed data.
    Tries up to max_candidates; falls back to the nearest station if none report waves.
    """
    stations = get_noaa_stations()
    stations.sort(key=lambda s: haversine(lat, lon, s["lat"], s["lon"]))

    for s in stations[:max_candidates]:
        cond = fetch_noaa_conditions(s["id"])
        if cond:
            return s, cond

    # Fallback: return nearest station even without wave data
    return stations[0], None


def stability_ai_image(prompt: str) -> Optional[str]:
    """Generate an image with Stability.ai; save to /static; return web path."""
    if not STABILITY_API_KEY:
        print("[stability] STABILITY_API_KEY not set; returning None.")
        return None

    url = "https://api.stability.ai/v2beta/stable-image/generate/core"
    headers = {"Authorization": f"Bearer {STABILITY_API_KEY}"}
    files = {
        "prompt": (None, prompt),
        "output_format": (None, "png"),
        "aspect_ratio": (None, "16:9"),
    }

    try:
        r = requests.post(url, headers=headers, files=files, timeout=60)
        if r.status_code != 200:
            print("[stability] API error:", r.status_code, r.text[:300])
            return None
        fname = f"surf_{int(datetime.utcnow().timestamp())}.png"
        path = os.path.join(STATIC_DIR, fname)
        with open(path, "wb") as f:
            f.write(r.content)
        return f"/static/{fname}"
    except Exception as e:
        print("[stability] Exception:", e)
        return None


# ====== Endpoints ======
@app.get("/health")
def health():
    return {"ok": True, "ts": datetime.utcnow().isoformat()}


@app.get("/summary")
def summary(lat: float, lon: float):
    station, cond = find_nearest_station_with_waves(lat, lon)
    if not cond:
        print(f"[summary] No wave data from nearest candidates. Fallback: {station['id']} {station['name']}")
        return {"summary": "No live wave data available nearby.", "station": station}

    text = f"{cond['wave_height_ft']} ft @ {cond['wave_period_s']} s, wind {cond['wind_speed_mph']} mph — {station['name']} ({station['id']})"
    print(f"[summary] Using {station['id']} {station['name']} -> {text}")
    return {"summary": text, "station": station}


@app.get("/forecast-image")
def forecast_image(lat: float, lon: float):
    # 30‑min cache by rounded location
    key = (round(lat, 3), round(lon, 3))
    now_ts = datetime.utcnow().timestamp()
    if key in _ai_cache and now_ts - _ai_cache[key]["ts"] < AI_CACHE_TTL_SECONDS:
        return _ai_cache[key]["data"]

    station, cond = find_nearest_station_with_waves(lat, lon)
    if not cond:
        print(f"[image] No wave data from nearest candidates. Fallback: {station['id']} {station['name']}")
        data = {"summary": "No live wave data available nearby.", "imageUrl": "/static/forecast.jpg", "station": station}
        _ai_cache[key] = {"data": data, "ts": now_ts}
        return data

    summary_text = f"{cond['wave_height_ft']} ft @ {cond['wave_period_s']} s, wind {cond['wind_speed_mph']} mph — {station['name']}"
    prompt = (
        f"Photorealistic surf scene at {station['name']}, "
        f"waves {cond['wave_height_ft']} feet at {cond['wave_period_s']} seconds, "
        f"wind {cond['wind_speed_mph']} mph. Modern camera, coastal viewpoint, natural colors, 16:9."
    )

    image_path = stability_ai_image(prompt) or "/static/forecast.jpg"
    data = {"summary": summary_text, "imageUrl": image_path, "station": station}
    _ai_cache[key] = {"data": data, "ts": now_ts}

    print(f"[image] Generated for {station['id']} {station['name']} -> {image_path}")
    return data

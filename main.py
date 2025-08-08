import os
import math
from datetime import datetime
from typing import Dict, Tuple, Optional

import requests
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ====== Config ======
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")  # optional; placeholder used if missing or errors
STATIC_DIR = "static"
AI_CACHE_TTL_SECONDS = 1800  # 30 minutes

os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI(title="Swell Intel Backend", version="1.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# In-memory cache for AI images keyed by rounded lat/lon
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


def _float_or_none(x: str) -> Optional[float]:
    # NDBC uses 'MM' for missing; return None in that case
    try:
        if x is None:
            return None
        if isinstance(x, str) and x.upper() == "MM":
            return None
        return float(x)
    except Exception:
        return None


def fetch_noaa_conditions(station_id: str) -> Optional[dict]:
    """
    Parse NDBC realtime2 text using header names, tolerant to 'MM'.
    Returns dict with wave_height_ft, wave_period_s (APD or DPD), wind_speed_mph if available.
    Returns None only if *no* usable wave height exists.
    """
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

    # Need at least wave height to consider it usable
    if "WVHT" not in idx:
        return None

    wvht_m = _float_or_none(latest[idx["WVHT"]]) if idx.get("WVHT") is not None else None
    if wvht_m is None:
        return None  # no wave height, bail

    # Prefer APD; fallback to DPD
    apd = _float_or_none(latest[idx["APD"]]) if "APD" in idx else None
    dpd = _float_or_none(latest[idx["DPD"]]) if "DPD" in idx else None
    period_s = apd if apd is not None else dpd

    # Wind speed: many feeds are m/s, some knots; heuristic conversion
    wspd_raw = _float_or_none(latest[idx["WSPD"]]) if "WSPD" in idx else None
    if wspd_raw is None:
        wind_mph = None
    else:
        wind_mph = round(wspd_raw * 1.15078, 1) if wspd_raw > 40 else round(wspd_raw * 2.23694, 1)

    return {
        "wave_height_ft": round(wvht_m * 3.281, 1),
        "wave_period_s": round(period_s, 1) if period_s is not None else None,
        "wind_speed_mph": wind_mph,
    }


def find_nearest_station_with_waves(lat: float, lon: float, max_candidates: int = 300):
    """
    Prefer numeric NDBC buoys (IDs like '41012'), then others.
    Try many nearest stations and return the first with usable wave height.
    If none report waves, fallback to nearest (no waves).
    """
    stations = get_noaa_stations()
    stations.sort(key=lambda s: haversine(lat, lon, s["lat"], s["lon"]))

    numeric = [s for s in stations if s["id"].isdigit()]
    non_numeric = [s for s in stations if not s["id"].isdigit()]

    tried = 0
    for s in numeric:
        if tried >= max_candidates:
            break
        tried += 1
        cond = fetch_noaa_conditions(s["id"])
        if cond:
            return s, cond

    for s in non_numeric[:max_candidates]:
        cond = fetch_noaa_conditions(s["id"])
        if cond:
            return s, cond

    return stations[0], None  # fallback only

def stability_ai_image(prompt: str) -> Optional[str]:
    """Generate an image with Stability.ai; save to /static; return web path."""
    if not STABILITY_API_KEY:
        print("[stability] STABILITY_API_KEY not set; using placeholder image.")
        return None

    url = "https://api.stability.ai/v2beta/stable-image/generate/core"
    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Accept": "image/*",            # REQUIRED
    }

    # Stability's v2beta core endpoint expects multipart form fields; include a model
    data = {
        "prompt": prompt,
        "model": "sd3.5-large",         # <- explicit model is safest
        "output_format": "png",
        "aspect_ratio": "16:9",
        # "style_preset": "photographic",  # optional, you can uncomment
    }

    try:
        # Send as multipart/form-data using 'data=' (text fields) without 'files'
        r = requests.post(url, headers=headers, data=data, timeout=60)
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


def make_abs(request: Request, path: str) -> str:
    base = str(request.base_url).rstrip("/")
    return f"{base}{path if path.startswith('/') else '/' + path}"


# ====== Endpoints ======
@app.get("/health")
def health():
    return {"ok": True, "ts": datetime.utcnow().isoformat()}


@app.get("/summary")
def summary(
    lat: float,
    lon: float,
    station_id: Optional[str] = Query(None, description="Override NOAA station ID (e.g., 41012)"),
):
    if station_id:
        cond = fetch_noaa_conditions(station_id)
        station = {"id": station_id, "name": f"NOAA {station_id}", "lat": lat, "lon": lon}
        if not cond:
            print(f"[summary] Override {station_id} -> no wave data")
            return {"summary": f"No live wave data on station {station_id}.", "station": station}

        parts = [f"{cond['wave_height_ft']} ft"]
        if cond.get("wave_period_s") is not None:
            parts.append(f"@ {cond['wave_period_s']} s")
        if cond.get("wind_speed_mph") is not None:
            parts.append(f"wind {cond['wind_speed_mph']} mph")
        text = ", ".join(parts) + f" — {station['name']} ({station['id']})"
        print(f"[summary] Override {station_id} -> {text}")
        return {"summary": text, "station": station}

    station, cond = find_nearest_station_with_waves(lat, lon)
    if not cond:
        print(f"[summary] No wave data from candidates. Fallback: {station['id']} {station['name']}")
        return {"summary": "No live wave data available nearby.", "station": station}

    parts = [f"{cond['wave_height_ft']} ft"]
    if cond.get("wave_period_s") is not None:
        parts.append(f"@ {cond['wave_period_s']} s")
    if cond.get("wind_speed_mph") is not None:
        parts.append(f"wind {cond['wind_speed_mph']} mph")
    text = ", ".join(parts) + f" — {station['name']} ({station['id']})"
    print(f"[summary] Using {station['id']} {station['name']} -> {text}")
    return {"summary": text, "station": station}


@app.get("/forecast-image")
def forecast_image(
    request: Request,
    lat: float,
    lon: float,
    station_id: Optional[str] = Query(None, description="Override NOAA station ID (e.g., 41012)"),
):
    # Cache by rounded lat/lon when not overriding
    key = None
    now_ts = datetime.utcnow().timestamp()
    if not station_id:
        key = (round(lat, 3), round(lon, 3))
        if key in _ai_cache and now_ts - _ai_cache[key]["ts"] < AI_CACHE_TTL_SECONDS:
            return _ai_cache[key]["data"]

    # Station selection
    if station_id:
        cond = fetch_noaa_conditions(station_id)
        station = {"id": station_id, "name": f"NOAA {station_id}", "lat": lat, "lon": lon}
    else:
        station, cond = find_nearest_station_with_waves(lat, lon)

    if not cond:
        print(f"[image] No wave data from candidates. Fallback: {station['id']} {station.get('name','')}")
        data = {
            "summary": "No live wave data available nearby.",
            "imageUrl": make_abs(request, "/static/forecast.jpg"),
            "station": station,
        }
        if key:
            _ai_cache[key] = {"data": data, "ts": now_ts}
        return data

    parts = [f"{cond['wave_height_ft']} ft"]
    if cond.get("wave_period_s") is not None:
        parts.append(f"@ {cond['wave_period_s']} s")
    if cond.get("wind_speed_mph") is not None:
        parts.append(f"wind {cond['wind_speed_mph']} mph")
    summary_text = ", ".join(parts) + f" — {station['name']}"

    prompt = (
        f"Photorealistic surf scene at {station['name']}, "
        f"waves {cond['wave_height_ft']} feet"
        + (f" at {cond['wave_period_s']} seconds" if cond.get('wave_period_s') is not None else "")
        + (f", wind {cond['wind_speed_mph']} mph" if cond.get('wind_speed_mph') is not None else "")
        + ". Natural colors, coastal perspective, 16:9."
    )

    image_rel = stability_ai_image(prompt) or "/static/forecast.jpg"
    data = {"summary": summary_text, "imageUrl": make_abs(request, image_rel), "station": station}
    if key:
        _ai_cache[key] = {"data": data, "ts": now_ts}

    print(f"[image] Generated for {station['id']} {station.get('name','')} -> {data['imageUrl']}")
    return data

import os
import math
import requests
from datetime import datetime
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Read API key from Render Environment
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
if not STABILITY_API_KEY:
    raise RuntimeError("Missing STABILITY_API_KEY in environment variables.")

app = FastAPI(title="Swell Intel Backend", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Simple cache for AI image results ---
cache = {}  # {(lat_rounded, lon_rounded): {"data": {...}, "ts": timestamp}}

# --- Helper functions ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def get_noaa_stations():
    url = "https://www.ndbc.noaa.gov/activestations.xml"
    r = requests.get(url, timeout=10)
    from xml.etree import ElementTree
    root = ElementTree.fromstring(r.content)
    stations = []
    for s in root.findall("station"):
        try:
            stations.append({
                "id": s.attrib["id"],
                "lat": float(s.attrib["lat"]),
                "lon": float(s.attrib["lon"]),
                "name": s.attrib.get("name", "")
            })
        except:
            continue
    return stations

def find_nearest_station(lat, lon):
    stations = get_noaa_stations()
    nearest = min(stations, key=lambda s: haversine(lat, lon, s["lat"], s["lon"]))
    return nearest

def fetch_noaa_conditions(station_id):
    url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt"
    r = requests.get(url, timeout=10)
    lines = r.text.splitlines()
    if len(lines) < 3:
        return None
    try:
        data = lines[2].split()
        wave_height_m = float(data[8])
        wave_period_s = float(data[9])
        wind_speed_kt = float(data[6])
        return {
            "wave_height_ft": round(wave_height_m * 3.281, 1),
            "wave_period_s": wave_period_s,
            "wind_speed_mph": round(wind_speed_kt * 1.15078, 1)
        }
    except:
        return None

def stability_ai_image(prompt):
    url = "https://api.stability.ai/v2beta/stable-image/generate/core"
    headers = {"Authorization": f"Bearer {STABILITY_API_KEY}"}
    files = {
        "prompt": (None, prompt),
        "output_format": (None, "png"),
        "aspect_ratio": (None, "16:9")
    }
    r = requests.post(url, headers=headers, files=files, timeout=60)
    if r.status_code != 200:
        print("Stability API error:", r.text)
        return None
    filename = f"static/surf_{int(datetime.utcnow().timestamp())}.png"
    with open(filename, "wb") as f:
        f.write(r.content)
    return f"/{filename}"

# --- Endpoints ---
@app.get("/health")
def health():
    return {"ok": True, "ts": datetime.utcnow().isoformat()}

@app.get("/summary")
def summary(lat: float, lon: float):
    station = find_nearest_station(lat, lon)
    cond = fetch_noaa_conditions(station["id"])
    if not cond:
        return {"summary": "No live data available."}
    return {
        "summary": f"{cond['wave_height_ft']} ft swell at {cond['wave_period_s']} sec, wind {cond['wind_speed_mph']} mph, station {station['name']} ({station['id']})",
        "station": station
    }

@app.get("/forecast-image")
def forecast_image(lat: float, lon: float):
    key = (round(lat, 3), round(lon, 3))
    now_ts = datetime.utcnow().timestamp()

    # Serve from cache if fresh
    if key in cache and now_ts - cache[key]['ts'] < 1800:  # 30 minutes
        return cache[key]['data']

    station = find_nearest_station(lat, lon)
    cond = fetch_noaa_conditions(station["id"])
    if not cond:
        return {"summary": "No live data", "imageUrl": "/static/forecast.jpg"}

    summary_text = f"{cond['wave_height_ft']} ft swell at {cond['wave_period_s']} sec, wind {cond['wind_speed_mph']} mph, location {station['name']}"
    prompt = f"Photorealistic surf scene at {station['name']}, waves {cond['wave_height_ft']} ft at {cond['wave_period_s']} sec, wind {cond['wind_speed_mph']} mph"

    image_path = stability_ai_image(prompt) or "/static/forecast.jpg"
    data = {"summary": summary_text, "imageUrl": image_path, "station": station}

    cache[key] = {"data": data, "ts": now_ts}
    return data

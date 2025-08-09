import os
import io
import math
import base64
import requests
from datetime import datetime, date, timedelta, timezone
from typing import Dict, Tuple, Optional, List

from fastapi import FastAPI, Query, Request, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image, ExifTags
from dateutil import parser as dtparse

# =========================
# Config & env
# =========================
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")  # optional for AI image
STATIC_DIR = "static"
AI_CACHE_TTL_SECONDS = 1800  # 30 min
os.makedirs(STATIC_DIR, exist_ok=True)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
BUCKET_PHOTOS = os.getenv("SUPABASE_BUCKET_PHOTOS", "photos")
BUCKET_PREVIEWS = os.getenv("SUPABASE_BUCKET_PREVIEWS", "previews")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("[supabase] WARN: missing SUPABASE_URL or SUPABASE_SERVICE_KEY")

app = FastAPI(title="Swell Intel Backend", version="3.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_ai_cache: Dict[Tuple[float, float], Dict] = {}

# =========================
# Helpers
# =========================
def make_abs(request: Request, path: str) -> str:
    base = str(request.base_url).rstrip("/")
    return f"{base}{path if path.startswith('/') else '/' + path}"

def clamp(v, lo, hi): return max(lo, min(hi, v))

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
    return 210 <= (wdir % 360) <= 330

def is_onshore_east_coast(wdir: Optional[float]) -> bool:
    if wdir is None: return False
    return 30 <= (wdir % 360) <= 150

def _float_or_none(x):
    try:
        if x is None: return None
        if isinstance(x, str) and x.upper() == "MM": return None
        return float(x)
    except:
        return None

# =========================
# NOAA + Open-Meteo (existing surf logic)
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
                h_ft_adj = max(h_ft - 1.0, 0.0)
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
        print("[weekly] error:", e)
        return []

# =========================
# Stability AI image (kept, with messy bias)
# =========================
def stability_ai_image(prompt: str):
    if not STABILITY_API_KEY:
        return None, "STABILITY_API_KEY not set"
    url = "https://api.stability.ai/v2beta/stable-image/generate/core"
    headers = {"Authorization": f"Bearer {STABILITY_API_KEY}", "Accept": "image/*"}
    files = {
        "prompt": (None, prompt),
        "model": (None, "sd3.5"),
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
# Supabase helpers
# =========================
def sb_headers(json=True):
    h = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
    }
    if json: h["Content-Type"] = "application/json"
    return h

def supabase_storage_upload(bucket: str, path: str, data: bytes, content_type: str) -> str:
    url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{path}"
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": content_type,
        "x-upsert": "true",
    }
    r = requests.post(url, headers=headers, data=data, timeout=60)
    if r.status_code not in (200, 201):
        raise HTTPException(500, f"Supabase upload failed: {r.status_code} {r.text[:200]}")
    # Public URL pattern for storage (previews should be in a public bucket)
    return f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{path}"

def supabase_storage_upload_private(bucket: str, path: str, data: bytes, content_type: str) -> str:
    # Upload to private bucket (no /public in URL)
    url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{path}"
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": content_type,
        "x-upsert": "true",
    }
    r = requests.post(url, headers=headers, data=data, timeout=60)
    if r.status_code not in (200, 201):
        raise HTTPException(500, f"Supabase upload failed: {r.status_code} {r.text[:200]}")
    # Return a storage path (not public). You can sign URLs later.
    return f"{bucket}/{path}"

def supabase_insert(table: str, payload: dict) -> dict:
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    headers = sb_headers(json=True)
    headers["Prefer"] = "return=representation"
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    if r.status_code not in (200, 201):
        raise HTTPException(500, f"Supabase insert failed: {r.status_code} {r.text[:200]}")
    return r.json()[0] if isinstance(r.json(), list) and r.json() else r.json()

def supabase_select_photos_today(limit: int = 500) -> List[dict]:
    today = date.today().isoformat()
    # photos created today (UTC)
    url = f"{SUPABASE_URL}/rest/v1/photos?select=*&created_at=gte.{today}T00:00:00Z&limit={limit}"
    r = requests.get(url, headers=sb_headers(json=False), timeout=30)
    if r.status_code != 200:
        print("[supabase] select photos failed:", r.status_code, r.text[:200])
        return []
    return r.json()

# =========================
# Uploader utils
# =========================
def exif_to_dict(img: Image.Image) -> dict:
    out = {}
    try:
        raw = img._getexif() or {}
        for k, v in raw.items():
            tag = ExifTags.TAGS.get(k, str(k))
            out[tag] = v
    except Exception:
        pass
    return out

def exif_get_datetime(exif: dict) -> Optional[datetime]:
    for key in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
        if key in exif:
            try:
                # EXIF format 'YYYY:MM:DD HH:MM:SS'
                s = exif[key]
                s = s.replace("-", ":") if "-" in s and ":" not in s[:10] else s
                dt = datetime.strptime(s, "%Y:%m:%d %H:%M:%S")
                return dt.replace(tzinfo=timezone.utc)  # assume UTC if no tz
            except Exception:
                try:
                    return dtparse.parse(exif[key])
                except Exception:
                    pass
    return None

def exif_get_gps(exif: dict) -> Tuple[Optional[float], Optional[float]]:
    gps = exif.get("GPSInfo")
    if not gps: return None, None

    def _conv(val):
        try:
            n, d = val
            return float(n) / float(d)
        except Exception:
            try:
                return float(val)
            except Exception:
                return None

    def _dms_to_deg(d, m, s, ref):
        deg = _conv(d) + _conv(m)/60.0 + _conv(s)/3600.0
        if ref in ["S", "W"]: deg = -deg
        return deg

    lat = lon = None
    try:
        lat = _dms_to_deg(gps[2][0], gps[2][1], gps[2][2], gps[1])
        lon = _dms_to_deg(gps[4][0], gps[4][1], gps[4][2], gps[3])
    except Exception:
        pass
    return lat, lon

def make_preview(image_bytes: bytes, max_w: int = 800) -> bytes:
    im = Image.open(io.BytesIO(image_bytes))
    im = im.convert("RGB")
    w, h = im.size
    if w > max_w:
        nh = int(h * (max_w / w))
        im = im.resize((max_w, nh), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=82)
    return buf.getvalue()

# =========================
# Routes — health
# =========================
@app.get("/health")
def health(): return {"ok": True, "ts": datetime.utcnow().isoformat()}

# =========================
# Routes — forecast & image (kept, messy bias)
# =========================
@app.get("/summary")
def summary(lat: float, lon: float, station_id: Optional[str] = Query(None)):
    if station_id:
        cond = fetch_noaa_conditions(station_id)
        station = {"id": station_id, "name": f"NOAA {station_id}", "lat": lat, "lon": lon}
        if not cond: return {"summary": f"No live wave data on station {station_id}.", "station": station}
    else:
        station, cond = find_nearest_station_with_waves(lat, lon)
        if not cond: return {"summary": "No live wave data available nearby.", "station": station}

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

    h_adj = max((cond.get("wave_height_ft") or 0) - 1.5, 0.0)
    wdir = cond.get("wind_dir_deg"); wtxt = cond.get("wind_dir_txt") or ""
    ws = cond.get("wind_speed_mph") or 0

    parts = [f"{h_adj:.1f} ft (adj)"]
    if cond.get("wave_period_s") is not None: parts.append(f"@ {cond['wave_period_s']} s")
    if ws: parts.append(f"wind {ws} mph {wtxt}")
    base_summary = ", ".join(parts) + f" — {station['name']}"

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

@app.get("/weekly")
def weekly(lat: float, lon: float, days: int = 5):
    return {"days": fetch_weekly(lat, lon, days)}

# =========================
# Uploader page & endpoints
# =========================
UPLOADER_HTML = """<!doctype html>
<html><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Swell Intel — Photo Uploader</title>
<style>
 body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Inter,Arial;background:#071B2F;color:#E5F0FF;margin:0;padding:24px}
 .card{max-width:720px;margin:0 auto;background:#0B2C4E;border:1px solid #123B66;border-radius:12px;padding:16px}
 h1{margin:0 0 12px;font-size:20px}
 label{display:block;margin-top:10px;font-size:14px;color:#9FC5FF}
 input,button{margin-top:6px}
 .btn{background:#2DD4BF;border:none;color:#062030;font-weight:700;padding:10px 14px;border-radius:8px;cursor:pointer}
 .muted{color:#9FC5FF;font-size:12px}
</style>
</head><body>
  <div class="card">
    <h1>Photo Uploader</h1>
    <p class="muted">Drop JPEG/PNG from your shoot. We’ll read EXIF for time & GPS (if present), create previews, and store them.</p>
    <form id="f" enctype="multipart/form-data" method="post" action="/photos/upload">
      <label>Photographer email/handle (for payouts/credit):</label>
      <input name="photographer" type="text" placeholder="you@example.com" required />
      <label>Fallback latitude (if EXIF GPS missing):</label>
      <input name="fallback_lat" type="text" placeholder="30.3200" />
      <label>Fallback longitude (if EXIF GPS missing):</label>
      <input name="fallback_lon" type="text" placeholder="-81.4000" />
      <label>Fallback Field of View (deg, optional):</label>
      <input name="fov_deg" type="text" placeholder="78" />
      <label>Select images:</label>
      <input name="files" type="file" accept="image/*" multiple required />
      <div style="margin-top:12px">
        <button class="btn" type="submit">Upload</button>
      </div>
    </form>
    <p id="res" class="muted"></p>
  </div>
<script>
  const form = document.getElementById('f');
  const resEl = document.getElementById('res');
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    resEl.textContent = 'Uploading...';
    const fd = new FormData(form);
    try {
      const r = await fetch('/photos/upload', { method: 'POST', body: fd });
      const j = await r.json();
      resEl.textContent = 'Done: ' + JSON.stringify(j, null, 2);
    } catch (err) {
      resEl.textContent = 'Error: ' + err.message;
    }
  });
</script>
</body></html>"""

@app.get("/upload", response_class=HTMLResponse)
def upload_page():
    return HTMLResponse(content=UPLOADER_HTML, status_code=200)

@app.post("/photos/upload")
async def photos_upload(
    request: Request,
    photographer: str = Form(...),
    fallback_lat: Optional[str] = Form(None),
    fallback_lon: Optional[str] = Form(None),
    fov_deg: Optional[str] = Form(None),
    files: List[UploadFile] = File(...)
):
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise HTTPException(500, "Supabase not configured")

    out = []
    for uf in files:
        content = await uf.read()
        # Preview
        preview_jpg = make_preview(content, max_w=800)

        # EXIF
        try:
            img = Image.open(io.BytesIO(content))
        except Exception:
            raise HTTPException(400, f"Unsupported image: {uf.filename}")
        exif = exif_to_dict(img)
        dt = exif_get_datetime(exif) or datetime.utcnow().replace(tzinfo=timezone.utc)
        lat, lon = exif_get_gps(exif)

        # Camera model → rough FOV if provided (optional)
        cam_model = exif.get("Model") or exif.get("Make") or ""
        fov = None
        try:
            fov = float(fov_deg) if fov_deg else None
        except:
            fov = None

        if lat is None or lon is None:
            try:
                if fallback_lat and fallback_lon:
                    lat = float(fallback_lat); lon = float(fallback_lon)
            except:
                pass

        # Storage paths
        ts = int(dt.timestamp())
        safe_name = os.path.basename(uf.filename).replace(" ", "_")
        photo_path = f"{photographer}/{ts}_{safe_name}"
        preview_path = f"{photographer}/{ts}_{os.path.splitext(safe_name)[0]}_preview.jpg"

        # Upload to Supabase Storage
        url_preview = supabase_storage_upload(BUCKET_PREVIEWS, preview_path, preview_jpg, "image/jpeg")
        # Private full image (store path; you can sign URL later)
        _full_path = supabase_storage_upload_private(BUCKET_PHOTOS, photo_path, content, uf.content_type or "application/octet-stream")
        url_full = _full_path  # stored as bucket/path

        # Insert row
        row = supabase_insert("photos", {
            "photographer": photographer,
            "taken_at": dt.isoformat(),
            "lat": lat,
            "lon": lon,
            "fov_deg": fov,
            "url_preview": url_preview,
            "url_full": url_full,
        })
        out.append(row)

    return {"inserted": len(out), "items": out}

# =========================
# Sessions + Matches
# =========================
@app.post("/sessions")
def create_session(
    user_email: str = Form(...),
    start_time: str = Form(...),  # ISO
    end_time: str = Form(...),    # ISO
    center_lat: float = Form(...),
    center_lon: float = Form(...),
    source: str = Form("manual")
):
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise HTTPException(500, "Supabase not configured")
    try:
        st = dtparse.parse(start_time)
        et = dtparse.parse(end_time)
    except Exception:
        raise HTTPException(400, "Invalid time format")
    row = supabase_insert("sessions", {
        "user_email": user_email,
        "start_time": st.isoformat(),
        "end_time": et.isoformat(),
        "center_lat": center_lat,
        "center_lon": center_lon,
        "source": source
    })
    return {"ok": True, "session": row}

@app.get("/matches/today")
def matches_today(lat: float, lon: float, radius_m: int = 400):
    """Return photos shot today within radius of given location."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise HTTPException(500, "Supabase not configured")
    photos = supabase_select_photos_today(limit=800)
    hits = []
    for p in photos:
        p_lat = p.get("lat"); p_lon = p.get("lon"); t = p.get("taken_at")
        if p_lat is None or p_lon is None or not t: continue
        try:
            d = haversine(lat, lon, float(p_lat), float(p_lon)) * 1000
        except Exception:
            continue
        if d <= radius_m:
            hits.append({
                "photo_id": p.get("id"),
                "url_preview": p.get("url_preview"),
                "taken_at": t,
                "distance_m": round(d, 1),
                "confidence": 0.8 if d < radius_m/2 else 0.6,  # naive score
                "photographer": p.get("photographer"),
            })
    # sort closest first
    hits.sort(key=lambda x: (x["distance_m"]))
    return {"count": len(hits), "items": hits}

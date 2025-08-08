
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from datetime import datetime

app = FastAPI(title="Swell Intel Backend", version="0.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/health")
def health():
    return {"ok": True, "ts": datetime.utcnow().isoformat()}

@app.get("/summary")
def summary(spot_id: str = Query("north-beach"), dt: str = Query(None)):
    return {
        "spotId": spot_id,
        "datetime": dt or datetime.utcnow().isoformat(),
        "summary": "Light offshore winds, chest-high sets, low tide. Best 7–10am."
    }

@app.get("/forecast-image")
def forecast_image(spot_id: str = Query("north-beach"), dt: str = Query(None)):
    return {
        "spotId": spot_id,
        "datetime": dt or datetime.utcnow().isoformat(),
        "imageUrl": "/static/forecast.jpg",
        "summary": "Photorealistic preview coming soon — placeholder image for MVP."
    }

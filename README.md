
# Swell Intel Backend (FastAPI)

## Local dev (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload

Open http://127.0.0.1:8000/health
Image: http://127.0.0.1:8000/static/forecast.jpg
API:   http://127.0.0.1:8000/forecast-image

## Deploy to Render (free)
- Create new Web Service from this folder/repo
- build: pip install -r requirements.txt
- start: uvicorn main:app --host 0.0.0.0 --port $PORT
- After deploy, use your public base URL in the mobile app's constants.js

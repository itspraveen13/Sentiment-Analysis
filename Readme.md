# Sentimetryx

Sentimetryx is a web app for sentiment analysis and related ML features (churn prediction, social listening, fake detection, meme analysis). The frontend is a Vite + React app and the backend is a Flask API.

**Prerequisites**
1. Node.js 18+ and npm
2. Python 3.11

**Install**
Frontend:
1. `npm install`

Backend:
1. `cd Backend`
2. `python -m venv .venv`
3. `.\.venv\Scripts\activate`
4. `pip install -r requirements.txt`

**Run**
1. Start backend: `npm run backend`
2. Start frontend: `npm run dev`

The backend runs on `http://localhost:5000` and the frontend on `http://localhost:5173` or `http://localhost:5174`.

**Key API Endpoints**
1. `POST /text_sentiment` JSON: `{ "text": "..." }`
2. `POST /file_sentiment` multipart: `file`
3. `POST /audio_sentiment` multipart: `file`
4. `POST /audio_live_sentiment` multipart: `file`
5. `POST /churn` multipart: `file`
6. `POST /social` JSON: `{ "keyword": "...", "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD", "count": 10 }`
7. `POST /fake_text` JSON: `{ "text": "..." }`
8. `POST /fake_file` multipart: `file`
9. `POST /meme` multipart: `file`

**Notes**
1. Some features require FFmpeg to be installed and available on PATH.
2. CORS is configured for `http://localhost:5173` and `http://localhost:5174`.

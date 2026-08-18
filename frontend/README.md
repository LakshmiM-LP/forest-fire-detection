# Forest Fire & Smoke Detection — Web Interface

React + Vite frontend for the FastAPI backend. Supports three modes:
image upload, video upload (async job with live log + result playback),
and live camera (WebSocket, real-time frame-by-frame detection).

## Setup

```
npm install
cp .env.example .env
```

Edit `.env` if your API isn't running on `http://localhost:8000`.

## Run (development)

```
npm run dev
```

Opens at `http://localhost:5173`. Make sure the FastAPI backend is
running separately (`uvicorn api.main:app --reload`) or via Docker.

## Build (production)

```
npm run build
```

Outputs to `dist/`. Serve it with any static host, or add an nginx
stage to the project's Dockerfile if you want it bundled into the
same container as the API later.

## Notes

- Live camera mode requires the browser tab to be served over
  `https://` or `localhost` — browsers block camera access on plain
  `http://` for any other host.
- The video mode polls `/video/status/{job_id}` every 2 seconds and
  expects the backend's `_run_video_job` to keep updating job state
  as `src/inference/video_temporal.py` runs.

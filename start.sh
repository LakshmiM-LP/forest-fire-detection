#!/bin/sh
set -e

# Start the FastAPI backend in the background
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Give it a moment to load the model before the dashboard
# starts sending requests to it
sleep 3

# Start the Streamlit dashboard in the foreground
# (keeps the container alive / lets Docker manage the process)
exec streamlit run dashboard/app.py --server.address=0.0.0.0 --server.port=8501
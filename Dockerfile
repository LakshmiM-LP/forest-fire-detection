FROM python:3.12-slim

WORKDIR /app

# Install FFmpeg and required system libraries
RUN apt-get update && \
    apt-get install -y ffmpeg libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
# NOTE: data/ is gitignored (raw datasets, uploaded videos) and is
# intentionally NOT copied here -- the container creates its own
# empty data/videos dir at runtime for uploads instead.
COPY api ./api
COPY dashboard ./dashboard
COPY src ./src
COPY models ./models
COPY start.sh ./start.sh

# Output / upload directories created at build time so the app
# doesn't need to create them on first request
RUN mkdir -p runs data/videos && chmod +x start.sh

# FastAPI + Streamlit ports
EXPOSE 8000 8501

# Start both the API and the dashboard
CMD ["./start.sh"]
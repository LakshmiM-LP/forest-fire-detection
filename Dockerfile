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
COPY api ./api
COPY dashboard ./dashboard
COPY src ./src
COPY models ./models
COPY data ./data

# Create output directory
RUN mkdir -p runs

# Streamlit port
EXPOSE 8501

# Start Streamlit
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
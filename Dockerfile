FROM python:3.10-slim

WORKDIR /app

#Installs deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Copies app code
COPY . .

#Container Apps will send traffic to this port
ENV PORT=8000
EXPOSE 8000

#Runs Flask via gunicorn (production server)
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT} dashboard.app:app"]

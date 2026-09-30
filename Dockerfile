FROM python:3.11-slim

WORKDIR /app

# scipy<1.17 pinned to honour lingam's constraint
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

# Offline-verifiable: pure-numpy fallbacks run without any optional backend
CMD ["python", "examples/run_demo.py"]

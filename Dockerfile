# Użyj bezpieczniejszego, lekkiego obrazu
FROM python:3.11.11-slim-bullseye

# Ustaw zmienne środowiskowe (np. dla pip)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Zainstaluj wymagane pakiety systemowe
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    gfortran \
    libatlas-base-dev \
    liblapack-dev \
    libblas-dev \
    libffi-dev \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Skopiuj pliki projektu    
COPY app.py .
COPY candle.gif .
COPY hm_model.pkl .
COPY requirements.txt .
COPY .env .

# Instaluj zależności Pythona
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Wystaw port Streamlit
EXPOSE 8501

# Domyślna komenda startowa
CMD ["streamlit", "run", "app.py"]
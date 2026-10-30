"""Погода из Open-Meteo (ключ не нужен)."""
from datetime import datetime, timedelta

import requests

import config

URL = "https://api.open-meteo.com/v1/forecast"


def get_weather(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "hourly": "precipitation",
        "past_days": 3,
        "forecast_days": 2,
        "wind_speed_unit": "ms",
        "timezone": "auto",
    }
    response = requests.get(URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    current = data["current"]
    now = datetime.fromisoformat(current["time"])
    hours = [datetime.fromisoformat(t) for t in data["hourly"]["time"]]
    rain = [r or 0.0 for r in data["hourly"]["precipitation"]]

    past = sum(r for t, r in zip(hours, rain) if now - timedelta(hours=72) <= t < now)
    ahead = sum(r for t, r in zip(hours, rain) if now < t <= now + timedelta(hours=6))

    return {
        "temp": round(current["temperature_2m"], 1),
        "humidity": round(current["relative_humidity_2m"]),
        "wind": round(current["wind_speed_10m"], 1),
        "rain_3d": past > config.RAIN_3D_MM,
        "rain_6h": ahead > config.RAIN_6H_MM,
    }

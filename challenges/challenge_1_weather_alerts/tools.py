"""Tool functions for the weather alerts agent."""

from typing import Dict, List, Optional

import httpx

# NWS 403s a blank User-Agent.
NWS_USER_AGENT = "adk-weather-alerts-agent"


def get_lat_lon(place: str) -> Optional[Dict[str, float]]:
    """Convert a place name to latitude/longitude.

    Uses Open-Meteo's geocoding API as no API key needed.

    Args:
        place: Free-text place name, e.g. "Washington, DC" or "Seattle, WA".

    Returns:
        Dict with "lat" and "lon" keys, or None if geocoding failed.
    """
    # "Washington, DC" finds nothing; "Washington" matches "Washington D.C."
    city = place.split(",")[0].strip()
    try:
        resp = httpx.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "country": "US"},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("results")
        if not results:
            return None
        return {"lat": float(results[0]["latitude"]), "lon": float(results[0]["longitude"])}
    except (httpx.HTTPError, KeyError, ValueError, IndexError):
        return None


def get_extended_weather_forecast(lat: float, lon: float) -> Optional[List[Dict[str, str]]]:
    """Fetch the extended weather forecast from the U.S. National Weather Service API
    based on a given latitude and longitude.

    Args:
        lat: Latitude of the location (e.g., 38.8977).
        lon: Longitude of the location (e.g., -77.0365).

    Returns:
        A list of forecast period dicts (name, detailedForecast, etc.), or None if
        data is unavailable or an error occurs.
    """
    headers = {"User-Agent": NWS_USER_AGENT}
    try:
        points_resp = httpx.get(
            f"https://api.weather.gov/points/{lat},{lon}",
            headers=headers,
            timeout=10,
            follow_redirects=True,
        )
        points_resp.raise_for_status()
        forecast_url = points_resp.json()["properties"]["forecast"]

        forecast_resp = httpx.get(
            forecast_url, headers=headers, timeout=10, follow_redirects=True
        )
        forecast_resp.raise_for_status()
        return forecast_resp.json()["properties"]["periods"]
    except (httpx.HTTPError, KeyError):
        return None

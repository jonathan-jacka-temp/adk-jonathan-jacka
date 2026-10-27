"""Tool functions for the ReadyNow! emergency assistant: geocoding, NWS weather, routing."""

from typing import Dict, List, Optional

import httpx

# NWS 403s a blank User-Agent.
NWS_USER_AGENT = "adk-readynow-agent"


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


def get_route(origin: str, destination: str) -> Optional[Dict[str, object]]:
    """Get a driving route between two US places, with turn-by-turn directions.

    Uses the public OSRM demo server rather than the Google Maps Directions API —
    no API key needed. The demo server is rate-limited and not for production use.

    Args:
        origin: Starting place name, e.g. "Washington, DC".
        destination: Destination place name, e.g. "Baltimore, MD".

    Returns:
        Dict with "distance_miles", "duration_minutes", and "steps" (a list of
        instruction strings), or None if either place can't be geocoded or no
        route is found.
    """
    start = get_lat_lon(origin)
    end = get_lat_lon(destination)
    if start is None or end is None:
        return None

    coords = f"{start['lon']},{start['lat']};{end['lon']},{end['lat']}"
    try:
        resp = httpx.get(
            f"https://router.project-osrm.org/route/v1/driving/{coords}",
            params={"overview": "false", "steps": "true"},
            timeout=15,
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("code") != "Ok" or not body.get("routes"):
            return None

        route = body["routes"][0]
        steps = [
            f"{s['maneuver']['type'].replace('_', ' ')} onto {s['name']}".strip()
            if s.get("name")
            else s["maneuver"]["type"].replace("_", " ")
            for s in route["legs"][0]["steps"]
        ]
        return {
            "distance_miles": round(route["distance"] / 1609.34, 1),
            "duration_minutes": round(route["duration"] / 60),
            "steps": steps,
        }
    except (httpx.HTTPError, KeyError, IndexError):
        return None

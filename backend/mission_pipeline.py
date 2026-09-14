"""Provider-based mission pipeline used by the dashboard.

The default replay provider is deliberately deterministic and clearly marked
as synthetic.  A live provider can be enabled only after its credentials and
data connector are configured; it must never silently fall back to replay
data.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from math import asin, cos, radians, sin, sqrt
import os
from pathlib import Path
from typing import Protocol


class ProviderNotConfigured(RuntimeError):
    """Raised when an operator requests a live provider without its access."""


@dataclass(frozen=True)
class Observation:
    id: str
    lat: float
    lon: float
    area_m2: float
    fdi: float
    confidence: float
    captured_at: str
    source: str
    verification: str


@dataclass(frozen=True)
class CurrentVector:
    u_ms: float
    v_ms: float
    source: str
    valid_at: str


class ObservationProvider(Protocol):
    def observations(self, scenario: str) -> list[Observation]: ...


class CurrentProvider(Protocol):
    def current_at(self, lat: float, lon: float) -> CurrentVector: ...


class ReplayObservationProvider:
    """Synthetic, repeatable mission data for demos and automated tests."""

    def observations(self, scenario: str) -> list[Observation]:
        if scenario != "Chennai - Pulicat pilot":
            raise ValueError(f"Unknown replay scenario: {scenario}")
        return [
            Observation("OBS-2401", 13.493, 80.516, 18_400, 0.048, 0.89,
                        "2026-08-24T04:20:00Z", "Synthetic replay scene", "drone-confirmed"),
            Observation("OBS-2402", 13.285, 80.742, 11_600, 0.036, 0.74,
                        "2026-08-24T04:20:00Z", "Synthetic replay scene", "awaiting verification"),
            Observation("OBS-2403", 12.866, 80.536, 7_900, 0.029, 0.62,
                        "2026-08-24T04:20:00Z", "Synthetic replay scene", "awaiting verification"),
        ]


class ReplayCurrentProvider:
    def current_at(self, lat: float, lon: float) -> CurrentVector:
        # A small spatial change makes the replay look like a gridded field,
        # while staying deterministic for a presentation and test suite.
        return CurrentVector(
            u_ms=round(-0.12 + (lat - 13.0) * 0.04, 3),
            v_ms=round(-0.07 + (lon - 80.5) * 0.06, 3),
            source="Synthetic replay current field",
            valid_at="2026-08-24T06:00:00Z",
        )


class EarthEngineObservationProvider:
    """Adapter for the existing GEE connector, enabled explicitly in live mode."""

    def observations(self, scenario: str) -> list[Observation]:
        from data.fetch_satellite import get_cloud_reduced_hotspots, init_gee

        if scenario != "Chennai - Pulicat pilot":
            raise ValueError(f"No AOI configured for: {scenario}")
        try:
            init_gee()
            today = datetime.now(timezone.utc).date()
            hotspots = get_cloud_reduced_hotspots(
                lon_range=[80.25, 80.9], lat_range=[12.65, 13.7],
                # A two-week lookback makes the provider resilient to product latency.
                start_date=str(today - timedelta(days=14)), end_date=str(today + timedelta(days=1)),
            )
        except Exception as exc:
            raise ProviderNotConfigured(
                "Earth Engine is not available. Configure EE_SERVICE_ACCOUNT and "
                "EE_KEY_FILE, then retry live scan."
            ) from exc

        captured_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        return [
            Observation(
                id=f"EE-{i + 1:03d}", lat=h["lat"], lon=h["lon"], area_m2=0,
                fdi=h["fdi"], confidence=min(0.95, max(0.35, h["fdi"] * 12)),
                captured_at=captured_at, source="Sentinel-2 / Google Earth Engine",
                verification="unverified satellite candidate",
            )
            for i, h in enumerate(hotspots)
        ]


class CopernicusCurrentProvider:
    def current_at(self, lat: float, lon: float) -> CurrentVector:
        """Read a surface-current NetCDF subset supplied by a scheduled download job.

        Set CURRENT_GRID_PATH to the Copernicus/HYCOM subset. This keeps download
        credentials outside the dashboard and lets the live path be exercised with
        a real, timestamped file as soon as access is granted.
        """
        path_value = os.getenv("CURRENT_GRID_PATH")
        if not path_value or not Path(path_value).is_file():
            raise ProviderNotConfigured(
                "Set CURRENT_GRID_PATH to a downloaded Copernicus/HYCOM surface-current "
                "NetCDF subset. Zero-current fallback is intentionally disabled."
            )
        try:
            import xarray as xr
            dataset = xr.open_dataset(path_value)
            u_name = next(name for name in ("uo", "water_u", "u") if name in dataset)
            v_name = next(name for name in ("vo", "water_v", "v") if name in dataset)
            lat_name = next(name for name in ("latitude", "lat") if name in dataset.coords)
            lon_name = next(name for name in ("longitude", "lon") if name in dataset.coords)
            sample = dataset[[u_name, v_name]].sel({lat_name: lat, lon_name: lon}, method="nearest")
            for dim in list(sample[u_name].dims):
                if dim not in (lat_name, lon_name):
                    sample = sample.isel({dim: 0})
            valid_at = str(sample.coords.get("time", "unknown"))
            return CurrentVector(float(sample[u_name]), float(sample[v_name]), "Configured surface-current grid", valid_at)
        except Exception as exc:
            raise ProviderNotConfigured(f"Could not read CURRENT_GRID_PATH: {exc}") from exc


def haversine_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    d_lat, d_lon = radians(b_lat - a_lat), radians(b_lon - a_lon)
    h = sin(d_lat / 2) ** 2 + cos(radians(a_lat)) * cos(radians(b_lat)) * sin(d_lon / 2) ** 2
    return 6371.0 * 2 * asin(sqrt(h))


def _drift(obs: Observation, current: CurrentVector, hours: int) -> tuple[list[list[float]], float]:
    """Simple replay advection. Production replaces this with a particle ensemble."""
    points = [[obs.lat, obs.lon]]
    lat, lon = obs.lat, obs.lon
    for _ in range(0, hours, 6):
        lat += current.v_ms * 6 * 3600 / 111_000
        lon += current.u_ms * 6 * 3600 / (111_000 * cos(radians(lat)))
        points.append([round(lat, 5), round(lon, 5)])
    # Conservative visual uncertainty, not a scientific confidence interval.
    uncertainty_km = round(2.5 + hours * 0.10 + (1 - obs.confidence) * 7, 1)
    return points, uncertainty_km


def _priority(obs: Observation, drift_end: list[float]) -> tuple[str, str, float]:
    habitat = (13.42, 80.32)
    distance = haversine_km(drift_end[0], drift_end[1], *habitat)
    score = min(10.0, obs.confidence * 4 + obs.area_m2 / 10_000 + max(0, 4 - distance / 8))
    return ("CRITICAL" if score >= 7 else "HIGH" if score >= 4 else "REVIEW", "Pulicat pilot habitat buffer", round(score, 1))


def run_mission(mode: str = "replay", scenario: str = "Chennai - Pulicat pilot", forecast_hours: int = 48) -> dict:
    """Run one mission cycle and return JSON-ready data for an API or dashboard."""
    if mode == "replay":
        observation_provider: ObservationProvider = ReplayObservationProvider()
        current_provider: CurrentProvider = ReplayCurrentProvider()
        mode_label = "SIMULATION REPLAY - synthetic observations and currents"
    elif mode == "live":
        observation_provider = EarthEngineObservationProvider()
        current_provider = CopernicusCurrentProvider()
        mode_label = "LIVE SCAN - external provider data"
    else:
        raise ValueError("mode must be 'replay' or 'live'")

    base = (13.08, 80.30)
    incidents = []
    for obs in observation_provider.observations(scenario):
        current = current_provider.current_at(obs.lat, obs.lon)
        track, uncertainty_km = _drift(obs, current, forecast_hours)
        priority, habitat, risk_score = _priority(obs, track[-1])
        eta_hours = round(haversine_km(*base, *track[len(track) // 2]) / (14 * 1.852), 1)
        incidents.append({
            **asdict(obs), "current": asdict(current), "drift_track": track,
            "uncertainty_km": uncertainty_km, "priority": priority,
            "habitat": habitat, "risk_score": risk_score, "intercept": track[len(track) // 2],
            "eta_hours": eta_hours,
        })

    incidents.sort(key=lambda item: item["risk_score"], reverse=True)
    return {
        "mission_id": f"AETHERSEA-{mode.upper()}-240824",
        "mode": mode,
        "mode_label": mode_label,
        "scenario": scenario,
        "generated_at": "2026-08-24T06:00:00Z" if mode == "replay" else datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "forecast_hours": forecast_hours,
        "base_port": {"name": "Chennai demonstration base", "lat": base[0], "lon": base[1]},
        "incidents": incidents,
        "data_derivation": {
            "replay": "Synthetic observations and current vectors stored in backend/mission_pipeline.py. They are deterministic for a reproducible demo.",
            "live": "Sentinel-2 candidates from Google Earth Engine; current provider requires a configured Copernicus Marine adapter.",
        }[mode],
    }

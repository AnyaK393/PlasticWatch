"""Extend the Pune historical cleanup dossier to a 3-month (monsoon) window.

Keeps the original pilot cleanup logs (5 Sep - 4 Oct 2026) untouched and back-fills
5 Jul - 4 Sep 2026 with seeded prototype logs. Accumulation is driven by REAL daily
Pune rainfall (data/pune_rainfall_2026.json, Open-Meteo archive), so heavy-rain weeks
produce heavier, more frequent cleanups exactly as they would on the ground.

Every log (original and generated) is enriched with the evidence fields the
recurrence root-cause engine reasons over:
  - rainfall_prev_48h_mm   antecedent rainfall (real data)
  - first_reported_at      when the accumulation was first reported (deposit timing)
  - waste_composition_pct  share of each TACO class in the removed load

Deterministic: re-running produces the same file.
    python data/generate_historical_dossier.py
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import random

DATA = Path(__file__).resolve().parent
DOSSIER_PATH = DATA / "historical_urban_dossier.json"
RAIN_PATH = DATA / "pune_rainfall_2026.json"

WINDOW_START = date(2026, 7, 5)
PILOT_START = date(2026, 9, 5)  # original pilot logs begin here and are preserved
WINDOW_END = date(2026, 10, 4)
IST = timezone(timedelta(hours=5, minutes=30))

CLASSES = ["plastic_bottle", "plastic_bag_wrapper", "can_metal", "carton_paper", "other_plastic"]

# Generation profiles describe how litter *arrives* at each site. The root-cause engine
# never reads these; it has to recover the driver from the logged evidence alone.
#   hour_bands: relative weight of first-report hour (night 0-5, early 5-9, midday 9-16, evening 16-21, late 21-24)
SITE_PROFILES = {
    "LOC-PUNE-01": {  # Kasba Peth core, 770 m from Mandai market
        "prefix": "SW", "composition": [0.22, 0.46, 0.06, 0.12, 0.14],
        "hour_bands": [0.06, 0.52, 0.14, 0.22, 0.06], "weekend_factor": 1.20,
        "base_kg_day": 7.4, "rain_kg_per_mm": 0.22, "threshold_kg": (30, 42),
    },
    "LOC-PUNE-02": {  # Nagzari corridor, 690 m from Swargate bus terminal
        "prefix": "NZ", "composition": [0.41, 0.19, 0.24, 0.09, 0.07],
        "hour_bands": [0.04, 0.36, 0.14, 0.40, 0.06], "weekend_factor": 0.68,
        "base_kg_day": 7.0, "rain_kg_per_mm": 0.20, "threshold_kg": (33, 50),
    },
    "LOC-PUNE-03": {  # Mula-Mutha confluence outfall
        "prefix": "MO", "composition": [0.18, 0.30, 0.05, 0.10, 0.37],
        "hour_bands": [0.20, 0.20, 0.20, 0.20, 0.20], "weekend_factor": 1.00,
        "base_kg_day": 3.2, "rain_kg_per_mm": 0.95, "threshold_kg": (55, 80),
    },
    "LOC-PUNE-04": {  # Paud Road takeaway kiosk strip, Kothrud
        "prefix": "KF", "composition": [0.36, 0.16, 0.27, 0.16, 0.05],
        "hour_bands": [0.03, 0.08, 0.22, 0.55, 0.12], "weekend_factor": 1.38,
        "base_kg_day": 3.7, "rain_kg_per_mm": 0.10, "threshold_kg": (21, 29),
    },
}
BAND_HOURS = [(0, 5), (5, 9), (9, 16), (16, 21), (21, 24)]


def _parse_day(ts: str) -> date:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(IST).date()


def _rain_prev_48h(daily: dict, day: date) -> float:
    return round(sum(float(daily.get((day - timedelta(days=k)).isoformat()) or 0.0) for k in (1, 2)), 1)


def _composition(rng: random.Random, profile: dict) -> dict:
    raw = [max(0.005, p * rng.uniform(0.8, 1.2)) for p in profile["composition"]]
    total = sum(raw)
    pct = [round(100.0 * r / total, 1) for r in raw]
    pct[-1] = round(100.0 - sum(pct[:-1]), 1)
    return dict(zip(CLASSES, pct))


def _first_reported(rng: random.Random, profile: dict, cleanup_day: date) -> str:
    band = rng.choices(range(5), weights=profile["hour_bands"])[0]
    lo, hi = BAND_HOURS[band]
    # Litter is noticed on the days it is dumped: weight candidate days by the weekly pattern
    days = [cleanup_day - timedelta(days=k) for k in (1, 2, 3)]
    wf = profile["weekend_factor"] ** 3
    report_day = rng.choices(days, weights=[wf if d.weekday() >= 5 else 1.0 for d in days])[0]
    hour = rng.randrange(lo, hi)
    minute = rng.choice([0, 10, 20, 30, 40, 50])
    local = datetime(report_day.year, report_day.month, report_day.day, hour, minute, tzinfo=IST)
    if local.date() == cleanup_day and hour >= 6:
        local -= timedelta(days=1)  # must precede the morning cleanup shift
    return local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _enrich(rng: random.Random, profile: dict, daily: dict, log: dict, source: str) -> dict:
    day = _parse_day(log["timestamp"])
    log.setdefault("rainfall_prev_48h_mm", _rain_prev_48h(daily, day))
    log.setdefault("first_reported_at", _first_reported(rng, profile, day))
    log.setdefault("waste_composition_pct", _composition(rng, profile))
    log.setdefault("record_source", source)
    return log


def build() -> dict:
    dossier = json.loads(DOSSIER_PATH.read_text(encoding="utf-8"))
    daily = json.loads(RAIN_PATH.read_text(encoding="utf-8"))["daily"]

    for loc in dossier["locations"]:
        profile = SITE_PROFILES[loc["site_id"]]
        rng = random.Random(f"pw-dossier-{loc['site_id']}")
        pilot_logs = [c for c in loc["cleanups"] if _parse_day(c["timestamp"]) >= PILOT_START]
        crews = sorted({c["crew_id"] for c in pilot_logs}) or ["PMC-CREW-1"]
        trucks = sorted({c["truck_id"] for c in pilot_logs}) or ["PMC-TRUCK-01"]
        template = pilot_logs[0] if pilot_logs else {}

        generated = []
        acc = rng.uniform(5.0, 15.0)
        threshold = rng.uniform(*profile["threshold_kg"])
        day = WINDOW_START
        while day < PILOT_START:
            rain = float(daily.get(day.isoformat()) or 0.0)
            weekday_factor = profile["weekend_factor"] if day.weekday() >= 5 else 1.0
            acc += profile["base_kg_day"] * weekday_factor * rng.uniform(0.75, 1.25)
            acc += profile["rain_kg_per_mm"] * rain * rng.uniform(0.8, 1.2)
            # Very heavy rain (>= 40 mm/day) triggers a pre-emptive inlet clearance
            preemptive = rain >= 40.0 and acc >= 0.5 * threshold
            if acc >= threshold or preemptive:
                cleanup_day = day + timedelta(days=1)
                weight = round(acc * rng.uniform(0.86, 0.96) * 2) / 2
                eff = round(min(99.0, max(76.0, rng.gauss(91.5, 3.8))), 1)
                hour, minute = rng.choice([6, 7, 8, 9]), rng.choice([0, 15, 30, 45])
                ts = datetime(cleanup_day.year, cleanup_day.month, cleanup_day.day, hour, minute, tzinfo=IST)
                generated.append({
                    "log_id": "",
                    "timestamp": ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "crew_id": rng.choice(crews),
                    "truck_id": rng.choice(trucks),
                    "weight_removed_kg": weight,
                    "clearance_effectiveness_pct": eff,
                    "verification_status": "PASS" if eff >= 85.0 else "FLAGGED",
                    "before_image": template.get("before_image", loc.get("sample_before_image")),
                    "after_image": template.get("after_image", loc.get("sample_after_image")),
                    "operator_signed_off": eff >= 85.0,
                    "trigger": "monsoon_preemptive" if preemptive and acc < threshold else "accumulation_threshold",
                    "notes": (
                        f"Pre-emptive clearance after {rain:.0f} mm rain day." if preemptive and acc < threshold
                        else f"Routine clearance of {weight:.1f} kg accumulation."
                    ),
                })
                acc = max(0.0, acc - weight)
                threshold = rng.uniform(*profile["threshold_kg"])
            day += timedelta(days=1)

        for idx, log in enumerate(generated, 1):
            log["log_id"] = f"LOG-{profile['prefix']}-H{idx:02d}"
            _enrich(rng, profile, daily, log, "seeded_prototype")
        for log in pilot_logs:
            log.setdefault("trigger", "accumulation_threshold")
            _enrich(rng, profile, daily, log, "pilot_log")

        loc["cleanups"] = sorted(generated + pilot_logs, key=lambda c: c["timestamp"])
        loc["total_recurrence_count"] = len(loc["cleanups"])

    dossier["version"] = "3.0.0"
    dossier["time_window_days"] = (WINDOW_END - WINDOW_START).days + 1
    dossier["start_date"] = f"{WINDOW_START.isoformat()}T00:00:00Z"
    dossier["end_date"] = f"{WINDOW_END.isoformat()}T23:59:59Z"
    dossier["rainfall_source"] = "data/pune_rainfall_2026.json (Open-Meteo archive, real)"
    dossier["data_notes"] = (
        "Logs from 2026-09-05 onward are the project's original demo dossier (committed 2026-10-06). "
        "2026-07-05 to 2026-09-04 logs are seeded prototype records generated from REAL daily Pune rainfall "
        "by data/generate_historical_dossier.py. Neither set is an official PMC export."
    )
    return dossier


if __name__ == "__main__":
    out = build()
    DOSSIER_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for loc in out["locations"]:
        months = {}
        for c in loc["cleanups"]:
            months.setdefault(c["timestamp"][:7], []).append(c["weight_removed_kg"])
        print(loc["site_id"], {m: (len(w), round(sum(w))) for m, w in sorted(months.items())})

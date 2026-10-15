"""Incentive Engine: pay reporters only on verified clearance.

A reward ledger driven by events the pipeline already produces:

    SUBMITTED ──quality gate / operator rejects──► VOID (₹0)
        │ report ACTIVE (passes gate or approved in review queue)
        ▼
    PENDING ──hotspot dispatched in a work order──► ESCROWED
                                                      │ verification final decision == PASS
                                                      ▼   (AI or operator override)
                                                   RELEASED
    Any open state ──anti-gaming rule / FLAGGED verification / operator──► HELD

Hook points (each idempotent, each safe to call more than once):
    on_report        end of PlasticWatchUrbanPipeline.submit_report
    on_review        after operator approve / reject in the review queue
    on_dispatch      inside PlasticWatchUrbanPipeline.dispatch_work_order
    on_verification  after a verification case's final decision is known

Storage (JSON, via backend.storage locking + atomic writes):
    data/reporters.json           reporter identities and channels
    data/incentive_programs.json  reward schemes per channel (config)
    data/reward_ledger.json       append-only; one full snapshot row per state change
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import logging
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from backend.recurrence_engine import haversine_m
from backend.storage import CorruptDataError, locked, read_json, write_json

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
REPORTERS_PATH = ROOT / "data" / "reporters.json"
PROGRAMS_PATH = ROOT / "data" / "incentive_programs.json"
LEDGER_PATH = ROOT / "data" / "reward_ledger.json"
VENDOR_REGISTRY_PATH = ROOT / "data" / "vendor_registry.json"

CHANNELS = ("citizen", "waste_picker", "sanitary_inspector", "gig_rider", "dashcam", "rwa")
DEFAULT_CHANNEL = "citizen"

SUBMITTED, PENDING, ESCROWED, RELEASED, HELD, VOID = "SUBMITTED", "PENDING", "ESCROWED", "RELEASED", "HELD", "VOID"
STATES = (SUBMITTED, PENDING, ESCROWED, RELEASED, HELD, VOID)
OPEN_STATES = (SUBMITTED, PENDING, ESCROWED, HELD)
TERMINAL_STATES = (RELEASED, VOID)

FIRST_REPORTER, CORROBORATOR = "first_reporter", "corroborator"

# Anti-gaming parameters
CLUSTER_RADIUS_M = 50.0             # same as DBSCAN eps: a nearby open reward makes you a corroborator
REPEAT_SITE_RADIUS_M = 35.0         # same as the 90-day fraud shield
REPEAT_SITE_WINDOW_DAYS = 90
VERIFICATION_MATCH_RADIUS_M = 75.0  # same as the verification GPS check
EXIF_MAX_DISTANCE_M = 250.0
EXIF_MAX_AGE_HOURS = 48.0           # loose: phone EXIF time is local, not UTC
DEFAULT_DAILY_CAP = 10
DEFAULT_TRUST = 0.7
MIN_OPERATOR_NOTE_CHARS = 10

TRUST_DELTA_REJECTED = -0.05
TRUST_DELTA_DUPLICATE_PHOTO = -0.10
TRUST_DELTA_RELEASED = 0.01

HOLD_KINDS = ("operator", "verification", "repeat_site", "evidence", "self_dealing")


class IncentiveError(ValueError):
    """An operator action or registration that the ledger rules do not allow."""


class UnknownReporterError(LookupError):
    """A report referenced a reporter_id that is not registered."""


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _iso(ts: Optional[datetime] = None) -> str:
    return (ts or _now()).isoformat()


def _parse_ts(value: Any) -> Optional[datetime]:
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _read_list(path: Path) -> List[Dict[str, Any]]:
    try:
        data = read_json(path, [])
    except CorruptDataError as exc:  # quarantined copy is kept for audit
        logger.error("%s", exc)
        return []
    return [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []


def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


# ── Reporters ────────────────────────────────────────────────────────────────
def hash_phone(phone: str) -> Optional[str]:
    digits = re.sub(r"\D", "", str(phone or ""))[-10:]
    if len(digits) < 10:
        return None
    return hashlib.sha256(f"plasticwatch:{digits}".encode()).hexdigest()[:16]


def public_reporter(reporter: Dict[str, Any]) -> Dict[str, Any]:
    """Reporter profile without the phone hash."""
    return {k: v for k, v in reporter.items() if k != "phone_hash"}


def list_reporters() -> List[Dict[str, Any]]:
    return _read_list(REPORTERS_PATH)


def get_reporter(reporter_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not reporter_id:
        return None
    return next((r for r in list_reporters() if r.get("reporter_id") == reporter_id), None)


def _next_reporter_id(reporters: List[Dict[str, Any]]) -> str:
    nums = []
    for r in reporters:
        try:
            nums.append(int(str(r.get("reporter_id", "")).split("-")[-1]))
        except ValueError:
            pass
    return f"RP-{max(nums, default=0) + 1:04d}"


def register_reporter(
    display_name: str,
    channel: str = DEFAULT_CHANNEL,
    phone: Optional[str] = None,
    partner_org: Optional[str] = None,
    ward: Optional[str] = None,
    household_link: Optional[str] = None,
    payout_method: Optional[str] = None,
    program_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], bool]:
    """Register a reporter. Returns (reporter, created).

    The same phone number always maps to the same reporter, so re-registering
    from a new device recovers the wallet instead of creating a duplicate.
    """
    display_name = (display_name or "").strip()[:80]
    channel = (channel or DEFAULT_CHANNEL).strip().lower()
    if len(display_name) < 2:
        raise IncentiveError("display_name is required")
    if channel not in CHANNELS:
        raise IncentiveError(f"channel must be one of {CHANNELS}")
    phone_hash = hash_phone(phone) if phone else None
    if phone and phone_hash is None:
        raise IncentiveError("phone must have at least 10 digits")
    if program_id and program_id not in {p["program_id"] for p in list_programs()}:
        raise IncentiveError(f"Unknown program_id {program_id}")

    with locked(REPORTERS_PATH):
        reporters = list_reporters()
        if phone_hash:
            existing = next((r for r in reporters if r.get("phone_hash") == phone_hash), None)
            if existing:
                return existing, False
        program = program_id or (_default_program(channel) or {}).get("program_id")
        reporter = {
            "reporter_id": _next_reporter_id(reporters),
            "display_name": display_name,
            "channel": channel,
            "partner_org": (partner_org or "").strip()[:80] or None,
            "ward": (ward or "").strip()[:80] or None,
            "phone_hash": phone_hash,
            "program_id": program,
            "payout_method": payout_method or (get_program(program) or {}).get("payout_method"),
            "household_link": (household_link or "").strip()[:40] or None,
            "trust_score": DEFAULT_TRUST,
            "joined_at": _iso(),
        }
        reporters.append(reporter)
        write_json(REPORTERS_PATH, reporters)
    return reporter, True


def resolve_identity(reporter_id: Optional[str], channel: Optional[str]) -> Tuple[Optional[Dict[str, Any]], str]:
    """Validate the identity sent with a report.

    A registered reporter's own channel wins over whatever the client sends.
    Raises UnknownReporterError for an unregistered reporter_id.
    """
    reporter_id = (reporter_id or "").strip() or None
    if reporter_id:
        reporter = get_reporter(reporter_id)
        if reporter is None:
            raise UnknownReporterError(f"Unknown reporter_id {reporter_id}; register via POST /api/reporters")
        return reporter, reporter.get("channel") or DEFAULT_CHANNEL
    channel = (channel or DEFAULT_CHANNEL).strip().lower()
    return None, channel if channel in CHANNELS else DEFAULT_CHANNEL


def _adjust_trust(reporter_id: str, delta: float) -> None:
    try:
        with locked(REPORTERS_PATH):
            reporters = list_reporters()
            for r in reporters:
                if r.get("reporter_id") == reporter_id:
                    current = float(r.get("trust_score", DEFAULT_TRUST))
                    r["trust_score"] = round(min(1.0, max(0.0, current + delta)), 3)
                    write_json(REPORTERS_PATH, reporters)
                    return
    except Exception as exc:  # trust is advisory; never fail a ledger change over it
        logger.warning("Could not adjust trust for %s: %s", reporter_id, exc)


def trust_factor(trust_score: Any) -> float:
    """Reward multiplier: full rate for trusted reporters, reduced as trust drops."""
    try:
        t = float(trust_score)
    except (TypeError, ValueError):
        t = DEFAULT_TRUST
    if t >= 0.7:
        return 1.0
    if t >= 0.4:
        return 0.5
    return 0.0


# ── Programs ─────────────────────────────────────────────────────────────────
def list_programs() -> List[Dict[str, Any]]:
    return _read_list(PROGRAMS_PATH)


def get_program(program_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not program_id:
        return None
    return next((p for p in list_programs() if p.get("program_id") == program_id), None)


def _default_program(channel: str) -> Optional[Dict[str, Any]]:
    candidates = [p for p in list_programs() if p.get("channel") == channel]
    return next((p for p in candidates if p.get("default_for_channel")), candidates[0] if candidates else None)


def program_for(reporter: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return get_program(reporter.get("program_id")) or _default_program(reporter.get("channel") or DEFAULT_CHANNEL)


def reward_amount(program: Optional[Dict[str, Any]], role: str, trust_score: Any) -> int:
    if not program:
        return 0
    key = "first_reporter_inr" if role == FIRST_REPORTER else "corroboration_inr"
    try:
        base = float(program.get(key, 0) or 0)
    except (TypeError, ValueError):
        base = 0.0
    return int(round(base * trust_factor(trust_score)))


# ── Ledger storage ───────────────────────────────────────────────────────────
def ledger_events(reward_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Raw append-only rows, oldest first."""
    rows = _read_list(LEDGER_PATH)
    return [r for r in rows if reward_id is None or r.get("reward_id") == reward_id]


def _fold(events: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    current: Dict[str, Dict[str, Any]] = {}
    for e in events:
        if e.get("reward_id"):
            current[e["reward_id"]] = e
    return current


def list_rewards(reporter_id: Optional[str] = None, state: Optional[str] = None) -> List[Dict[str, Any]]:
    """Current state of every reward (the latest snapshot row), newest first."""
    rewards = [
        r for r in _fold(ledger_events()).values()
        if (reporter_id is None or r.get("reporter_id") == reporter_id) and (state is None or r.get("state") == state)
    ]
    return sorted(rewards, key=lambda r: str(r.get("report_submitted_at") or r.get("created_at") or ""), reverse=True)


def get_reward(reward_id: str) -> Optional[Dict[str, Any]]:
    return _fold(ledger_events(reward_id)).get(reward_id)


def reward_id_for(report_id: str) -> str:
    return f"RW-{report_id}"


_EVENT_FIELDS = ("event_id", "event", "previous_state", "reason", "actor", "recorded_at")


def _append(events: List[Dict[str, Any]], snapshot: Dict[str, Any], event: str, reason: str, actor: str) -> Dict[str, Any]:
    """Append a full snapshot row recording one state change (caller holds the lock)."""
    previous = _fold(events).get(snapshot["reward_id"])
    row = {k: v for k, v in snapshot.items() if k not in _EVENT_FIELDS}
    row.update({
        "event_id": f"EV-{len(events) + 1:06d}",
        "event": event,
        "previous_state": previous.get("state") if previous else None,
        "reason": reason,
        "actor": actor,
        "recorded_at": _iso(),
    })
    if row["state"] == RELEASED and not row.get("released_at"):
        row["released_at"] = row["recorded_at"]
    events.append(row)
    return row


def _hold(snapshot: Dict[str, Any], kind: str, reason: str) -> Dict[str, Any]:
    from_state = snapshot["state"] if snapshot["state"] != HELD else (snapshot.get("hold") or {}).get("from_state", PENDING)
    return {**snapshot, "state": HELD, "hold": {"kind": kind, "reason": reason, "from_state": from_state}}


# ── Hook: report submitted ───────────────────────────────────────────────────
def _initial_state(report: Dict[str, Any]) -> str:
    from backend.taco_adapter import REVIEW_STATUS

    if report.get("verification") == "rejected" or str(report.get("status", "")).upper() == "REJECTED":
        return VOID
    if report.get("verification") == REVIEW_STATUS:
        return SUBMITTED
    return PENDING


def _photo_fingerprint(image_bytes: Optional[bytes]) -> Optional[str]:
    return hashlib.sha1(image_bytes).hexdigest() if image_bytes else None


def _exif_problem(image_bytes: Optional[bytes], lat: float, lon: float, submitted_at: Optional[datetime]) -> Optional[str]:
    """Capture GPS/time that contradicts the submission (missing EXIF is not a problem)."""
    if not image_bytes:
        return None
    try:
        from backend.verification_engine import _exif_facts, _load_pil

        facts = _exif_facts(_load_pil(image_bytes))
    except Exception as exc:
        logger.debug("EXIF read failed: %s", exc)
        return None
    if facts.get("lat") is not None and facts.get("lon") is not None:
        d = haversine_m(lat, lon, facts["lat"], facts["lon"])
        if d > EXIF_MAX_DISTANCE_M:
            return f"Photo GPS is {d:.0f} m from the reported location"
    captured = facts.get("captured_at")
    if captured is not None and submitted_at is not None:
        gap_h = abs((submitted_at.replace(tzinfo=None) - captured).total_seconds()) / 3600.0
        if gap_h > EXIF_MAX_AGE_HOURS:
            return f"Photo was captured {gap_h:.0f} h away from submission time"
    return None


def _new_reward(report: Dict[str, Any], reporter: Dict[str, Any], events: List[Dict[str, Any]],
                image_bytes: Optional[bytes], apply_rules: bool) -> Tuple[Dict[str, Any], str, str]:
    """Build the first snapshot for a report. Returns (snapshot, event, reason)."""
    lat, lon = float(report["lat"]), float(report["lon"])
    program = program_for(reporter)
    current = list(_fold(events).values())
    submitted = _parse_ts(report.get("submitted_at")) or _now()

    nearby_open = any(
        r.get("state") in OPEN_STATES and haversine_m(lat, lon, float(r["lat"]), float(r["lon"])) <= CLUSTER_RADIUS_M
        for r in current if r.get("lat") is not None
    )
    role = CORROBORATOR if nearby_open else FIRST_REPORTER
    fingerprint = _photo_fingerprint(image_bytes)
    snapshot = {
        "reward_id": reward_id_for(report["id"]),
        "report_id": report["id"],
        "reporter_id": reporter["reporter_id"],
        "channel": reporter.get("channel") or DEFAULT_CHANNEL,
        "program_id": (program or {}).get("program_id"),
        "payout_method": reporter.get("payout_method") or (program or {}).get("payout_method"),
        "role": role,
        "amount_inr": reward_amount(program, role, reporter.get("trust_score")),
        "state": _initial_state(report),
        "lat": lat,
        "lon": lon,
        "ward": report.get("ward"),
        "photo_fingerprint": fingerprint,
        "report_submitted_at": report.get("submitted_at"),
        "hotspot": None,
        "work_order_id": None,
        "hold": None,
        "evidence": None,
        "provenance": "derived",
        "created_at": _iso(),
        "released_at": None,
    }
    if snapshot["state"] == VOID:
        return snapshot, "voided", "Report rejected"
    if not apply_rules:
        return snapshot, "backfilled", "Ledger backfill for an existing report"

    if fingerprint and any(r.get("photo_fingerprint") == fingerprint for r in current):
        return {**snapshot, "state": VOID}, "voided", "Same photo was already submitted for another report"

    cap = int((program or {}).get("daily_cap", DEFAULT_DAILY_CAP) or DEFAULT_DAILY_CAP)
    today = submitted.date()
    todays = sum(
        1 for r in current
        if r.get("reporter_id") == reporter["reporter_id"] and r.get("state") != VOID
        and (_parse_ts(r.get("report_submitted_at")) or submitted).date() == today
    )
    if todays >= cap:
        return {**snapshot, "state": VOID}, "voided", f"Daily cap of {cap} rewarded reports reached"

    window_start = submitted - timedelta(days=REPEAT_SITE_WINDOW_DAYS)
    for r in current:
        released_at = _parse_ts(r.get("released_at"))
        if (r.get("state") == RELEASED and released_at and released_at >= window_start
                and haversine_m(lat, lon, float(r["lat"]), float(r["lon"])) <= REPEAT_SITE_RADIUS_M):
            reason = (f"Site within {REPEAT_SITE_RADIUS_M:.0f} m was already paid out on {released_at.date()} "
                      f"({r['reward_id']}); repeat payouts need operator review")
            return _hold(snapshot, "repeat_site", reason), "held", reason

    problem = _exif_problem(image_bytes, lat, lon, submitted)
    if problem:
        return _hold(snapshot, "evidence", problem), "held", problem

    if snapshot["state"] == SUBMITTED:
        return snapshot, "submitted", "Awaiting operator review of the photo"
    return snapshot, "pending", "Report accepted; reward pending verified cleanup"


def on_report(report: Dict[str, Any], image_bytes: Optional[bytes] = None) -> Optional[Dict[str, Any]]:
    """Create the reward for a new report. Anonymous reports earn nothing and are skipped."""
    reporter = get_reporter(report.get("reporter_id"))
    if reporter is None or not report.get("id"):
        return None
    with locked(LEDGER_PATH):
        events = ledger_events()
        existing = _fold(events).get(reward_id_for(report["id"]))
        if existing:
            return existing
        snapshot, event, reason = _new_reward(report, reporter, events, image_bytes, apply_rules=True)
        row = _append(events, snapshot, event, reason, "system")
        write_json(LEDGER_PATH, events)
    if event == "voided" and "photo" in reason:
        _adjust_trust(reporter["reporter_id"], TRUST_DELTA_DUPLICATE_PHOTO)
    return row


def backfill_from_reports(reports: Iterable[Dict[str, Any]]) -> int:
    """Create rewards for reporter-tagged reports that have none (seed data, pre-ledger reports).

    Anti-gaming rules are not re-applied to history. Returns the number created.
    """
    known = {r["reporter_id"]: r for r in list_reporters() if r.get("reporter_id")}
    todo = [r for r in reports if r.get("reporter_id") in known and r.get("id")]
    if not todo:
        return 0
    created = 0
    with locked(LEDGER_PATH):
        events = ledger_events()
        current = _fold(events)
        for report in sorted(todo, key=lambda r: str(r.get("submitted_at", ""))):
            if reward_id_for(report["id"]) in current:
                continue
            try:
                snapshot, event, reason = _new_reward(report, known[report["reporter_id"]], events, None, apply_rules=False)
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping backfill for %s: %s", report.get("id"), exc)
                continue
            current[snapshot["reward_id"]] = _append(events, snapshot, event, reason, "system")
            created += 1
        if created:
            write_json(LEDGER_PATH, events)
    return created


# ── Hook: operator review queue ──────────────────────────────────────────────
def on_review(report: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Operator approved (→ PENDING) or rejected (→ VOID) a queued capture."""
    if not report or not report.get("id"):
        return None
    if get_reward(reward_id_for(report["id"])) is None:
        return on_report(report)
    target = _initial_state(report)
    if target == SUBMITTED:
        return get_reward(reward_id_for(report["id"]))
    with locked(LEDGER_PATH):
        events = ledger_events()
        current = _fold(events).get(reward_id_for(report["id"]))
        if current is None or current.get("state") != SUBMITTED:
            return current
        if target == VOID:
            row = _append(events, {**current, "state": VOID}, "voided", "Operator rejected the report", "operator")
        else:
            row = _append(events, {**current, "state": PENDING}, "approved", "Operator approved the report", "operator")
        write_json(LEDGER_PATH, events)
    if target == VOID:
        _adjust_trust(current["reporter_id"], TRUST_DELTA_REJECTED)
    return row


# ── Hook: work order dispatched ──────────────────────────────────────────────
def on_dispatch(hotspots: Iterable[Dict[str, Any]], work_order_id: Optional[str]) -> List[Dict[str, Any]]:
    """Escrow PENDING rewards for every dispatched hotspot and settle first-reporter roles.

    Within a hotspot the earliest PENDING report is the first reporter, unless the
    site already has an escrowed reward from an earlier, still-open work order.
    """
    changed: List[Dict[str, Any]] = []
    hotspots = [h for h in hotspots if h.get("report_ids")]
    if not hotspots:
        return changed
    # Tagged reports that predate the ledger must be escrowed (and win first-reporter) too
    backfill_from_reports(r for h in hotspots for r in h.get("reports", []) if isinstance(r, dict))
    reporters = {r["reporter_id"]: r for r in list_reporters() if r.get("reporter_id")}
    with locked(LEDGER_PATH):
        events = ledger_events()
        current = _fold(events)
        for h in hotspots:
            rewards = [current[reward_id_for(rid)] for rid in h["report_ids"] if reward_id_for(rid) in current]
            pending = sorted((r for r in rewards if r.get("state") == PENDING), key=lambda r: str(r.get("report_submitted_at") or ""))
            open_cycle = any(r.get("state") == ESCROWED for r in rewards)
            hotspot_info = {
                "label": h.get("id") or h.get("hotspot_id"),
                "site_key": h.get("site_key"),
                "report_count": len(h["report_ids"]),
                "is_merged": bool(h.get("is_merged")),
            }
            for i, r in enumerate(pending):
                role = FIRST_REPORTER if (i == 0 and not open_cycle) else CORROBORATOR
                reporter = reporters.get(r["reporter_id"], {})
                snapshot = {
                    **r,
                    "state": ESCROWED,
                    "role": role,
                    "amount_inr": reward_amount(get_program(r.get("program_id")), role, reporter.get("trust_score")),
                    "hotspot": hotspot_info,
                    "work_order_id": work_order_id,
                }
                reason = f"Hotspot {hotspot_info['label']} dispatched in {work_order_id or 'work order'}"
                row = _append(events, snapshot, "escrowed", reason, "system")
                current[row["reward_id"]] = row
                changed.append(row)
        if changed:
            write_json(LEDGER_PATH, events)
    return changed


# ── Hook: cleanup verification decided ───────────────────────────────────────
def _vendor_name(vendor_id: Optional[str]) -> Optional[str]:
    if not vendor_id:
        return None
    try:
        vendors = read_json(VENDOR_REGISTRY_PATH, [])
    except CorruptDataError:
        return None
    return next((v.get("name") for v in vendors if isinstance(v, dict) and v.get("vendor_id") == vendor_id), None)


def _self_dealing(reporter: Optional[Dict[str, Any]], case: Dict[str, Any]) -> Optional[str]:
    """Reporter belongs to the crew or vendor that cleaned the site."""
    if not reporter:
        return None
    mine = {_norm(reporter.get("partner_org")), _norm(reporter.get("display_name"))} - {""}
    theirs = {_norm(case.get("contractor")), _norm(case.get("vendor_id")), _norm(_vendor_name(case.get("vendor_id")))} - {""}
    for a in mine:
        for b in theirs:
            if a == b or (min(len(a), len(b)) >= 6 and (a in b or b in a)):
                return f"Reporter is affiliated with the cleaning contractor ({case.get('contractor') or case.get('vendor_id')})"
    return None


def _case_matches(case: Dict[str, Any], reward: Dict[str, Any]) -> bool:
    if case.get("report_ids"):
        return reward.get("report_id") in set(case["report_ids"])
    try:
        return haversine_m(float(case["lat"]), float(case["lon"]), float(reward["lat"]), float(reward["lon"])) <= VERIFICATION_MATCH_RADIUS_M
    except (KeyError, TypeError, ValueError):
        return False


def on_verification(case: Dict[str, Any], final: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Settle escrowed rewards near a verification case once its final decision is known.

    PASS releases (unless the reporter is affiliated with the contractor); FLAGGED FOR
    REVIEW holds; INCONCLUSIVE changes nothing (the contractor must re-shoot).
    """
    decision = str((final or {}).get("decision") or "").upper()
    if decision not in ("PASS", "FLAGGED FOR REVIEW") or not case or not case.get("case_id"):
        return []
    evidence = {
        "verification_case": case["case_id"],
        "decision": decision,
        "decision_source": final.get("source", "AI"),
        "fingerprint": case.get("evidence_fingerprint"),
        "override_id": (final.get("override") or {}).get("override_id"),
        "site_name": case.get("site_name"),
        "contractor": case.get("contractor"),
    }
    reporters = {r["reporter_id"]: r for r in list_reporters() if r.get("reporter_id")}
    changed: List[Dict[str, Any]] = []
    released_reporters: List[str] = []
    with locked(LEDGER_PATH):
        events = ledger_events()
        for r in list(_fold(events).values()):
            in_escrow = r.get("state") == ESCROWED
            held_after_dispatch = r.get("state") == HELD and (r.get("hold") or {}).get("from_state") == ESCROWED
            if not (in_escrow or held_after_dispatch) or not _case_matches(case, r):
                continue
            prior = r.get("evidence") or {}
            if (prior.get("verification_case"), prior.get("decision"), prior.get("fingerprint")) == \
                    (evidence["verification_case"], decision, evidence["fingerprint"]):
                continue  # already applied
            snapshot = {**r, "evidence": evidence}
            hold_kind = (r.get("hold") or {}).get("kind")
            label = f"{case['case_id']} {decision} ({evidence['decision_source']})"
            if decision == "PASS":
                conflict = _self_dealing(reporters.get(r["reporter_id"]), case)
                if conflict:
                    row = _append(events, _hold(snapshot, "self_dealing", conflict), "held", conflict, "system")
                elif in_escrow or hold_kind == "verification":
                    row = _append(events, {**snapshot, "state": RELEASED, "hold": None}, "released",
                                  f"Verified clear: {label}", "system")
                    released_reporters.append(r["reporter_id"])
                else:
                    row = _append(events, snapshot, "evidence_attached",
                                  f"Verified clear ({label}) but still held: {(r.get('hold') or {}).get('reason')}", "system")
            elif in_escrow:
                row = _append(events, _hold(snapshot, "verification", f"Cleanup verification {label}"), "held",
                              f"Cleanup verification {label}", "system")
            else:
                row = _append(events, snapshot, "evidence_attached", f"Verification {label}", "system")
            changed.append(row)
        if changed:
            write_json(LEDGER_PATH, events)
    for reporter_id in released_reporters:
        _adjust_trust(reporter_id, TRUST_DELTA_RELEASED)
    return changed


# ── Operator actions ─────────────────────────────────────────────────────────
def _validate_operator(operator: str, note: str) -> Tuple[str, str]:
    operator, note = (operator or "").strip(), (note or "").strip()
    if len(operator) < 2:
        raise IncentiveError("Operator name is required")
    if len(note) < MIN_OPERATOR_NOTE_CHARS:
        raise IncentiveError(f"Note must be at least {MIN_OPERATOR_NOTE_CHARS} characters")
    return operator[:80], note[:500]


def hold_reward(reward_id: str, operator: str, note: str) -> Dict[str, Any]:
    """Operator freezes an open reward pending investigation."""
    operator, note = _validate_operator(operator, note)
    with locked(LEDGER_PATH):
        events = ledger_events()
        current = _fold(events).get(reward_id)
        if current is None:
            raise KeyError(reward_id)
        if current["state"] not in (SUBMITTED, PENDING, ESCROWED):
            raise IncentiveError(f"Only SUBMITTED, PENDING or ESCROWED rewards can be held (current: {current['state']})")
        row = _append(events, _hold(current, "operator", note), "held", note, operator)
        write_json(LEDGER_PATH, events)
    return row


def release_reward(reward_id: str, operator: str, note: str) -> Dict[str, Any]:
    """Operator resolves a hold.

    Payment still requires verified clearance: a held reward with PASS evidence is
    RELEASED; otherwise it returns to the stage it was held at and releases when
    verification passes (use the verification override for a disputed AI verdict).
    """
    operator, note = _validate_operator(operator, note)
    with locked(LEDGER_PATH):
        events = ledger_events()
        current = _fold(events).get(reward_id)
        if current is None:
            raise KeyError(reward_id)
        if current["state"] != HELD:
            raise IncentiveError(f"Only HELD rewards can be released by an operator (current: {current['state']})")
        if (current.get("evidence") or {}).get("decision") == "PASS":
            row = _append(events, {**current, "state": RELEASED, "hold": None}, "released", note, operator)
        else:
            back_to = (current.get("hold") or {}).get("from_state") or PENDING
            row = _append(events, {**current, "state": back_to, "hold": None}, "hold_cleared",
                          f"{note} (reward releases when cleanup verification passes)", operator)
        write_json(LEDGER_PATH, events)
    if row["state"] == RELEASED:
        _adjust_trust(row["reporter_id"], TRUST_DELTA_RELEASED)
    return row


# ── Views ────────────────────────────────────────────────────────────────────
def status_message(reward: Dict[str, Any]) -> str:
    amt = f"₹{reward.get('amount_inr', 0)}"
    state = reward.get("state")
    if state == SUBMITTED:
        return f"{amt} once an operator confirms your photo and the drain is verified clear."
    if state == PENDING:
        share = " (corroborating an earlier report here)" if reward.get("role") == CORROBORATOR else ""
        return f"{amt} pending{share}, released when the drain is verified clear."
    if state == ESCROWED:
        return f"{amt} in escrow: a crew has been dispatched ({reward.get('work_order_id') or 'work order'})."
    if state == RELEASED:
        return f"{amt} released: cleanup verified clear ✅"
    if state == HELD:
        return f"{amt} on hold for review: {(reward.get('hold') or {}).get('reason', 'under review')}"
    return "No reward for this report."


def timeline(reward: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Reported → merged into hotspot → crew dispatched → verified clear."""
    history = ledger_events(reward["reward_id"])
    first_at = lambda *names: next((e["recorded_at"] for e in history if e.get("event") in names), None)  # noqa: E731
    hotspot = reward.get("hotspot") or {}
    evidence = reward.get("evidence") or {}
    escrowed_at = first_at("escrowed")
    return [
        {"stage": "reported", "label": f"Reported {reward['report_id']}", "done": True, "at": reward.get("report_submitted_at")},
        {"stage": "clustered", "label": (f"Merged into {hotspot.get('label')} ({hotspot.get('report_count')} reports)"
                                         if hotspot.get("is_merged") else f"Hotspot {hotspot.get('label')}") if hotspot else "Awaiting hotspot dispatch",
         "done": bool(hotspot), "at": escrowed_at},
        {"stage": "dispatched", "label": f"Crew dispatched ({reward.get('work_order_id')})" if escrowed_at else "Crew not yet dispatched",
         "done": bool(escrowed_at), "at": escrowed_at},
        {"stage": "verified", "label": (f"Verified clear ✅ ₹{reward.get('amount_inr', 0)} released" if reward.get("state") == RELEASED
                                        else f"Verification {evidence.get('decision')}" if evidence else "Awaiting cleanup verification"),
         "done": reward.get("state") == RELEASED, "at": reward.get("released_at"),
         "verification_case": evidence.get("verification_case")},
    ]


def wallet(reporter_id: str) -> Optional[Dict[str, Any]]:
    reporter = get_reporter(reporter_id)
    if reporter is None:
        return None
    rewards = list_rewards(reporter_id=reporter_id)

    def total(*states: str) -> int:
        return sum(int(r.get("amount_inr") or 0) for r in rewards if r.get("state") in states)

    return {
        "reporter": public_reporter(reporter),
        "program": get_program(reporter.get("program_id")) or program_for(reporter),
        "totals": {
            "pending_inr": total(SUBMITTED, PENDING),
            "escrowed_inr": total(ESCROWED),
            "held_inr": total(HELD),
            "released_inr": total(RELEASED),
            "reports": len(rewards),
            "verified_clearances": sum(1 for r in rewards if r.get("state") == RELEASED),
            "void_reports": sum(1 for r in rewards if r.get("state") == VOID),
        },
        "rewards": [{**r, "status_message": status_message(r), "timeline": timeline(r)} for r in rewards],
    }


def adoption_stats(reports: Iterable[Dict[str, Any]], days: Optional[int] = 7, now: Optional[datetime] = None) -> Dict[str, Any]:
    """Who is reporting, through which channel, and how much of the drain network they cover.

    ``reports`` are pipeline report dicts (anonymous ones count as citizen reports);
    ``days`` is the activity window (None = all time).
    """
    from backend.ward_engine import drain_coverage, reports_in_window

    window = reports_in_window(reports, days, now)
    rewards = list_rewards()
    channels: Dict[str, Dict[str, Any]] = {}
    for ch in CHANNELS:
        mine = [r for r in window if (r.get("channel") or DEFAULT_CHANNEL) == ch]
        released = [w for w in rewards if w.get("channel") == ch and w.get("state") == RELEASED]
        cov = drain_coverage(mine)
        channels[ch] = {
            "reports": len(mine),
            "active_reporters": len({r["reporter_id"] for r in mine if r.get("reporter_id")}),
            "anonymous_reports": sum(1 for r in mine if not r.get("reporter_id")),
            "coverage_pct": cov["coverage_pct"],
            "covered_km": cov["covered_km"],
            "verified_clearances": len(released),
            "released_inr": sum(int(w.get("amount_inr") or 0) for w in released),
        }
    overall = drain_coverage(window)
    return {
        "window_days": days,
        "reports": len(window),
        "active_reporters": len({r["reporter_id"] for r in window if r.get("reporter_id")}),
        "coverage": overall,
        "by_channel": channels,
    }


def summary() -> Dict[str, Any]:
    """Ledger KPIs for the dashboard: by channel, by state, and ₹ per verified clearance."""
    rewards = list_rewards()
    reporters = list_reporters()
    by_state = {s: {"count": 0, "amount_inr": 0} for s in STATES}
    by_channel: Dict[str, Dict[str, Any]] = {
        c: {"reporters": 0, "reports": 0, "released_inr": 0, "verified_clearances": 0} for c in CHANNELS
    }
    for rp in reporters:
        by_channel.setdefault(rp.get("channel") or DEFAULT_CHANNEL, {"reporters": 0, "reports": 0, "released_inr": 0, "verified_clearances": 0})
        by_channel[rp.get("channel") or DEFAULT_CHANNEL]["reporters"] += 1
    for r in rewards:
        s = by_state.setdefault(r.get("state"), {"count": 0, "amount_inr": 0})
        s["count"] += 1
        s["amount_inr"] += int(r.get("amount_inr") or 0)
        ch = by_channel.setdefault(r.get("channel") or DEFAULT_CHANNEL, {"reporters": 0, "reports": 0, "released_inr": 0, "verified_clearances": 0})
        ch["reports"] += 1
        if r.get("state") == RELEASED:
            ch["released_inr"] += int(r.get("amount_inr") or 0)
            ch["verified_clearances"] += 1
    cleared_sites = {(r.get("evidence") or {}).get("verification_case") for r in rewards if r.get("state") == RELEASED} - {None}
    released_inr = by_state[RELEASED]["amount_inr"]
    return {
        "reporters": len(reporters),
        "rewards": len(rewards),
        "by_state": by_state,
        "by_channel": by_channel,
        "released_inr": released_inr,
        "verified_clearances": len(cleared_sites),
        "inr_per_verified_clearance": round(released_inr / len(cleared_sites), 2) if cleared_sites else None,
        "payout_queue": [r for r in rewards if r.get("state") == HELD],
    }

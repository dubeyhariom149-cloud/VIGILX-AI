import csv
import json
import os
from datetime import datetime
from pathlib import Path

# Authoritative path: events/events.json
BASE_DIR = Path(__file__).resolve().parent.parent
EVENT_DIR = BASE_DIR / "events"
EVENT_FILE = EVENT_DIR / "events.json"
EVENT_CSV = EVENT_DIR / "events.csv"

# Mirror paths for backwards compatibility with root-level scripts
ROOT_EVENT_FILE = BASE_DIR / "events.json"
ROOT_EVENT_CSV = BASE_DIR / "events.csv"

EVENT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_event(e):
    """Normalize legacy and current event dictionary keys."""
    if not isinstance(e, dict):
        return e

    evt_type = e.get("event") or e.get("event_type", "INCIDENT")
    desc = e.get("description") or e.get("message", "")
    persons = e.get("persons") if "persons" in e else e.get("person_count", 0)
    vehicles = e.get("vehicles", 0)
    phones = e.get("phones", 0)
    source = e.get("source") or e.get("camera_source", "CAM 01")
    severity = e.get("severity", "LOW")
    evidence = e.get("evidence", "")
    ts = e.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Maintain dual keys for full backward and forward compatibility
    return {
        "timestamp": ts,
        "event": evt_type,
        "event_type": evt_type,
        "description": desc,
        "message": desc,
        "severity": severity,
        "persons": persons,
        "person_count": persons,
        "vehicles": vehicles,
        "phones": phones,
        "source": source,
        "evidence": evidence,
    }


def load_events():
    """Load all events from the authoritative store (fallback to root if needed)."""
    target = EVENT_FILE if EVENT_FILE.exists() else ROOT_EVENT_FILE
    if not target.exists():
        return []

    try:
        with open(target, "r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, list):
            return [normalize_event(e) for e in data if isinstance(e, dict)]
        return []
    except (json.JSONDecodeError, OSError):
        return []


def save_events(events):
    """Save normalized events to authoritative and mirror JSON and CSV files."""
    normalized = [normalize_event(e) for e in events if isinstance(e, dict)]

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for e in normalized:
        key = (e.get("timestamp"), e.get("event"))
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    # Sort chronologically by timestamp
    deduped.sort(key=lambda x: x.get("timestamp", ""))

    # 1. Authoritative and mirror JSON files
    for path in [EVENT_FILE, ROOT_EVENT_FILE]:
        try:
            with open(path, "w", encoding="utf-8") as file:
                json.dump(deduped, file, indent=2, ensure_ascii=False)
        except OSError:
            pass

    # 2. Authoritative and mirror CSV files
    if deduped:
        fields = sorted(set().union(*(e.keys() for e in deduped)))
        for path in [EVENT_CSV, ROOT_EVENT_CSV]:
            try:
                with open(path, "w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=fields)
                    writer.writeheader()
                    for e in deduped:
                        writer.writerow(e)
            except OSError:
                pass

    return deduped


def save_event(
    event_type,
    message,
    person_count=0,
    severity="LOW",
    vehicles=0,
    phones=0,
    source="CAM 01",
    evidence="",
):
    """Append a new event chronologically and persist across JSON/CSV stores."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    event = {
        "timestamp": now_str,
        "event": event_type,
        "event_type": event_type,
        "description": message,
        "message": message,
        "severity": severity,
        "persons": person_count,
        "person_count": person_count,
        "vehicles": vehicles,
        "phones": phones,
        "source": source,
        "evidence": evidence,
    }

    events = load_events()
    events.append(event)
    save_events(events)
    return event


def get_events(limit=20):
    """Return the most recent events, sorted newest first."""
    events = load_events()
    return list(reversed(events[-limit:]))


if __name__ == "__main__":
    print("VIGILX-AI Event Logger Self-Test")
    events_before = len(load_events())
    print(f"Total events in store: {events_before}")

    test_evt = save_event(
        event_type="DIAGNOSTIC_TEST",
        message="Self-test execution",
        person_count=1,
        severity="LOW"
    )
    events_after = len(load_events())
    print(f"Event saved: {test_evt['timestamp']} | count now: {events_after}")
    assert events_after == events_before + 1

    # Clean up test event so production log remains pristine
    cleaned = [e for e in load_events() if e.get("event") != "DIAGNOSTIC_TEST"]
    save_events(cleaned)
    print(f"Pristine state restored: {len(load_events())} events.")
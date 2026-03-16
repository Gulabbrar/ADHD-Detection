"""
serial_reader.py
================
Background EEG data worker.
Reads JSON from serial port and saves directly to SQLite via database.py.
Used as a daemon thread by modules/eeg.py.
"""
import serial, json, time, threading
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────────────────────
SERIAL_PORT     = "COM6"
BAUD_RATE       = 9600
RECONNECT_DELAY = 5
# ─────────────────────────────────────────────────────────────────────────────

# Module-level thread state
_thread: threading.Thread | None = None
_stop_event = threading.Event()
_status = {"connected": False, "samples": 0, "last_error": ""}


def _parse(raw: str) -> dict | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if data.get("status") != "live" or "attention" not in data:
        return None
    return data


def _worker(patient_id: int, session_id: str, stop: threading.Event):
    """Main serial reading loop — runs in a daemon thread."""
    from database import save_eeg_signal
    _status["samples"] = 0
    _status["last_error"] = ""

    while not stop.is_set():
        try:
            with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2) as ser:
                _status["connected"] = True
                _status["last_error"] = ""
                while not stop.is_set():
                    try:
                        raw = ser.readline().decode("utf-8", errors="replace")
                    except Exception as e:
                        _status["last_error"] = str(e)
                        break
                    data = _parse(raw)
                    if data is None:
                        continue
                    try:
                        save_eeg_signal(patient_id, session_id, data)
                        _status["samples"] += 1
                    except Exception as e:
                        _status["last_error"] = f"DB write: {e}"
        except serial.SerialException as e:
            _status["connected"] = False
            _status["last_error"] = str(e)
            stop.wait(RECONNECT_DELAY)
        except Exception as e:
            _status["connected"] = False
            _status["last_error"] = str(e)
            stop.wait(RECONNECT_DELAY)

    _status["connected"] = False


def start(patient_id: int, session_id: str):
    global _thread, _stop_event
    if _thread and _thread.is_alive():
        return  # already running
    _stop_event.clear()
    _thread = threading.Thread(
        target=_worker,
        args=(patient_id, session_id, _stop_event),
        daemon=True,
        name="eeg_serial"
    )
    _thread.start()


def stop():
    global _stop_event
    _stop_event.set()


def is_running() -> bool:
    return _thread is not None and _thread.is_alive() and not _stop_event.is_set()


def get_status() -> dict:
    return dict(_status)

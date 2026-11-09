"""Isolate memory-heavy calibration from the Qt process on Windows."""

from __future__ import annotations

import copy
import multiprocessing
import queue
import time
import traceback
from pathlib import Path

try:
    from . import engine
    from .project import Project
except ImportError:
    import engine
    from project import Project


def _perform_calibration(snapshot, directory, cancel, progress):
    record, report = engine.run_project(snapshot, Path(directory), cancel, progress)
    shadow = Project(Path(directory), copy.deepcopy(snapshot))
    shadow.data["active_run"] = record
    updates = {}
    layer_total = sum(len(scan["layers"]) for scan in shadow.data["scans"] if scan["path"])
    finished_layers = 0
    for scan in shadow.data["scans"]:
        engine._check_cancel(cancel)
        if not scan["path"]:
            continue
        if not scan["layers"] or report["scans"].get(scan["id"], {}).get("status") != "ready":
            finished_layers += len(scan["layers"])
            continue
        try:
            start = finished_layers
            updates[scan["id"]] = engine.refine_layers(
                shadow, scan, cancel,
                lambda value, total, message, layers=len(scan["layers"]): progress(
                    950 + round(50 * (start + layers * value / max(1, total)) / max(1, layer_total)),
                    1000, message))
        except InterruptedError:
            raise
        except Exception as exc:
            report["scans"][scan["id"]]["local_error"] = str(exc)
        finished_layers += len(scan["layers"])
    if layer_total:
        progress(1000, 1000, "Color matching complete")
    return record, report, updates


def _calibrate_child(snapshot, directory, cancel, messages):
    try:
        result = _perform_calibration(snapshot, directory, cancel,
                                      lambda *values: messages.put(("progress", values)))
        messages.put(("completed", result))
    except InterruptedError:
        messages.put(("cancelled", None))
    except Exception:
        # A worker stopped by cancellation may fail in its own way: still a cancel.
        messages.put(("cancelled", None) if cancel.is_set() else ("failed", traceback.format_exc()))


def run_calibration(snapshot: dict, directory: Path, cancel, progress):
    """Return a completed run, or surface a child crash without killing the UI."""
    messages = None
    try:
        context = multiprocessing.get_context("spawn")
        child_cancel = context.Event()
        messages = context.Queue()
        process = context.Process(target=_calibrate_child,
                                  args=(snapshot, str(directory), child_cancel, messages),
                                  name="scan-color-calibration")
        process.start()
    except PermissionError:
        if messages is not None:
            messages.close()
        progress(0, 1, "Process isolation unavailable; continuing with memory-limited workers")
        return _perform_calibration(snapshot, directory, cancel, progress)
    cancelled_at = None
    try:
        while True:
            if cancel.is_set():
                child_cancel.set()
                cancelled_at = cancelled_at or time.monotonic()
                if time.monotonic() - cancelled_at > 5:
                    raise InterruptedError("Color matching was cancelled")
            try:
                kind, payload = messages.get(timeout=.25)
            except queue.Empty:
                if process.is_alive():
                    continue
                process.join()
                if cancel.is_set():
                    raise InterruptedError("Color matching was cancelled")
                raise RuntimeError(f"Calibration process exited unexpectedly (code {process.exitcode}); "
                                   "the previous result is unchanged")
            except (EOFError, OSError) as exc:
                raise RuntimeError("Calibration process disconnected; the previous result is unchanged") from exc
            if kind == "progress":
                progress(*payload)
            elif kind == "completed":
                process.join(timeout=5)
                if process.exitcode not in (None, 0):
                    raise RuntimeError(f"Calibration process exited with code {process.exitcode}")
                return payload
            elif kind == "cancelled":
                raise InterruptedError("Color matching was cancelled")
            elif kind == "failed":
                if cancel.is_set():
                    raise InterruptedError("Color matching was cancelled")
                raise RuntimeError(payload)
    finally:
        if process.is_alive():
            child_cancel.set()
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join()
        messages.close()

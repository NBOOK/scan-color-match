"""Small, explicit and automatically saved project model."""

from __future__ import annotations

import json
import os
import uuid
import hashlib
import io
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=128)
def _asset_digest(path: str, modified: int, changed: int, size: int) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_id() -> str:
    return uuid.uuid4().hex[:12]


class Project:
    def __init__(self, directory: Path, data: dict):
        self.directory = directory.resolve()
        self.data = data

    @classmethod
    def create(cls, directory: Path) -> "Project":
        directory = directory.resolve()
        directory.mkdir(parents=True, exist_ok=True)
        project_file = directory / "project.json"
        if project_file.exists():
            raise FileExistsError(f"A project already exists in {directory}")
        project = cls(directory, {
            "version": 1,
            "calibration_revision": 0,
            "active_run": None,
            "fit_metric": "de76",
            "passes": 2,
            "scans": [],
        })
        project.save()
        return project

    @classmethod
    def open(cls, directory: Path) -> "Project":
        directory = directory.resolve()
        with (directory / "project.json").open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if data.get("version") != 1 or not isinstance(data.get("scans"), list):
            raise ValueError("Unsupported project format")
        project = cls(directory, data)
        run = data.get("active_run")
        if run and "input_signature" not in run and run.get("revision") == data.get("calibration_revision"):
            run["input_signature"] = project.calibration_signature()
        return project

    def save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / "project.json"
        temporary = path.with_name("project.json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self.data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)

    def invalidate(self) -> None:
        self.data["calibration_revision"] += 1
        self.save()

    def add_scan(self, path: str = "", *, save: bool = True) -> dict:
        scan = {
            "id": new_id(), "path": str(Path(path).resolve()) if path else "",
            "mode": "shared", "include_shared": False, "refs": [], "layers": [],
            "global_visible": True, "global_name": "global",
            "local_output": None, "local_dirty": False,
        }
        self.data["scans"].append(scan)
        if save:
            self.invalidate()
        return scan

    def add_reference(self, scan: dict, path: str, *, save: bool = True) -> dict:
        reference = {
            "id": new_id(), "path": str(Path(path).resolve()),
            "enabled": True, "mask_mode": "exclude",
        }
        if not scan["refs"] and scan.get("mode") in ("shared", "follow"):
            # Default once a scan has references: its own LUT, also helping the shared one.
            scan["mode"], scan["include_shared"] = "own", True
            scan.pop("follow", None)
        scan["refs"].append(reference)
        if save:
            self.invalidate()
        return reference

    def normalize_lut_modes(self) -> bool:
        """Keep LUT modes meaningful; return True when something had to change.

        Own LUT needs references, and only scans without references can follow
        (reuse the Own LUT of) another scan. A follower falls back to Shared when the
        scan it follows is gone or no longer has its own LUT.
        """
        changed = False
        by_id = {scan["id"]: scan for scan in self.data["scans"]}
        for scan in self.data["scans"]:
            if scan.get("mode") == "own" and not scan["refs"]:
                scan["mode"] = "shared"
                changed = True
        for scan in self.data["scans"]:
            if scan.get("mode") != "follow":
                if "follow" in scan:
                    scan.pop("follow")
                    changed = True
                continue
            target = by_id.get(scan.get("follow"))
            if (scan["refs"] or target is None or target is scan
                    or target.get("mode") != "own" or not target["refs"]):
                scan["mode"] = "shared"
                scan.pop("follow", None)
                changed = True
        return changed

    def scan(self, scan_id: str) -> dict:
        return next(item for item in self.data["scans"] if item["id"] == scan_id)

    def reference(self, scan: dict, ref_id: str) -> dict:
        return next(item for item in scan["refs"] if item["id"] == ref_id)

    def exclusion_signature(self, ref: dict):
        if not ref.get("exclude_mask"):
            return None
        path = self.directory / ref["exclude_mask"]
        try:
            stat = path.stat()
            actual = _asset_digest(str(path), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
        except OSError:
            actual = "missing"
        return (ref["exclude_mask"], ref.get("exclude_mask_digest"), actual,
                ref.get("mask_mode", "exclude"))

    def set_reference_mask_mode(self, ref: dict, mode: str) -> bool:
        """Switch whether painted reference pixels are excluded or the only ones used."""
        if mode not in ("exclude", "include"):
            raise ValueError(f"Unknown reference mask mode: {mode}")
        if ref.get("mask_mode", "exclude") == mode:
            return False
        ref["mask_mode"] = mode
        # The mode only changes evidence when something is painted.
        if ref.get("exclude_mask"):
            self.invalidate()
        else:
            self.save()
        return True

    def set_reference_exclusion(self, ref: dict, mask) -> bool:
        """Commit the painted binary, reference-coordinate sampling mask.

        White pixels are painted; ``mask_mode`` decides whether they are excluded
        or are the only reference pixels used as evidence.

        Assets are immutable: a failed save or an in-flight reader keeps its old mask.
        Empty masks use the original project representation and signature.
        """
        import numpy as np
        from PIL import Image
        with Image.open(ref["path"]) as image:
            expected = (image.height, image.width)
        mask = np.asarray(mask)
        if mask.shape != expected:
            raise ValueError("Reference exclusion mask size differs from the reference")
        previous = {key: ref[key] for key in ("exclude_mask", "exclude_mask_digest") if key in ref}
        replacement = {}
        if np.any(mask):
            buffer = io.BytesIO()
            Image.fromarray(np.where(mask > 0, 255, 0).astype(np.uint8), "L").save(buffer, format="PNG")
            contents = buffer.getvalue()
            digest = hashlib.sha256(contents).hexdigest()
            relative = f"assets/reference-masks/{ref['id']}-{digest}.png"
            path = self.directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(contents)
                os.replace(temporary, path)
            replacement = {"exclude_mask": relative, "exclude_mask_digest": digest}
        if replacement == previous:
            return False
        for key in ("exclude_mask", "exclude_mask_digest"):
            ref.pop(key, None)
        ref.update(replacement)
        revision = self.data["calibration_revision"]
        try:
            self.invalidate()
        except Exception:
            for key in ("exclude_mask", "exclude_mask_digest"):
                ref.pop(key, None)
            ref.update(previous)
            self.data["calibration_revision"] = revision
            raise
        return True

    def run_dir(self) -> Path | None:
        run = self.data.get("active_run")
        return self.directory / "cache" / "runs" / run["id"] if run else None

    def is_current(self) -> bool:
        run = self.data.get("active_run")
        if not run:
            return False
        if "input_signature" in run:
            return run["input_signature"] == self.calibration_signature()
        return run["revision"] == self.data["calibration_revision"]

    def calibration_signature(self) -> str:
        """Only inputs that affect global matching, not UI or local layer state."""
        inputs = [{
            "id": scan["id"], "path": scan["path"], "mode": scan["mode"],
            **({"follow": scan.get("follow")} if scan["mode"] == "follow" else {}),
            "include_shared": scan.get("include_shared", False),
            "refs": [{"id": ref["id"], "path": ref["path"],
                      "enabled": ref.get("enabled", True),
                      **({"exclude_mask": self.exclusion_signature(ref)}
                         if ref.get("exclude_mask") else {})}
                     for ref in scan["refs"]],
        } for scan in self.data["scans"]]
        # How the global LUT is fitted changes every result.
        inputs = {"scans": inputs, "fit_metric": self.data.get("fit_metric", "de76"),
                  "passes": self.data.get("passes", 2)}
        payload = json.dumps(inputs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

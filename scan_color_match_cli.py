"""Command-line access to the same calibration engine used by the desktop app."""

from __future__ import annotations

import os

# The bundled OpenBLAS (built for at most 2 threads) crashes when several Python
# threads call into it at once. Work is parallel per scan instead; set before numpy
# loads, and inherited by every spawned worker process.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import copy
import json
import shutil
import sys
import threading
import numpy as np
from PIL import Image
from multiprocessing import freeze_support
from pathlib import Path

import engine
from calibration_worker import run_calibration
from project import Project


def _progress(value: int, total: int, message: str) -> None:
    percent = round(100 * value / max(1, total))
    print(f"[{percent:3d}%] {message}", file=sys.stderr, flush=True)


def _activate(project: Project) -> dict:
    snapshot = copy.deepcopy(project.data)
    record, report, updates = run_calibration(
        snapshot, project.directory, threading.Event(), _progress)
    if project.data["calibration_revision"] != record["revision"]:
        raise RuntimeError("Project inputs changed while calibration was running")
    record["input_signature"] = project.calibration_signature()
    project.data["active_run"] = record
    for scan in project.data["scans"]:
        scan["evaluation"] = None
        if scan["id"] in updates:
            scan["layers"], scan["local_output"] = updates[scan["id"]]
            scan["local_dirty"] = False
        else:
            scan["local_output"] = None
            scan["local_dirty"] = bool(scan["layers"])
    project.save()
    report_path = project.run_dir() / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _export_ready(project: Project, destination: Path, image_format: str,
                  previews: bool) -> list[str]:
    outputs = []
    for scan in project.data["scans"]:
        try:
            output = engine.export_scan(project, scan, destination, image_format,
                                        include_report=True, include_preview=previews)
        except ValueError as exc:
            print(f"Skipped {Path(scan['path']).name}: {exc}", file=sys.stderr)
            continue
        outputs.append(str(output))
    return outputs


def match_pair(args) -> None:
    work = args.work_dir or args.output_dir / f".{args.scan.stem}-scan-color-project"
    if (work / "project.json").exists():
        raise FileExistsError(f"Working project already exists: {work}")
    project = Project.create(work)
    scan = project.add_scan(str(args.scan), save=False)
    scan["mode"] = "own"
    reference = project.add_reference(scan, str(args.reference), save=False)
    reference["mask_mode"] = args.mask_mode
    if args.reference_mask:
        with Image.open(args.reference_mask) as source:
            if source.format != "PNG":
                raise ValueError("Reference sampling mask must be a PNG")
            mask = np.asarray(source.convert("L"), dtype=np.uint8)
        if np.any((mask != 0) & (mask != 255)):
            raise ValueError("Reference sampling mask must contain only black and white")
        project.set_reference_exclusion(reference, mask)
    project.invalidate()
    report = _activate(project)
    outputs = _export_ready(project, args.output_dir, args.format, args.preview)
    lut = project.run_dir() / "lut" / f"{scan['id']}.cube"
    copied_lut = None
    if lut.exists():
        target = args.output_dir / f"{args.scan.stem}-color-match.cube"
        suffix = 2
        while target.exists():
            target = args.output_dir / f"{args.scan.stem}-color-match-{suffix}.cube"
            suffix += 1
        shutil.copy2(lut, target)
        copied_lut = str(target)
    print(json.dumps({"project": str(project.directory), "outputs": outputs,
                      "lut": copied_lut, "report": report}, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Scan Color Match command-line interface")
    commands = result.add_subparsers(dest="command", required=True)
    pair = commands.add_parser("match", help="match one scan to one reference")
    pair.add_argument("--scan", type=Path, required=True)
    pair.add_argument("--reference", type=Path, required=True)
    pair.add_argument("--output-dir", type=Path, required=True)
    pair.add_argument("--work-dir", type=Path)
    pair.add_argument("--format", choices=("png", "webp"), default="png")
    pair.add_argument("--preview", action="store_true")
    pair.add_argument("--reference-mask", type=Path, metavar="PATH",
                      help="reference-sized binary PNG: white = painted, black = not painted")
    pair.add_argument("--mask-mode", choices=("exclude", "include"), default="exclude",
                      help="exclude: painted pixels are not used (default); "
                           "include: only painted pixels are used")
    pair.set_defaults(handler=match_pair)
    return result


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    freeze_support()
    main()

"""Color calibration and non-destructive local corrections for the desktop UI."""

from __future__ import annotations

import os

# The bundled OpenBLAS (built for at most 2 threads) crashes when several Python
# threads call into it at once. Work is parallel per scan instead; set before numpy
# loads, and inherited by every spawned worker process.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import copy
import ctypes
import faulthandler
import hashlib
import importlib.util
import json
import math
import multiprocessing
import queue
import shutil
import struct
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw

cv2.setNumThreads(1)

MATCHER_PATH = Path(__file__).resolve().with_name("match_scan_to_reference.py")
SUPPORTED = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
Progress = Callable[[int, int, str], None]
FALLBACK_MEMORY_BUDGET = 1_500_000_000
MAX_MEMORY_BUDGET = 12_000_000_000


def _available_physical_memory() -> int | None:
    """Read current free RAM without introducing another required dependency."""
    if os.name == "nt":
        status = ctypes.create_string_buffer(64)
        ctypes.c_uint32.from_buffer(status).value = 64
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(status):
            return struct.unpack_from("<Q", status, 16)[0]
    elif hasattr(os, "sysconf"):
        try:
            return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError):
            pass
    return None


def _working_memory_budget() -> int:
    available = _available_physical_memory()
    if available is None:
        return FALLBACK_MEMORY_BUDGET
    return max(512_000_000, min(MAX_MEMORY_BUDGET, int(available * .4)))


def _scan_memory_weight(scan: dict) -> int:
    """Conservative native/Python working-set estimate for one full-size scan task."""
    try:
        width, height = validate_image(scan["path"])
    except Exception:
        return 128_000_000
    reference_pixels = 0
    for ref in scan.get("refs", []):
        if ref.get("enabled", True):
            try:
                rw, rh = validate_image(ref["path"])
                reference_pixels = max(reference_pixels, rw * rh)
            except Exception:
                pass
    estimate = width * height * 28 + reference_pixels * 8
    return max(128_000_000, estimate)


class _MemoryGate:
    def __init__(self, budget: int, cancel: threading.Event):
        self.budget = budget
        self.cancel = cancel
        self.used = 0
        self.condition = threading.Condition()

    def run(self, weight: int, operation, *args):
        weight = min(weight, self.budget)
        with self.condition:
            while self.used + weight > self.budget:
                _check_cancel(self.cancel)
                self.condition.wait(.1)
            _check_cancel(self.cancel)
            self.used += weight
        try:
            return operation(*args)
        finally:
            with self.condition:
                self.used -= weight
                self.condition.notify_all()


_CORRECTION_CANCEL = None
_CORRECTION_MATCHER = None
_CORRECTION_PROGRESS_QUEUE = None


def _init_correction_worker(cancel, progress_queue):
    global _CORRECTION_CANCEL, _CORRECTION_MATCHER, _CORRECTION_PROGRESS_QUEUE
    faulthandler.enable(all_threads=False)
    _CORRECTION_CANCEL = cancel
    _CORRECTION_MATCHER = _matcher()
    _CORRECTION_PROGRESS_QUEUE = progress_queue


def _correction_worker_count(weights: dict[str, int], budget: int, requested: int) -> int:
    heaviest = sorted(weights.values(), reverse=True)
    count = 0
    used = 0
    for weight in heaviest[:requested]:
        if count and used + weight > budget:
            break
        used += weight
        count += 1
    return max(1, count)


def _matcher():
    spec = importlib.util.spec_from_file_location("scan_match_core", MATCHER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Color matching core is missing: {MATCHER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_image(path: str) -> tuple[int, int]:
    source = Path(path)
    if source.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Unsupported format: {source.suffix}")
    if source.suffix.lower() == ".png":
        with source.open("rb") as handle:
            header = handle.read(25)
        if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 25 and header[24] == 16:
            raise ValueError("16-bit PNG is not supported in this version")
    with Image.open(source) as image:
        if image.mode in {"I", "F", "I;16", "I;16B", "I;16L"}:
            raise ValueError("16-bit images are not supported in this version")
        if image.format == "TIFF":
            bits = image.tag_v2.get(258, ())
            if isinstance(bits, int):
                bits = (bits,)
            if any(bit > 8 for bit in bits):
                raise ValueError("16-bit TIFF is not supported in this version")
        return image.size


def load_image(path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    validate_image(path)
    with Image.open(path) as image:
        info = {key: image.info[key] for key in ("icc_profile", "dpi") if key in image.info}
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    return rgba[:, :, :3].copy(), rgba[:, :, 3].copy(), info


MASK_MODES = ("exclude", "include")


def reference_mask_mode(ref: dict) -> str:
    """How a reference's painted sampling mask is read: painted = excluded or the only area used."""
    mode = ref.get("mask_mode", "exclude")
    if mode not in MASK_MODES:
        raise ValueError(f"Unknown reference mask mode: {mode}")
    return mode


def load_reference_exclusion(directory: Path, ref: dict) -> np.ndarray | None:
    """Effective exclusion in reference coordinates (white = not used as evidence).

    Include mode inverts the painted mask, so every consumer keeps one exclusion gate.
    An empty painted mask never restricts sampling in either mode.
    """
    mask = load_reference_mask(directory, ref)
    if mask is None or reference_mask_mode(ref) == "exclude":
        return mask
    return np.where(mask > 0, 0, 255).astype(np.uint8)


def load_reference_mask(directory: Path, ref: dict) -> np.ndarray | None:
    """Read and validate the painted binary source asset; never silently allow bad evidence."""
    if not ref.get("exclude_mask"):
        return None
    path = (Path(directory) / ref["exclude_mask"]).resolve()
    if not path.is_relative_to(Path(directory).resolve()):
        raise ValueError("Reference exclusion mask must be inside the project")
    try:
        contents = path.read_bytes()
        if (ref.get("exclude_mask_digest") and
                hashlib.sha256(contents).hexdigest() != ref["exclude_mask_digest"]):
            raise ValueError("Reference exclusion mask content changed")
        with Image.open(path) as source:
            if source.format != "PNG" or source.mode not in {"L", "1"}:
                raise ValueError("Reference exclusion mask must be a grayscale PNG")
            mask = np.asarray(source.convert("L"), dtype=np.uint8).copy()
        with Image.open(ref["path"]) as source:
            expected = (source.height, source.width)
        if mask.shape != expected:
            raise ValueError("Reference exclusion mask size differs from the reference")
        if np.any((mask != 0) & (mask != 255)):
            raise ValueError("Reference exclusion mask must contain only black and white")
        return mask
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid reference exclusion mask: {exc}") from exc


def warp_exclusion(mask: np.ndarray, transform: np.ndarray, size: tuple[int, int],
                   margin: int = 2) -> np.ndarray:
    """Conservative binary warp, including subpixel features during minification.

    Inverse resampling fills magnified regions; forward splatting every excluded
    source pixel prevents thin marks from falling between destination samples.
    A source-pixel guard protects RGB interpolation, followed by a working-pixel
    guard for the Sobel/resampling boundary. Display/edit warps use margin=0.
    """
    binary = (mask > 0).astype(np.uint8)
    if margin:
        binary = cv2.dilate(binary, np.ones((3, 3), np.uint8))
    transformed = cv2.warpPerspective(binary.astype(np.float32), transform, size,
                                      flags=cv2.INTER_LINEAR) > 0
    # Splatting only boundary pixels is equivalent: a destination pixel that holds
    # both excluded and allowed source pixels also holds a boundary pixel, and one
    # that is fully excluded is already caught by the interpolated warp. This keeps
    # large masks (such as inverted "use only" masks) fast. Image edges count as
    # boundary so partially covered edge pixels are still splatted.
    boundary = binary & ~cv2.erode(binary, np.ones((3, 3), np.uint8),
                                   borderType=cv2.BORDER_CONSTANT, borderValue=0)
    for top in range(0, boundary.shape[0], 64):
        ys, xs = np.nonzero(boundary[top:top + 64])
        if not len(xs):
            continue
        points = np.column_stack((xs, ys + top)).astype(np.float64)
        points = cv2.perspectiveTransform(points.reshape(-1, 1, 2), transform).reshape(-1, 2)
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        inside = ((points[:, 0] >= -.5) & (points[:, 0] < size[0] - .5)
                  & (points[:, 1] >= -.5) & (points[:, 1] < size[1] - .5))
        points = np.floor(points[inside] + .5).astype(np.int32)
        transformed[points[:, 1], points[:, 0]] = True
    if margin:
        transformed = cv2.dilate(transformed.astype(np.uint8),
                                  np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)) > 0
    return transformed


def _exclude_reference(valid: np.ndarray, mask: np.ndarray | None,
                       transform: np.ndarray, sampled_size=None) -> np.ndarray:
    """One evidence gate for training, spatial correction, refinement and evaluation."""
    if mask is None:
        return valid
    if sampled_size is not None and sampled_size != (mask.shape[1], mask.shape[0]):
        # Evaluation first downsamples large references. Protect that interpolation
        # footprint before mapping it to scan pixels, including magnified crops.
        scaling = np.diag([sampled_size[0] / mask.shape[1],
                           sampled_size[1] / mask.shape[0], 1.0])
        mask = warp_exclusion(mask, scaling, sampled_size, margin=2)
        transform = transform @ np.linalg.inv(scaling)
    return valid & ~warp_exclusion(mask, transform, (valid.shape[1], valid.shape[0]))


def reference_valid(valid: np.ndarray, directory: Path, ref: dict,
                    transform: np.ndarray, sampled_size=None) -> np.ndarray:
    return _exclude_reference(valid, load_reference_exclusion(directory, ref),
                              transform, sampled_size)


def edit_reference_exclusion(mask: np.ndarray, points, kind: str, radius: int,
                             transform: np.ndarray | None = None) -> np.ndarray:
    """Rasterize a complete display-space shape, then map its coverage to the source."""
    if len(points) == 0:
        return mask.copy()
    points = np.asarray(points, np.float64)
    padding = max(1, int(radius)) + 2 if kind in {"paint", "erase"} else 2
    origin = np.floor(points.min(axis=0) - padding).astype(int)
    end = np.ceil(points.max(axis=0) + padding).astype(int)
    size = tuple(int(value) for value in np.maximum(1, end - origin + 1))
    delta = np.zeros((size[1], size[0]), np.uint8)
    vertices = np.round(points - origin).astype(np.int32)
    if kind in {"paint", "erase"}:
        cv2.polylines(delta, [vertices], False, 255, max(1, 2 * radius), cv2.LINE_8)
        for point in (vertices[0], vertices[-1]):
            cv2.circle(delta, tuple(point), max(1, radius), 255, -1)
    elif kind in {"rectangle", "polygon"}:
        cv2.fillPoly(delta, [vertices], 255)
    else:
        raise ValueError(f"Unknown exclusion edit: {kind}")
    translation = np.array([[1, 0, origin[0]], [0, 1, origin[1]], [0, 0, 1]], np.float64)
    to_reference = np.linalg.inv(transform) if transform is not None else np.eye(3)
    coverage = warp_exclusion(delta, to_reference @ translation,
                              (mask.shape[1], mask.shape[0]), margin=0)
    result = mask.copy()
    result[coverage] = 0 if kind == "erase" else 255
    return result


def replace_file(source: Path, destination: Path, attempts: int = 40) -> None:
    """os.replace that waits out brief sharing violations on Windows.

    Another thread may be reading ``destination`` (a preview, a thumbnail) at the very
    moment it is replaced; Windows then refuses the rename for a few milliseconds.
    """
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(.05)


def _save_png(path: Path, rgb: np.ndarray, alpha: np.ndarray | None = None, info: dict | None = None,
              compress_level: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if alpha is None or np.all(alpha == 255):
        image = Image.fromarray(rgb, "RGB")
    else:
        image = Image.fromarray(np.dstack((rgb, alpha)), "RGBA")
    kwargs = {key: value for key, value in (info or {}).items() if key in {"icc_profile", "dpi"}}
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    image.save(temporary, compress_level=compress_level, **kwargs)
    replace_file(temporary, path)


def _check_cancel(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise InterruptedError("Color matching was cancelled")


def _align(ms, scan_rgb: np.ndarray, ref_rgb: np.ndarray):
    homography, matches, inliers, ratio = ms.estimate_homography(scan_rgb, ref_rgb)
    return homography / homography[2, 2], matches, inliers, ratio


def alignment_signature(scan: dict, ref: dict) -> str:
    """Geometry inputs only: color sampling edits must not move the reference view."""
    inputs = []
    for path in (scan["path"], ref["path"]):
        stat = Path(path).stat()
        inputs.append((str(Path(path).resolve()), stat.st_mtime_ns, stat.st_size))
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def _aligned_work(ms, scan_rgb, scan_alpha, ref_rgb, ref_alpha, homography,
                  exclude_mask=None):
    height, width = scan_rgb.shape[:2]
    scale = min(0.32, 1700.0 / max(height, width))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    scan = cv2.resize(scan_rgb, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    scan_valid = cv2.resize(scan_alpha, size, interpolation=cv2.INTER_NEAREST) > 0
    transform = np.diag([scale, scale, 1.0]) @ homography
    reference = cv2.warpPerspective(ref_rgb.astype(np.float32) / 255.0, transform, size)
    footprint = cv2.warpPerspective(
        np.ones(ref_rgb.shape[:2], np.uint8), transform, size, flags=cv2.INTER_NEAREST
    ) > 0
    footprint = cv2.erode(footprint.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    usable = cv2.warpPerspective(ref_alpha, transform, size, flags=cv2.INTER_NEAREST) > 0
    valid = footprint & usable & scan_valid
    valid = _exclude_reference(valid, exclude_mask, transform)
    if np.count_nonzero(valid) < 10000:
        if exclude_mask is not None:
            raise ValueError("Too few aligned pixels inside the reference sampling mask; "
                             "enlarge the included area or reduce the exclusions")
        raise ValueError("Too little aligned reference coverage")
    edge = np.maximum(ms.gradient_strength(scan), ms.gradient_strength(reference))
    limit = float(np.quantile(edge[valid], 0.68))
    candidate = valid & (edge <= limit)
    if np.count_nonzero(candidate) < 10000:
        if exclude_mask is not None:
            raise ValueError("Too few reliable color samples inside the reference sampling mask; "
                             "enlarge the included area or reduce the exclusions")
        raise ValueError("Too few reliable flat-color samples")
    return {
        "scan": scan, "reference": reference, "footprint": footprint,
        "valid": valid, "candidate": candidate, "edge": edge,
        "limit": max(limit, 1e-5), "scale": scale,
    }


def image_delta_e76(rgb: np.ndarray, evaluation: Path) -> float | None:
    """Mean ΔE76 of a full-size scan result on a run's saved evaluation pixels.

    The image is reduced exactly like the aligned working image, so the original scan,
    the global result and any composed result are measured on the same pixels.
    """
    if not Path(evaluation).exists():
        return None
    with np.load(evaluation) as data:
        indexes, targets, size = data["indexes"], data["targets"], tuple(int(v) for v in data["size"])
    if rgb.shape[1] == size[0] and rgb.shape[0] == size[1]:
        small = rgb
    else:
        small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    return _metric(small.reshape(-1, 3)[indexes].astype(np.float32) / 255.0, targets)


def output_delta_e76(project, scan: dict, output: str | None) -> float | None:
    """ΔE76 of what is shown for a scan: a composed local result, or its base image."""
    run = project.run_dir()
    if run is None:
        return None
    evaluation = run / "evaluation" / f"{scan['id']}.npz"
    if not evaluation.exists():
        return None
    if output:
        path = project.directory / output
    elif scan.get("global_visible", True):
        path = run / "base" / f"{scan['id']}.png"
    else:
        path = Path(scan["path"])
    return image_delta_e76(load_image(str(path))[0], evaluation)


def _metric(rgb, target) -> float:
    left = cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.float32), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    right = cv2.cvtColor(target.reshape(-1, 1, 3).astype(np.float32), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    return round(float(np.mean(np.linalg.norm(left - right, axis=1))), 4)


def _balanced_samples(sample_groups):
    if not sample_groups:
        return None
    rng = np.random.default_rng(3703)
    count = min(len(pair[0]) for pair in sample_groups)
    sources, targets, tiles = [], [], []
    for source, target, position, _ in sample_groups:
        selected = rng.choice(len(source), count, replace=False)
        sources.append(source[selected])
        targets.append(target[selected])
        tiles.append(position[selected])
    return np.concatenate(sources), np.concatenate(targets), np.concatenate(tiles)


PERCEPTUAL_SAMPLES = 12000


def _fit_pairs(ms, source, target, perceptual=False):
    """Fit one polynomial LUT, to RGB error or (perceptual) to mean ΔE76.

    The perceptual optimizer is iterative; 12k pairs pin 60 coefficients down just as
    well as 30k and keep it a few times faster.
    """
    if not perceptual:
        return ms.fit_mapping(source, target)
    if len(source) > PERCEPTUAL_SAMPLES:
        chosen = np.random.default_rng(5021).choice(len(source), PERCEPTUAL_SAMPLES, replace=False)
        source, target = source[chosen], target[chosen]
    return ms.fit_perceptual_mapping(source, target)


def _fit_balanced(ms, sample_groups, perceptual=False):
    samples = _balanced_samples(sample_groups)
    return _fit_pairs(ms, samples[0], samples[1], perceptual) if samples is not None else None


def _fit_guarded_balanced(ms, sample_groups, perceptual=False):
    source, target, tiles = _balanced_samples(sample_groups)
    return _guarded_mapping(ms, source, target, _support_radius(source, tiles), perceptual)


def _sample_group(ref_samples, seed: int):
    rng = np.random.default_rng(seed)
    per_ref = max(1000, 30000 // len(ref_samples))
    source_parts, target_parts, tile_parts = [], [], []
    for source, target, tiles in ref_samples:
        indexes = rng.choice(len(source), min(per_ref, len(source)), replace=False)
        source_parts.append(source[indexes])
        target_parts.append(target[indexes])
        tile_parts.append(tiles[indexes])
    source = np.concatenate(source_parts)
    target = np.concatenate(target_parts)
    tiles = np.concatenate(tile_parts)
    indexes = rng.choice(len(source), min(30000, len(source)), replace=False)
    return source[indexes], target[indexes], tiles[indexes]


def _color_index(source: np.ndarray):
    # The same 33-point lattice is used by the exported cube. Quantization
    # removes duplicate colors before building the small nearest-color index.
    occupied = np.unique(np.round(np.clip(source, 0, 1) * 32).astype(np.uint8), axis=0)
    if len(occupied) == 1:
        occupied = np.repeat(occupied, 2, axis=0)
    return cv2.flann_Index(np.ascontiguousarray(occupied.astype(np.float32) / 32),
                           dict(algorithm=1, trees=4))


def _nearest_color_distance(index, colors: np.ndarray) -> np.ndarray:
    _, squared = index.knnSearch(np.ascontiguousarray(colors.astype(np.float32)), 1,
                                 params=dict(checks=128))
    return np.sqrt(np.maximum(squared[:, 0], 0))


def _support_radius(source: np.ndarray, tiles: np.ndarray) -> float:
    """Estimate how far matched colors generalize using spatially held-out pixels."""
    distances = []
    for fold in range(5):
        held = tiles % 5 == fold
        if held.sum() < 100 or (~held).sum() < 100:
            continue
        index = _color_index(source[~held])
        chosen = np.flatnonzero(held)
        if len(chosen) > 3000:
            chosen = np.random.default_rng(901 + fold).choice(chosen, 3000, replace=False)
        distances.append(_nearest_color_distance(index, source[chosen]))
    # A grid cell is the minimum meaningful distance for a 33-point LUT.
    return max(1 / 32, float(np.quantile(np.concatenate(distances), .9))
               if distances else 1 / 32)


def _guarded_mapping(ms, source: np.ndarray, target: np.ndarray, radius: float,
                     perceptual=False) -> np.ndarray:
    fitted = _fit_pairs(ms, source, target, perceptual)
    axis = np.linspace(0, 1, 33, dtype=np.float32)
    blue, green, red = np.meshgrid(axis, axis, axis, indexing="ij")
    colors = np.stack((red, green, blue), axis=-1).reshape(-1, 3)
    corrected = ms.apply_mapping(colors, fitted).reshape(-1, 3)
    distance = _nearest_color_distance(_color_index(source), colors)
    fade = np.clip((distance - radius) / radius, 0, 1)
    confidence = 1 - fade * fade * (3 - 2 * fade)
    grid = colors + (corrected - colors) * confidence[:, None]
    return np.clip(grid.reshape(33, 33, 33, 3), 0, 1).astype(np.float32)


def _apply_mapping(ms, rgb: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    if mapping.ndim == 2:
        return ms.apply_mapping(rgb, mapping)
    # Grid axes are B, G, R to match .cube's R-fastest row order.
    coords = np.clip(rgb.astype(np.float32), 0, 1) * 32
    lo = np.minimum(coords.astype(np.int32), 31)
    frac = coords - lo
    result = np.zeros_like(coords, dtype=np.float32)
    for b in (0, 1):
        for g in (0, 1):
            for r in (0, 1):
                weight = ((frac[..., 0] if r else 1 - frac[..., 0])
                          * (frac[..., 1] if g else 1 - frac[..., 1])
                          * (frac[..., 2] if b else 1 - frac[..., 2]))
                result += mapping[lo[..., 2] + b, lo[..., 1] + g,
                                  lo[..., 0] + r] * weight[..., None]
    return result


def _write_mapping_cube(ms, path: Path, mapping: np.ndarray) -> None:
    if mapping.ndim == 2:
        ms.write_cube(path, mapping)
        return
    lines = [f'TITLE "{path.stem}"', 'LUT_3D_SIZE 33',
             'DOMAIN_MIN 0.0 0.0 0.0', 'DOMAIN_MAX 1.0 1.0 1.0']
    lines.extend(f'{r:.7f} {g:.7f} {b:.7f}' for r, g, b in mapping.reshape(-1, 3))
    path.write_text('\n'.join(lines) + '\n', encoding='ascii')


def _apply_mapping_rows(ms, image: np.ndarray, mapping: np.ndarray, rows: int = 96):
    result = np.empty_like(image, dtype=np.float32)
    for top in range(0, image.shape[0], rows):
        bottom = min(image.shape[0], top + rows)
        result[top:bottom] = _apply_mapping(ms, image[top:bottom], mapping)
    return result


def _validate_mapping(ms, groups: list[tuple[np.ndarray, np.ndarray]], perceptual=False):
    """Validate against held-out scan groups, or held-out pixels for one scan."""
    results = []
    if len(groups) == 1:
        source, target = groups[0][:2]
        rng = np.random.default_rng(739)
        order = rng.permutation(len(source))
        split = max(100, int(.8 * len(order)))
        model = _fit_pairs(ms, source[order[:split]], target[order[:split]], perceptual)
        test_source, test_target = source[order[split:]], target[order[split:]]
        results.append({
            "raw": _metric(test_source, test_target),
            "corrected": _metric(ms.apply_mapping(test_source, model), test_target),
        })
    else:
        folds = min(5, len(groups))
        for fold in range(folds):
            train = [pair for index, pair in enumerate(groups) if index % folds != fold]
            model = _fit_balanced(ms, train, perceptual=perceptual)
            for index, group in enumerate(groups):
                if index % folds != fold:
                    continue
                source, target = group[:2]
                rng = np.random.default_rng(index + 619)
                selected = rng.choice(len(source), min(4000, len(source)), replace=False)
                test_source, test_target = source[selected], target[selected]
                results.append({
                    "raw": _metric(test_source, test_target),
                    "corrected": _metric(ms.apply_mapping(test_source, model), test_target),
                })
    improved = sum(row["corrected"] < row["raw"] for row in results)
    return {
        "pages": results,
        "improved": improved,
        "total": len(results),
        "median_raw": round(float(np.median([row["raw"] for row in results])), 4),
        "median_corrected": round(float(np.median([row["corrected"] for row in results])), 4),
    }


def _validate_guarded(ms, groups, perceptual=False):
    """Use different image regions, not neighboring pixels, for boxed-area validation."""
    results = []
    if len(groups) == 1:
        source, target, tiles, _ = groups[0]
        for fold in range(5):
            held = tiles % 5 == fold
            if held.sum() < 100 or (~held).sum() < 10000:
                continue
            train = (source[~held], target[~held], tiles[~held], True)
            model = _fit_guarded_balanced(ms, [train], perceptual)
            chosen = np.flatnonzero(held)
            if len(chosen) > 4000:
                chosen = np.random.default_rng(800 + fold).choice(chosen, 4000, replace=False)
            results.append({
                "raw": _metric(source[chosen], target[chosen]),
                "corrected": _metric(_apply_mapping(ms, source[chosen], model), target[chosen]),
            })
    else:
        folds = min(5, len(groups))
        for fold in range(folds):
            train = [group for index, group in enumerate(groups) if index % folds != fold]
            model = _fit_guarded_balanced(ms, train, perceptual)
            for index, group in enumerate(groups):
                if index % folds != fold:
                    continue
                source, target = group[:2]
                chosen = np.random.default_rng(index + 619).choice(
                    len(source), min(4000, len(source)), replace=False)
                results.append({
                    "raw": _metric(source[chosen], target[chosen]),
                    "corrected": _metric(_apply_mapping(ms, source[chosen], model), target[chosen]),
                })
    if len(results) < (2 if len(groups) == 1 else len(groups)):
        return {"pages": results, "improved": 0, "total": len(results),
                "median_raw": 0, "median_corrected": 1e9}
    return {
        "pages": results,
        "improved": sum(row["corrected"] < row["raw"] for row in results),
        "total": len(results),
        "median_raw": round(float(np.median([row["raw"] for row in results])), 4),
        "median_corrected": round(float(np.median([row["corrected"] for row in results])), 4),
    }


def _resize_rows(field, width, height, top, bottom, interpolation=cv2.INTER_LINEAR):
    small_h, small_w = field.shape[:2]
    xs = ((np.arange(width, dtype=np.float32) + .5) * small_w / width - .5)[None, :]
    ys = ((np.arange(top, bottom, dtype=np.float32) + .5) * small_h / height - .5)[:, None]
    map_x = np.broadcast_to(xs, (bottom - top, width)).copy()
    map_y = np.broadcast_to(ys, (bottom - top, width)).copy()
    return cv2.remap(field, map_x, map_y, interpolation, borderMode=cv2.BORDER_REPLICATE)


def _spatial(ms, scan_rgb, scan_alpha, refs, mapping, cancel, stage_progress=None):
    """Return a low-frequency Lab field and soft alpha in working resolution."""
    first = refs[0]["work"]
    small_h, small_w = first["scan"].shape[:2]
    numer = np.zeros((small_h, small_w, 3), np.float32)
    weight_sum = np.zeros((small_h, small_w), np.float32)
    footprint_union = np.zeros((small_h, small_w), np.uint8)
    global_scan = _apply_mapping_rows(ms, first["scan"], mapping)
    global_lab = cv2.cvtColor(global_scan, cv2.COLOR_RGB2LAB)
    if stage_progress:
        stage_progress(.22)
    before_errors, target_samples = [], []
    conflicting_references = False
    if len(refs) > 1:
        for first_index, first_ref in enumerate(refs):
            for second_ref in refs[first_index + 1:]:
                overlap = first_ref["work"]["candidate"] & second_ref["work"]["candidate"]
                if np.count_nonzero(overlap) < 500:
                    continue
                first_lab = cv2.cvtColor(first_ref["work"]["reference"], cv2.COLOR_RGB2LAB)
                second_lab = cv2.cvtColor(second_ref["work"]["reference"], cv2.COLOR_RGB2LAB)
                disagreement = np.linalg.norm(first_lab[overlap] - second_lab[overlap], axis=1)
                if float(np.median(disagreement)) > 6.0:
                    conflicting_references = True
    if stage_progress:
        stage_progress(.25)
    for item in refs:
        _check_cancel(cancel)
        work = item["work"]
        reference_lab = cv2.cvtColor(work["reference"], cv2.COLOR_RGB2LAB)
        residual = reference_lab - global_lab
        residual[:, :, 0] = np.clip(residual[:, :, 0], -7, 7)
        residual[:, :, 1:] = np.clip(residual[:, :, 1:], -6, 6)
        magnitude = np.linalg.norm(residual / np.array([8, 7, 7], np.float32), axis=2)
        weight = work["valid"].astype(np.float32)
        weight *= np.exp(-np.square(work["edge"] / work["limit"]))
        weight *= np.exp(-np.square(magnitude / 2.2))
        numer += residual * weight[:, :, None]
        weight_sum += weight
        footprint_union |= work["valid"].astype(np.uint8)
        indexes = np.flatnonzero(work["candidate"])
        rng = np.random.default_rng(int(item["ref"]["id"][:8], 16))
        indexes = rng.choice(indexes, min(20000, len(indexes)), replace=False)
        before_errors.append(global_scan.reshape(-1, 3)[indexes])
        target_samples.append(work["reference"].reshape(-1, 3)[indexes])
        item["evaluation_indexes"] = indexes
    if stage_progress:
        stage_progress(.31)
    sigma = max(24.0, 230.0 * first["scale"])
    denominator = cv2.GaussianBlur(weight_sum, (0, 0), sigmaX=sigma, sigmaY=sigma)
    field = np.stack([
        cv2.GaussianBlur(numer[:, :, channel], (0, 0), sigmaX=sigma, sigmaY=sigma)
        / np.maximum(denominator, 1e-5)
        for channel in range(3)
    ], axis=2)
    if stage_progress:
        stage_progress(.38)
    field[:, :, 0] = np.clip(field[:, :, 0], -7.6, 7.6)
    field[:, :, 1:] = np.clip(field[:, :, 1:], -9, 9)
    inside = footprint_union > 0
    support_scale = max(float(np.percentile(denominator[inside], 75)), 1e-5)
    alpha = np.clip(denominator / support_scale, 0, 1)
    padded = np.pad(footprint_union, 1, constant_values=1)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    taper = np.clip(distance / max(1.0, 300.0 * first["scale"]), 0, 1)
    taper = taper * taper * (3 - 2 * taper)
    alpha *= taper
    alpha[~inside] = 0
    targets = np.concatenate(target_samples)
    before = _metric(np.concatenate(before_errors), targets)
    all_indexes = np.concatenate([item["evaluation_indexes"] for item in refs])
    raw = _metric(first["scan"].reshape(-1, 3)[all_indexes], targets)
    after_values, after_targets = [], []
    corrected = cv2.cvtColor(global_lab + field, cv2.COLOR_LAB2RGB)
    combined = corrected * alpha[:, :, None] + global_scan * (1 - alpha[:, :, None])
    for index, item in enumerate(refs):
        chosen = item["evaluation_indexes"]
        after_values.append(combined.reshape(-1, 3)[chosen])
        after_targets.append(target_samples[index])
    after = _metric(np.concatenate(after_values), np.concatenate(after_targets))
    accepted = after + .02 < before and not conflicting_references
    if stage_progress:
        stage_progress(.45)
    return (field, alpha) if accepted else (None, None), {
        "raw_delta_e76": raw, "global_delta_e76": before, "spatial_delta_e76": after,
        "accepted": accepted, "reference_conflict": conflicting_references,
        # Popped by the caller and saved, so later results are measured on these pixels.
        "_samples": (all_indexes.astype(np.int64), targets.astype(np.float32),
                     (small_w, small_h)),
    }


DEFAULT_PASSES = 2
MAX_PASSES = 5
FIT_METRICS = ("de76", "rgb")
FIELD_REDUCTION = 4
AUTO_STRENGTH_STEP = 5


def fit_settings(data: dict) -> tuple[str, int]:
    """Project-wide fitting target and number of global passes."""
    metric = data.get("fit_metric", "de76")
    if metric not in FIT_METRICS:
        metric = "de76"
    try:
        passes = int(data.get("passes", DEFAULT_PASSES))
    except (TypeError, ValueError):
        passes = DEFAULT_PASSES
    return metric, max(1, min(MAX_PASSES, passes))


def global_strength(scan: dict, state: dict | None) -> int:
    """Strength the global result is shown at: the user's value, else the automatic one.

    100% is one pass; every further 100% is one more pass, blended in between.
    """
    state = state or {}
    explicit = scan.get("global_strength")
    if explicit is not None:
        return max(0, min(MAX_PASSES * 100, int(explicit)))
    return int(state.get("auto_strength", 100 * max(1, state.get("passes", 1))))


def can_extend_passes(scan: dict) -> bool:
    """Only an Own LUT learns from this scan alone, so only it can get more passes later."""
    return scan["mode"] == "own" and bool(scan["refs"])


def _apply_pass(ms, rgb, step, width, height, top, bottom):
    """One global pass on rows [top, bottom) of a width x height image: LUT, then spatial."""
    out = _apply_mapping(ms, rgb, step["mapping"])
    field = step.get("field")
    if field is not None:
        field_rows = _resize_rows(field, width, height, top, bottom, cv2.INTER_CUBIC)
        alpha_rows = np.clip(_resize_rows(step["alpha"], width, height, top, bottom), 0, 1)
        alpha_rows[alpha_rows < (0.5 / 255)] = 0
        moved = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_RGB2LAB) + field_rows, cv2.COLOR_LAB2RGB)
        out = moved * alpha_rows[:, :, None] + out * (1 - alpha_rows[:, :, None])
    return np.clip(out, 0, 1).astype(np.float32, copy=False)


def _render_rows(ms, src, chain, strength, width, height, top, bottom):
    """The global result at ``strength``: whole passes, then a blend into the next one."""
    whole, fraction = divmod(max(0.0, float(strength)), 100.0)
    whole = int(whole)
    if whole >= len(chain):
        whole, fraction = len(chain), 0.0
    state = src
    for step in chain[:whole]:
        state = _apply_pass(ms, state, step, width, height, top, bottom)
    if fraction > 0:
        following = _apply_pass(ms, state, chain[whole], width, height, top, bottom)
        state = state + (fraction / 100.0) * (following - state)
    return state


def _render_bands(scan_rgb, cancel, workers, row_progress, band):
    """Run ``band(top, bottom)`` over 64-row bands on a thread pool.

    The heavy work (numpy, OpenCV) releases the GIL, and threads start instantly,
    where spawned processes spent seconds importing numpy and OpenCV each time.
    """
    height = scan_rgb.shape[0]
    lock = threading.Lock()
    done = [0]

    def run(top):
        _check_cancel(cancel)
        bottom = min(height, top + 64)
        band(top, bottom)
        if row_progress:
            with lock:
                done[0] += bottom - top
                rows = done[0]
            row_progress(rows)
    workers = min(max(1, workers), max(1, math.ceil(height / 64)))
    if workers == 1:
        for top in range(0, height, 64):
            run(top)
        return
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="compose") as pool:
        for future in [pool.submit(run, top) for top in range(0, height, 64)]:
            future.result()


def _compose(ms, scan_rgb, scan_alpha, info, chain, strength, destination, cancel,
             workers=1, row_progress=None):
    """Render the global result at ``strength``, save it, and return the pixels."""
    height, width = scan_rgb.shape[:2]
    output = np.empty_like(scan_rgb)

    def band(top, bottom):
        src = scan_rgb[top:bottom].astype(np.float32) / 255.0
        result = _render_rows(ms, src, chain, strength, width, height, top, bottom)
        output[top:bottom] = np.clip(np.round(result * 255), 0, 255).astype(np.uint8)
    _render_bands(scan_rgb, cancel, workers, row_progress, band)
    _check_cancel(cancel)
    # An internal cache image: no compression is fastest (exports are encoded separately).
    _save_png(destination, output, scan_alpha, info, compress_level=0)
    return output


# ---------------------------------------------------------------------------
# Saved working data: one run keeps what later passes need, so adding a pass
# (or showing a different strength) never decodes, aligns or warps again.

def _work_path(run_dir: Path, scan_id: str) -> Path:
    return run_dir / "work" / f"{scan_id}.npz"


def _passes_dir(run_dir: Path, scan_id: str) -> Path:
    return run_dir / "passes" / scan_id


def _save_work(run_dir, scan_id, works, ref_ids, plan, guarded):
    path = _work_path(run_dir, scan_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "scan": np.round(works[0]["scan"] * 255).astype(np.uint8),
        "ref_ids": np.array(ref_ids), "guarded": np.array(bool(guarded)),
        "plan_pixels": plan[0], "plan_targets": plan[1].astype(np.float32),
        "plan_tiles": plan[2],
    }
    for index, work in enumerate(works):
        arrays[f"r{index}_reference"] = work["reference"].astype(np.float16)
        arrays[f"r{index}_valid"] = work["valid"]
        arrays[f"r{index}_candidate"] = work["candidate"]
        arrays[f"r{index}_edge"] = work["edge"].astype(np.float16)
        arrays[f"r{index}_meta"] = np.array([work["limit"], work["scale"]], np.float64)
    temporary = path.with_name(path.stem + "-tmp.npz")
    np.savez(temporary, **arrays)
    replace_file(temporary, path)


def _load_work(run_dir, scan_id):
    with np.load(_work_path(run_dir, scan_id)) as data:
        scan = data["scan"].astype(np.float32) / 255.0
        refs = []
        for index, ref_id in enumerate(data["ref_ids"]):
            limit, scale = data[f"r{index}_meta"]
            refs.append({"ref": {"id": str(ref_id)}, "work": {
                "scan": scan,
                "reference": data[f"r{index}_reference"].astype(np.float32),
                "valid": data[f"r{index}_valid"], "candidate": data[f"r{index}_candidate"],
                "edge": data[f"r{index}_edge"].astype(np.float32),
                "limit": float(limit), "scale": float(scale)}})
        plan = (data["plan_pixels"], data["plan_targets"], data["plan_tiles"])
        return scan, refs, plan, bool(data["guarded"])


def _sample_plan(ref_samples, seed: int):
    """Which working-resolution pixels train the LUT (same choice as _sample_group).

    Pixel positions, not colors: every pass reads them again from its own input.
    """
    rng = np.random.default_rng(seed)
    per_ref = max(1000, 30000 // len(ref_samples))
    pixel_parts, target_parts, tile_parts = [], [], []
    for pixels, target, tiles in ref_samples:
        chosen = rng.choice(len(pixels), min(per_ref, len(pixels)), replace=False)
        pixel_parts.append(pixels[chosen])
        target_parts.append(target[chosen])
        tile_parts.append(tiles[chosen])
    pixels = np.concatenate(pixel_parts)
    target = np.concatenate(target_parts)
    tiles = np.concatenate(tile_parts)
    chosen = rng.choice(len(pixels), min(30000, len(pixels)), replace=False)
    return pixels[chosen], target[chosen], tiles[chosen]


def load_chain(run_dir: Path, scan: dict) -> list[dict]:
    """The global passes a scan's result is made of, in order."""
    if scan["refs"]:
        source, with_spatial = scan["id"], True
    elif scan["mode"] == "follow" and scan.get("follow"):
        source, with_spatial = scan["follow"], False
    else:
        shared = run_dir / "lut" / "shared.npy"
        return [{"mapping": np.load(shared)}] if shared.exists() else []
    chain = []
    directory = _passes_dir(run_dir, source)
    while (directory / f"p{len(chain) + 1}.npz").exists():
        with np.load(directory / f"p{len(chain) + 1}.npz") as data:
            step = {"mapping": data["mapping"]}
            if with_spatial and "field" in data:
                step["field"], step["alpha"] = data["field"], data["alpha"]
        chain.append(step)
    if not chain and not with_spatial and (run_dir / "lut" / f"{source}.npy").exists():
        # A run written before global passes existed: one LUT per Own scan.
        chain = [{"mapping": np.load(run_dir / "lut" / f"{source}.npy")}]
    return chain


def _fit_validated(ms, groups, metric, warnings=None):
    """Fit the LUT that held-out validation accepts, trying the chosen target first.

    Validation always measures ΔE76, whatever the LUT was fitted to.
    """
    guarded = any(group[3] for group in groups)
    order = ("perceptual", "rgb") if metric == "de76" else ("rgb", "perceptual")
    last = None
    for method in order:
        perceptual = method == "perceptual"
        try:
            validation = (_validate_guarded(ms, groups, perceptual) if guarded else
                          _validate_mapping(ms, groups, perceptual))
        except ImportError:
            if warnings is not None:
                warnings.append("Perceptual fitting requires scipy; install requirements.txt")
            continue
        last = validation
        if (validation["median_corrected"] < validation["median_raw"]
                and validation["improved"] >= math.ceil(.5 * validation["total"])):
            mapping = (_fit_guarded_balanced(ms, groups, perceptual) if guarded else
                       _fit_balanced(ms, groups, perceptual))
            return mapping, validation, ("guarded " if guarded else "") + method
    return None, last, None


def _auto_strength(eval_values, eval_targets, passes):
    """Strength (0..100 x passes, 5% steps) with the lowest ΔE76 on the evaluation pixels."""
    scores = {}
    for strength in range(0, 100 * passes + 1, AUTO_STRENGTH_STEP):
        whole, fraction = divmod(strength, 100)
        if whole >= passes:
            values = eval_values[passes]
        else:
            values = eval_values[whole] + fraction / 100 * (eval_values[whole + 1] - eval_values[whole])
        scores[strength] = _metric(np.clip(values, 0, 1), eval_targets)
    best = min(scores, key=lambda value: (scores[value], -value))
    return best, scores


def _advance_passes(ms, scan, run_dir, target, metric, shared_mapping, cancel, warnings=None,
                    pass_done=None):
    """Fit passes for one scan with references until ``target`` (or validation stops it).

    Pass k fits a LUT and a spatial field to the output of pass k - 1 at working
    resolution. A later pass that held-out validation rejects ends the sequence.
    Returns the scan's report updates.
    """
    directory = _passes_dir(run_dir, scan["id"])
    directory.mkdir(parents=True, exist_ok=True)
    scan_small, refs, plan, guarded = _load_work(run_dir, scan["id"])
    height, width = scan_small.shape[:2]
    evaluation = run_dir / "evaluation" / f"{scan['id']}.npz"
    state_path = run_dir / "work" / f"{scan['id']}-state.npy"
    done = len(load_chain(run_dir, scan))
    state = np.load(state_path).astype(np.float32) if done else scan_small
    history = []
    if done:
        with np.load(directory / "summary.npz") as data:
            eval_values = list(data["eval_values"].astype(np.float32))
        history = json.loads((directory / "passes.json").read_text(encoding="utf-8"))
    updates = {}
    stopped = None
    for number in range(done + 1, target + 1):
        _check_cancel(cancel)
        group = (state.reshape(-1, 3)[plan[0]], plan[1], plan[2], guarded)
        if scan["mode"] == "own":
            mapping, validation, method = _fit_validated(ms, [group], metric, warnings)
            if mapping is None:
                if number == 1:
                    return {"error": ("Restricted color sampling did not improve held-out regions"
                                      if guarded else "Own LUT failed held-out validation"),
                            "own_validation": validation, "guarded_color_support": guarded}
                # A whole extra pass overshoots, but part of it may still help: compute it
                # anyway, only as the end point the strength can blend towards, and stop.
                perceptual = metric == "de76"
                mapping = (_fit_guarded_balanced(ms, [group], perceptual) if guarded else
                           _fit_balanced(ms, [group], perceptual))
                method = ("guarded " if guarded else "") + ("perceptual" if perceptual else "rgb")
                stopped = number
        else:
            if number > 1 or shared_mapping is None:
                break
            mapping, validation, method = shared_mapping, None, "shared"
        spatial_refs = [{"ref": item["ref"], "work": {**item["work"], "scan": state}} for item in refs]
        (field, alpha), spatial = _spatial(ms, None, None, spatial_refs, mapping, cancel)
        samples = spatial.pop("_samples")
        step = {"mapping": np.asarray(mapping, np.float32 if np.ndim(mapping) == 4 else np.float64)}
        if field is not None:
            reduced = (max(1, round(width / FIELD_REDUCTION)), max(1, round(height / FIELD_REDUCTION)))
            step["field"] = cv2.resize(field, reduced, interpolation=cv2.INTER_AREA).astype(np.float32)
            step["alpha"] = cv2.resize(alpha, reduced, interpolation=cv2.INTER_AREA).astype(np.float32)
        if number == 1:
            indexes, targets, size = samples
            evaluation.parent.mkdir(parents=True, exist_ok=True)
            np.savez(evaluation, indexes=indexes, targets=targets, size=np.array(size))
            eval_values = [state.reshape(-1, 3)[indexes].astype(np.float32)]
            updates["spatial"] = spatial
            if validation is not None:
                updates["own_validation"] = validation
        with np.load(evaluation) as data:
            eval_indexes = data["indexes"]
        state = _apply_pass(ms, state, step, width, height, 0, height)
        eval_values.append(state.reshape(-1, 3)[eval_indexes].astype(np.float32))
        temporary = directory / f"p{number}-tmp.npz"
        np.savez(temporary, **step)
        replace_file(temporary, directory / f"p{number}.npz")
        history.append({"pass": number, "method": method, "validation": validation,
                        "validated": stopped != number,
                        "spatial_used": field is not None,
                        "spatial_delta_e76": spatial.get("spatial_delta_e76")})
        if pass_done:
            pass_done(number)
        if stopped:
            break
    passes = done + sum(1 for row in history if row["pass"] > done)
    if passes == 0:
        return {"error": "Shared LUT required"}
    np.save(state_path, state.astype(np.float16))
    temporary = directory / "summary-tmp.npz"
    np.savez(temporary, eval_values=np.stack(eval_values).astype(np.float16))
    replace_file(temporary, directory / "summary.npz")
    (directory / "passes.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    with np.load(evaluation) as data:
        eval_targets = data["targets"]
    best, scores = _auto_strength(eval_values, eval_targets, passes)
    updates.update({
        "passes": passes, "pass_history": history, "auto_strength": best,
        "strength_scores": {str(key): value for key, value in scores.items()},
        "passes_stopped_at": stopped, "guarded_color_support": guarded,
        "fitting_method": history[0]["method"] if history else None,
    })
    return updates


def _align_scan(ms, scan, run_dir, cancel, previous=None, ref_done=None):
    """Align references (reusing an unchanged earlier alignment) and save working data."""
    _check_cancel(cancel)
    reference_report = {}
    try:
        scan_rgb, scan_alpha, _ = load_image(scan["path"])
    except Exception as exc:
        return scan["id"], None, [], {"error": str(exc)}, reference_report
    previous_dir, previous_refs = previous or (None, {})
    successful, works, ref_samples = [], [], []
    for ref in scan["refs"]:
        _check_cancel(cancel)
        if not ref["enabled"]:
            continue
        try:
            ref_rgb, ref_alpha, _ = load_image(ref["path"])
            signature = alignment_signature(scan, ref)
            earlier = previous_refs.get(ref["id"], {})
            earlier_path = previous_dir / "align" / f"{ref['id']}.npy" if previous_dir else None
            reused = bool(earlier_path and earlier.get("status") == "aligned"
                          and earlier.get("alignment_signature") == signature
                          and earlier_path.exists())
            if reused:
                homography = np.load(earlier_path)
                matches, inliers = earlier["matches"], earlier["inliers"]
                ratio = earlier["inlier_ratio"]
            else:
                homography, matches, inliers, ratio = _align(ms, scan_rgb, ref_rgb)
            if matches < 40 or inliers < 25 or ratio < .25:
                raise ValueError(f"Weak alignment: {inliers}/{matches} inliers")
            work = _aligned_work(ms, scan_rgb, scan_alpha, ref_rgb, ref_alpha, homography,
                                 load_reference_exclusion(run_dir.parents[2], ref))
            del ref_rgb, ref_alpha
            indexes = np.flatnonzero(work["candidate"])
            rng = np.random.default_rng(int(ref["id"][:8], 16))
            indexes = rng.choice(indexes, min(30000, len(indexes)), replace=False)
            target = work["reference"].reshape(-1, 3)[indexes]
            work_width = work["scan"].shape[1]
            tiles = (indexes // work_width // 64) * math.ceil(work_width / 64) + (indexes % work_width // 64)
            ref_samples.append((indexes, target, tiles.astype(np.int32)))
            successful.append(ref)
            works.append(work)
            np.save(run_dir / "align" / f"{ref['id']}.npy", homography)
            preview = np.clip(np.round(work["reference"] * 255), 0, 255).astype(np.uint8)
            _save_png(run_dir / "reference-preview" / f"{ref['id']}.png", preview)
            reference_report[ref["id"]] = {
                "matches": matches, "inliers": inliers, "inlier_ratio": round(ratio, 4),
                "scan_id": scan["id"], "status": "aligned", "alignment_reused": reused,
                "alignment_signature": signature,
                "sampling_mask": reference_mask_mode(ref) if ref.get("exclude_mask") else None,
                "sample_pixels": int(np.count_nonzero(work["candidate"])),
            }
        except InterruptedError:
            raise
        except Exception as exc:
            reference_report[ref["id"]] = {
                "scan_id": scan["id"], "status": "failed", "error": str(exc),
            }
        if ref_done:
            ref_done()
    del scan_rgb, scan_alpha
    group = None
    if successful:
        guarded = any(ref.get("exclude_mask") for ref in successful)
        plan = _sample_plan(ref_samples, int(scan["id"][:8], 16))
        _save_work(run_dir, scan["id"], works, [ref["id"] for ref in successful], plan, guarded)
        group = (works[0]["scan"].reshape(-1, 3)[plan[0]], plan[1], plan[2], guarded)
    return (scan["id"], group, successful,
            {"aligned_references": len(successful)}, reference_report)


def apply_existing_lut(project, scan: dict, cancel: threading.Event | None = None) -> dict:
    """Correct one scan without references with LUTs the active run already fitted.

    Such a scan trains nothing, so adding it (or switching it between the shared LUT
    and following another scan) needs no new calibration: only its own pixels.
    Returns the scan's report entry.
    """
    run = project.run_dir()
    if run is None:
        raise ValueError("Run Color Match first")
    if scan["refs"]:
        raise ValueError("Scans with references need Run Color Match")
    chain = load_chain(run, scan)
    updates = {"passes": len(chain), "auto_strength": 100 * len(chain)}
    strength = global_strength(scan, updates)
    _, result = _correct_scan(_matcher(), scan, run, chain, strength, cancel or threading.Event(), 1)
    updates.update(result)
    return updates


STRENGTH_CACHE_PER_SCAN = 4


def _strength_cache(run_dir: Path, scan_id: str, strength: int) -> Path:
    return run_dir / "strengths" / f"{scan_id}-{int(strength)}.png"


def _link_or_copy(source: Path, destination: Path) -> None:
    """Hard-link when possible (instant, no extra space), else copy; then swap in."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.stem}-{uuid.uuid4().hex[:8]}.tmp{destination.suffix}")
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copyfile(source, temporary)
    replace_file(temporary, destination)


def _remember_strength(run_dir: Path, scan_id: str, strength: int, updates: dict) -> None:
    """Keep a rendered global result, so switching back to its strength is instant.

    Results are always replaced, never rewritten in place, so a hard link stays intact.
    """
    try:
        cache = _strength_cache(run_dir, scan_id, strength)
        _link_or_copy(run_dir / "base" / f"{scan_id}.png", cache)
        cache.with_suffix(".json").write_text(json.dumps(
            {key: updates[key] for key in ("base_delta_e76",) if key in updates}), encoding="utf-8")
        kept = sorted(cache.parent.glob(f"{scan_id}-*.png"), key=lambda item: item.stat().st_mtime)
        for old in kept[:-STRENGTH_CACHE_PER_SCAN]:
            old.unlink(missing_ok=True)
            old.with_suffix(".json").unlink(missing_ok=True)
    except OSError:
        pass  # only a cache


def _correct_scan(ms, scan, run_dir, chain, strength, cancel, compose_workers,
                  stage_progress=None):
    _check_cancel(cancel)
    if not chain:
        return scan["id"], {"error": (
            "Own LUT needs an aligned reference" if scan["mode"] == "own" else
            "The followed scan has no validated Own LUT" if scan["mode"] == "follow" else
            "Shared LUT required")}
    try:
        strength = max(0, min(100 * len(chain), int(strength)))
        scan_rgb, scan_alpha, info = load_image(scan["path"])
        if stage_progress:
            stage_progress(.05)
        destination = run_dir / "base" / f"{scan['id']}.png"
        temporary = destination.with_name(f"{scan['id']}-{uuid.uuid4().hex[:8]}.tmp.png")
        output = _compose(ms, scan_rgb, scan_alpha, info, chain, strength, temporary, cancel,
                          compose_workers,
                          (lambda rows: stage_progress(.05 + .9 * rows / scan_rgb.shape[0]))
                          if stage_progress else None)
        replace_file(temporary, destination)
        updates = {"base_strength": strength}
        evaluation = run_dir / "evaluation" / f"{scan['id']}.npz"
        if scan["refs"] and evaluation.exists():
            updates["base_delta_e76"] = image_delta_e76(output, evaluation)
        if stage_progress:
            stage_progress(1.0)
        updates["status"] = "ready"
        updates["lut"] = scan["mode"] if scan["mode"] in ("own", "follow") else "shared"
        if scan["mode"] == "follow":
            updates["lut_source"] = scan.get("follow")
        _remember_strength(run_dir, scan["id"], strength, updates)
        return scan["id"], updates
    except InterruptedError:
        raise
    except Exception as exc:
        return scan["id"], {"error": str(exc)}


def _correct_scan_in_process(scan, run_dir, chain, strength, compose_workers):
    def report_fraction(fraction):
        if _CORRECTION_PROGRESS_QUEUE is not None:
            _CORRECTION_PROGRESS_QUEUE.put((scan["id"], fraction))
    return _correct_scan(_CORRECTION_MATCHER, scan, run_dir, chain, strength,
                         _CORRECTION_CANCEL, compose_workers, report_fraction)


def _correction_process_entry(scan, run_dir, chain, strength, compose_workers,
                              cancel, progress_queue, result_queue):
    """Run one correction in a plain spawned process, outside ProcessPool internals."""
    try:
        _init_correction_worker(cancel, progress_queue)
        result_queue.put(("completed", _correct_scan_in_process(
            scan, run_dir, chain, strength, compose_workers)))
    except BaseException:
        result_queue.put(("failed", traceback.format_exc()))


class _RunProgress:
    """Progress in estimated work, so the bar moves at an even pace.

    Relative costs measured on real scans: SIFT alignment of one reference, one pass
    of fitting (the ΔE76 fit is iterative), and full-size composition per pixel and pass.
    """

    def __init__(self, progress, end, total=1000):
        self.progress, self.end, self.total = progress, end, total
        self.weights, self.done = {}, {}
        self.lock = threading.Lock()

    def add(self, key, weight):
        self.weights[key] = max(1e-6, float(weight))
        self.done[key] = 0.0

    def set(self, key, fraction, message):
        with self.lock:
            self.done[key] = max(self.done[key], min(1.0, fraction))
            value = sum(self.weights[k] * self.done[k] for k in self.weights)
            share = value / max(1e-6, sum(self.weights.values()))
            self.progress(round(self.end * share), self.total, message)


def _image_megapixels(path: str) -> float:
    try:
        width, height = validate_image(path)
        return width * height / 1e6
    except Exception:
        return 10.0


def run_project(data: dict, directory: Path, cancel: threading.Event, progress: Progress):
    return _run_project_parallel(data, directory, cancel, progress)


def _run_project_parallel(data: dict, directory: Path, cancel: threading.Event, progress: Progress):
    ms = _matcher()
    metric, default_passes = fit_settings(data)
    run_id = uuid.uuid4().hex[:12]
    run_dir = directory / "cache" / "runs" / run_id
    for name in ("base", "lut", "align", "reference-preview", "work", "passes", "evaluation"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    all_scans = [scan for scan in data["scans"] if scan["path"]]
    if not all_scans:
        raise ValueError("Add at least one scan before running")
    previous = None
    earlier_run = data.get("active_run")
    if earlier_run:
        earlier_dir = directory / "cache" / "runs" / earlier_run["id"]
        try:
            earlier_report = json.loads((earlier_dir / "report.json").read_text(encoding="utf-8"))
            previous = (earlier_dir, earlier_report.get("references", {}))
        except (OSError, ValueError):
            previous = None
    report = {"scans": {}, "references": {}, "shared": None, "warnings": [],
              "fit_metric": metric, "passes": default_passes}
    groups, group_refs = {}, {}
    local_steps = sum(len(scan["layers"]) for scan in all_scans)
    tracker = _RunProgress(progress, 950 if local_steps else 1000)

    def target_passes(scan):
        explicit = scan.get("global_strength")
        wanted = max(default_passes, math.ceil(explicit / 100) if explicit else 0)
        return min(MAX_PASSES, wanted) if can_extend_passes(scan) else 1

    fit_cost = 2.5 if metric == "de76" else 1.0
    megapixels = {scan["id"]: _image_megapixels(scan["path"]) for scan in all_scans}
    for scan in all_scans:
        enabled = [ref for ref in scan["refs"] if ref.get("enabled", True)]
        reused = sum(
            1 for ref in enabled
            if previous and previous[1].get(ref["id"], {}).get("alignment_signature")
            and (previous[0] / "align" / f"{ref['id']}.npy").exists())
        tracker.add(("align", scan["id"]), .15 * megapixels[scan["id"]]
                    + 2.5 * (len(enabled) - reused) + .4 * reused)
        if enabled:
            tracker.add(("fit", scan["id"]), (fit_cost + 1.0) * target_passes(scan))
        tracker.add(("compose", scan["id"]), .25 * megapixels[scan["id"]]
                    * (1 + .6 * (target_passes(scan) - 1)))
    workers = min(8, os.cpu_count() or 1, len(all_scans))
    memory_gate = _MemoryGate(_working_memory_budget(), cancel)
    weights = {scan["id"]: _scan_memory_weight(scan) for scan in all_scans}
    tracker.set(("align", all_scans[0]["id"]), 0, f"Aligning with up to {workers} workers · "
                f"{memory_gate.budget / 1e9:.1f} GB image-memory budget")

    def align_job(scan):
        enabled = max(1, sum(ref.get("enabled", True) for ref in scan["refs"]))
        counter = {"done": 0}

        def ref_done():
            counter["done"] += 1
            tracker.set(("align", scan["id"]), .2 + .8 * counter["done"] / enabled,
                        f"Aligned a reference of {Path(scan['path']).name}")
        return _align_scan(ms, scan, run_dir, cancel, previous, ref_done)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="scan-align") as pool:
        futures = {pool.submit(memory_gate.run, weights[scan["id"]], align_job, scan): scan
                   for scan in all_scans}
        for future in as_completed(futures):
            _check_cancel(cancel)
            scan_id, group, successful_refs, scan_report, reference_report = future.result()
            if group is not None:
                groups[scan_id] = group
                group_refs[scan_id] = successful_refs
            report["scans"][scan_id] = scan_report
            report["references"].update(reference_report)
            tracker.set(("align", scan_id), 1, f"Aligned {Path(futures[future]['path']).name}")
    _check_cancel(cancel)
    shared_ids = [
        scan["id"] for scan in all_scans if scan["id"] in groups
        and (scan["mode"] == "shared" or scan.get("include_shared", False))
    ]
    shared_groups = [groups[scan_id] for scan_id in shared_ids]
    shared = None
    if shared_groups:
        tracker.set(("align", all_scans[0]["id"]), 1, "Fitting the shared LUT")
        shared, validation, method = _fit_validated(ms, shared_groups, metric, report["warnings"])
        shared_guarded = any(group[3] for group in shared_groups)
        report["shared"] = {"contributing_scans": len(shared_ids), "validation": validation,
                            "guarded_color_support": shared_guarded, "fitting_method": method}
        if shared is None:
            report["warnings"].append("Shared LUT failed restricted color sampling validation"
                                      if shared_guarded else "Shared LUT failed held-out validation")
            report["shared"]["error"] = "Shared LUT failed validation; try Own LUT"
            if validation:
                report["shared"]["error_detail"] = (
                    "Cross-page correction increased held-out color error. Use Own LUT "
                    "or references with consistent colors. "
                    f"Median page Delta E: {validation['median_raw']:.4f} -> "
                    f"{validation['median_corrected']:.4f}")
            for scan in all_scans:
                if scan["mode"] == "shared":
                    report["scans"][scan["id"]]["error"] = report["shared"]["error"]
    if shared is not None:
        np.save(run_dir / "lut" / "shared.npy", shared)
        _write_mapping_cube(ms, run_dir / "lut" / "shared.cube", shared)

    def fit_job(scan):
        target = target_passes(scan)

        def pass_done(number):
            tracker.set(("fit", scan["id"]), number / target,
                        f"Fitted pass {number} of {Path(scan['path']).name}")
        mapping = None if scan["mode"] == "own" else shared
        return scan["id"], _advance_passes(ms, scan, run_dir, target, metric, mapping, cancel,
                                           report["warnings"], pass_done)

    fitted = [scan for scan in all_scans if scan["id"] in groups
              and (scan["mode"] == "own" or shared is not None)]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="scan-fit") as pool:
        futures = [pool.submit(fit_job, scan) for scan in fitted]
        for future in as_completed(futures):
            _check_cancel(cancel)
            scan_id, updates = future.result()
            entry = report["scans"][scan_id]
            for key, value in updates.items():
                if key == "error":
                    entry.setdefault(key, value)
                else:
                    entry[key] = value
            tracker.set(("fit", scan_id), 1, "Fitted global passes")
    for scan in all_scans:
        if scan["mode"] == "own" and (_passes_dir(run_dir, scan["id"]) / "p1.npz").exists():
            first = load_chain(run_dir, scan)[0]["mapping"]
            np.save(run_dir / "lut" / f"{scan['id']}.npy", first)
            _write_mapping_cube(ms, run_dir / "lut" / f"{scan['id']}.cube", first)
    for scan in all_scans:
        if not scan["refs"]:
            chain = load_chain(run_dir, scan)
            report["scans"][scan["id"]].update({"passes": len(chain), "auto_strength": 100 * len(chain)})

    process_workers = _correction_worker_count(weights, memory_gate.budget, workers)
    total_cpu_workers = min(8, os.cpu_count() or 1)
    compose_workers = max(1, total_cpu_workers // process_workers)
    context = multiprocessing.get_context("spawn")
    process_cancel = context.Event()
    process_progress = context.Queue()
    process_results = context.Queue()
    waiting = list(all_scans)
    active = {}
    tracker.set(("compose", all_scans[0]["id"]), 0,
                f"Correcting with {process_workers} scan process(es) × "
                f"{compose_workers} shared-memory band process(es)")
    try:
        while waiting or active:
            while waiting and len(active) < process_workers:
                scan = waiting.pop(0)
                state = report["scans"][scan["id"]]
                chain = [] if state.get("error") else load_chain(run_dir, scan)
                process = context.Process(
                    target=_correction_process_entry,
                    args=(scan, run_dir, chain, global_strength(scan, state),
                          compose_workers, process_cancel, process_progress, process_results),
                    name=f"scan-correction-{scan['id'][:8]}")
                process.start()
                active[scan["id"]] = (process, scan)
            if cancel.is_set():
                process_cancel.set()
                raise InterruptedError("Color matching was cancelled")
            while True:
                try:
                    scan_id, fraction = process_progress.get_nowait()
                except queue.Empty:
                    break
                tracker.set(("compose", scan_id), fraction, "Correcting full-size scans")
            try:
                kind, payload = process_results.get(timeout=.1)
            except queue.Empty:
                crashed = next(((scan_id, item) for scan_id, item in active.items()
                                if item[0].exitcode not in (None, 0)), None)
                if crashed:
                    raise RuntimeError(
                        f"Correction process exited unexpectedly (code {crashed[1][0].exitcode})")
                continue
            if kind == "failed":
                raise RuntimeError(payload)
            scan_id, updates = payload
            process, scan = active.pop(scan_id)
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(
                    f"Correction process exited unexpectedly (code {process.exitcode})")
            for key, value in updates.items():
                if key == "error":
                    report["scans"][scan_id].setdefault(key, value)
                else:
                    report["scans"][scan_id][key] = value
            tracker.set(("compose", scan_id), 1, f"Corrected {Path(scan['path']).name}")
    finally:
        process_cancel.set()
        for process, _scan in active.values():
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)
        process_progress.close()
        process_results.close()
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    progress(tracker.end, 1000,
             "Color matching complete" if not local_steps else "Refining correction layers")
    return {"id": run_id, "revision": data["calibration_revision"]}, report


def extend_passes(project, scan: dict, target: int, cancel: threading.Event,
                  progress: Progress | None = None) -> dict:
    """Fit more global passes for an Own LUT scan of the active run, reusing its saved data."""
    run = project.run_dir()
    if run is None or not can_extend_passes(scan):
        raise ValueError("Only scans with their own LUT can get more passes")
    metric, _ = fit_settings(project.data)
    target = min(MAX_PASSES, target)
    if progress:
        progress(0, target, "Fitting more global passes...")
    return _advance_passes(_matcher(), scan, run, target, metric, None, cancel, [],
                           (lambda number: progress(number, target, f"Fitted pass {number}"))
                           if progress else None)


def render_global(project, scan: dict, strength: int, cancel: threading.Event,
                  progress: Progress | None = None) -> dict:
    """Recompose a scan's global result at another strength; returns report updates."""
    run = project.run_dir()
    if run is None:
        raise ValueError("Run Color Match first")
    chain = load_chain(run, scan)
    strength = max(0, min(100 * len(chain), int(strength)))
    cache = _strength_cache(run, scan["id"], strength)
    if cache.exists() and cache.with_suffix(".json").exists():
        # Rendered before in this run: swap it back in instead of recomposing.
        _link_or_copy(cache, run / "base" / f"{scan['id']}.png")
        updates = json.loads(cache.with_suffix(".json").read_text(encoding="utf-8"))
        updates.update({"base_strength": strength, "status": "ready",
                        "lut": scan["mode"] if scan["mode"] in ("own", "follow") else "shared"})
        if scan["mode"] == "follow":
            updates["lut_source"] = scan.get("follow")
        if progress:
            progress(1000, 1000, "Reused the rendered global result")
        return updates
    if not chain:
        raise ValueError("Shared LUT required")
    whole, fraction = divmod(strength, 100)
    if whole >= len(chain):
        whole, fraction = len(chain), 0
    needed = [whole] + ([whole + 1] if fraction else [])
    key = (str(run), scan["id"], Path(scan["path"]).stat().st_mtime_ns)
    ends = {k: _ENDPOINTS.get(key + (k,)) for k in needed}
    if progress:
        progress(0, 1000, "Rendering the global result...")
    missing = [k for k, value in ends.items() if value is None]
    meta = _ENDPOINTS.get(key + ("meta",))
    if missing or meta is None:
        rgb, alpha, info = load_image(scan["path"])
        _ENDPOINTS.put(key + ("meta",), (alpha, info, rgb.shape))
    else:
        alpha, info, shape = meta
        rgb = np.empty(shape, np.uint8)  # only its shape is needed below
    if missing:
        # One sweep over the image computes every pass up to the last end needed.
        ms = _matcher()
        height, width = rgb.shape[:2]
        fresh = {k: (rgb.copy() if k == 0 else np.empty_like(rgb)) for k in missing}
        last = max(missing)

        def band(top, bottom):
            state = rgb[top:bottom].astype(np.float32) / 255.0
            for number in range(1, last + 1):
                state = _apply_pass(ms, state, chain[number - 1], width, height, top, bottom)
                if number in fresh:
                    fresh[number][top:bottom] = np.clip(np.round(state * 255), 0, 255).astype(np.uint8)
        _render_bands(rgb, cancel, max(1, min(8, os.cpu_count() or 1)),
                      (lambda rows: progress(round(850 * rows / height), 1000,
                                             "Rendering the global result...")) if progress else None,
                      band)
        for k, value in fresh.items():
            _ENDPOINTS.put(key + (k,), value)
            ends[k] = value
    if fraction:
        # Blending two finished passes: no colour work, a second or so.
        output = np.empty_like(rgb)
        low, high = ends[whole], ends[whole + 1]
        for top in range(0, rgb.shape[0], 256):
            bottom = top + 256
            mixed = low[top:bottom].astype(np.float32)
            mixed += (fraction / 100.0) * (high[top:bottom].astype(np.float32) - mixed)
            output[top:bottom] = np.clip(np.round(mixed), 0, 255).astype(np.uint8)
    else:
        output = ends[whole]
    if progress:
        progress(900, 1000, "Saving the global result...")
    destination = run / "base" / f"{scan['id']}.png"
    temporary = destination.with_name(f"{scan['id']}-{uuid.uuid4().hex[:8]}.tmp.png")
    _save_png(temporary, output, alpha, info, compress_level=0)
    replace_file(temporary, destination)
    updates = {"base_strength": strength, "status": "ready",
               "lut": scan["mode"] if scan["mode"] in ("own", "follow") else "shared"}
    if scan["mode"] == "follow":
        updates["lut_source"] = scan.get("follow")
    evaluation = run / "evaluation" / f"{scan['id']}.npz"
    if scan["refs"] and evaluation.exists():
        updates["base_delta_e76"] = image_delta_e76(output, evaluation)
    _remember_strength(run, scan["id"], strength, updates)
    if progress:
        progress(1000, 1000, "Global result updated")
    return updates


class _EndpointCache:
    """Full-size results of whole passes, so strengths in between only need a blend."""

    def __init__(self, entries=6):
        self.entries = entries
        self.items = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            value = self.items.get(key)
            if value is not None:
                self.items.move_to_end(key)
            return value

    def put(self, key, value):
        with self.lock:
            self.items[key] = value
            self.items.move_to_end(key)
            while len(self.items) > self.entries:
                self.items.popitem(last=False)


_ENDPOINTS = _EndpointCache()


def global_preview(project, scan: dict, strength: int, max_side: int = 1600) -> np.ndarray:
    """A small rendering of the global result at ``strength`` for instant feedback."""
    run = project.run_dir()
    chain = load_chain(run, scan) if run else []
    if run and _work_path(run, scan["id"]).exists():
        # The saved working image already is a small copy of the scan.
        with np.load(_work_path(run, scan["id"])) as data:
            small = data["scan"].astype(np.float32) / 255
    else:
        rgb, _, _ = load_image(scan["path"])
        height, width = rgb.shape[:2]
        scale = min(1.0, max_side / max(height, width))
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        del rgb
    height, width = small.shape[:2]
    result = _render_rows(_matcher(), small, chain, strength, width, height, 0, height)
    return np.clip(np.round(result * 255), 0, 255).astype(np.uint8)


def new_layer(project, scan: dict, ref_id: str) -> dict:
    if not any(ref["id"] == ref_id for ref in scan["refs"]):
        raise ValueError("Reference is not attached to this scan")
    try:
        from .project import new_id
    except ImportError:
        from project import new_id
    layer_id = new_id()
    relative_mask = f"cache/masks/{layer_id}.png"
    width, height = validate_image(scan["path"])
    mask_path = project.directory / relative_mask
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (width, height)).save(mask_path)
    layer = {
        "id": layer_id, "name": f"Correction {len(scan['layers']) + 1}",
        "reference_id": ref_id, "mask": relative_mask,
        "strength": 100, "feather": 24, "hardness": 80, "protect_lines": True,
        "soft_mask": True,
        "visible": True, "overlay_visible": False, "model": None, "model_run": None,
    }
    scan["layers"].append(layer)
    # An unfitted layer has no effect yet: the composed result stays valid.
    layer["needs_refit"] = True
    project.save()
    return layer


def brush_stamps(points: list[tuple[float, float]], radius: int):
    """Round-tip centers at uniform spacing along a stroke.

    Returns ``(stamps, tail)``. ``stamps`` only grows as points are appended, so a
    live preview can apply new stamps incrementally; ``tail`` is the stroke end when it
    is not already covered by the last spaced stamp (it moves with every new point).
    """
    coords = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    spacing = max(1.0, radius * .35)
    previous = coords[0]
    stamps = [previous]
    travelled = 0.0
    for endpoint in coords[1:]:
        delta = endpoint - previous
        length = float(np.hypot(delta[0], delta[1]))
        if length > 1e-6:
            distance = spacing - travelled
            while distance <= length:
                stamps.append(previous + delta * (distance / length))
                distance += spacing
            travelled = length - (distance - spacing)
        previous = endpoint
    tail = coords[-1] if np.hypot(*(coords[-1] - stamps[-1])) > .25 else None
    return stamps, tail


def stamp_coverage(coverage: np.ndarray, x0: int, y0: int, centers, radius: int,
                   feather: int) -> None:
    """Accumulate round brush tips into ``coverage`` (whose top-left is x0, y0) in place."""
    feather = max(0, min(int(feather), radius))
    outer = radius + feather
    inner = radius - feather
    x1, y1 = x0 + coverage.shape[1], y0 + coverage.shape[0]
    for center in centers:
        cx, cy = float(center[0]), float(center[1])
        sx0 = max(x0, int(np.floor(cx - outer - 2)))
        sy0 = max(y0, int(np.floor(cy - outer - 2)))
        sx1 = min(x1, int(np.ceil(cx + outer + 3)))
        sy1 = min(y1, int(np.ceil(cy + outer + 3)))
        if sx0 >= sx1 or sy0 >= sy1:
            continue
        xs = np.arange(sx0, sx1, dtype=np.float32)[None, :]
        for band_y0 in range(sy0, sy1, 256):
            band_y1 = min(sy1, band_y0 + 256)
            ys = np.arange(band_y0, band_y1, dtype=np.float32)[:, None]
            distance = np.hypot(xs - cx, ys - cy)
            if feather:
                t = np.clip((distance - inner) / (2 * feather), 0, 1)
                opacity = 1 - t * t * (3 - 2 * t)
            else:
                opacity = np.clip(radius + .5 - distance, 0, 1)
            target = coverage[band_y0 - y0:band_y1 - y0, sx0 - x0:sx1 - x0]
            target += (1 - target) * opacity


def brush_coverage(shape: tuple[int, int], points: list[tuple[float, float]],
                   radius: int, feather: int):
    """Stamp a round tip at uniform spacing and accumulate overlapping opacity."""
    coords = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    feather = max(0, min(int(feather), radius))
    outer = radius + feather
    x0 = max(0, int(np.floor(coords[:, 0].min() - outer - 2)))
    y0 = max(0, int(np.floor(coords[:, 1].min() - outer - 2)))
    x1 = min(shape[1], int(np.ceil(coords[:, 0].max() + outer + 3)))
    y1 = min(shape[0], int(np.ceil(coords[:, 1].max() + outer + 3)))
    coverage = np.zeros((y1 - y0, x1 - x0), np.float32)
    stamps, tail = brush_stamps(points, radius)
    stamp_coverage(coverage, x0, y0, stamps + ([tail] if tail is not None else []),
                   radius, feather)
    return x0, y0, x1, y1, coverage


def paint_mask(project, scan: dict, layer: dict, points: list[tuple[float, float]], radius: int,
               erase: bool, mask: np.ndarray | None = None, persist: bool = True,
               hardness: int | None = None):
    if not points:
        return mask
    path = project.directory / layer["mask"]
    if mask is None:
        with Image.open(path) as source:
            mask = np.asarray(source.convert("L"), dtype=np.uint8).copy()
    else:
        mask = mask.copy()
    if not layer.get("soft_mask", False):
        mask = np.round(_layer_gate(mask, int(layer["feather"])) * 255).astype(np.uint8)
        layer["soft_mask"] = True
    layer.setdefault("hardness", max(0, 100 - round(100 * int(layer.get("feather", 0)) / max(1, radius))))
    if hardness is not None:
        layer["hardness"] = max(0, min(100, int(hardness)))
    feather = round(radius * (100 - int(layer["hardness"])) / 100)
    x0, y0, x1, y1, coverage = brush_coverage(mask.shape, points, radius, feather)
    region = mask[y0:y1, x0:x1]
    if erase:
        region[:] = np.round(region.astype(np.float32) * (1 - coverage)).astype(np.uint8)
    else:
        region[:] = np.round(region + (255 - region) * coverage).astype(np.uint8)
    if persist:
        temporary = path.with_name(path.stem + ".tmp.png")
        Image.fromarray(mask, "L").save(temporary)
        replace_file(temporary, path)
    if layer.get("model"):  # an unfitted layer has no effect on the result yet
        scan["local_dirty"] = True
    if persist:
        project.save()
    return mask


def polygon_coverage(shape: tuple[int, int], vertices: list[tuple[float, float]],
                     feather: int):
    """Return a polygon mask with a hardness-controlled edge around its boundary.

    The soft edge blurs an antialiased, subpixel-accurate fill with three box passes
    (a compact approximation of a Gaussian, like a feathered selection). Straight edges
    keep their 50% point on the outline and fade to exactly 0 and 1 within about
    +-feather, like the brush; corners round off smoothly instead of forming creases.
    """
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        raise ValueError("Select at least three polygon points")
    if not np.isfinite(points).all() or abs(cv2.contourArea(points.astype(np.float32))) < 1:
        raise ValueError("The polygon must enclose an area")
    feather = max(0, int(feather))
    box_radius = max(1, round(feather / 3)) if feather else 0
    margin = 3 * box_radius + 3
    x0 = max(0, int(np.floor(points[:, 0].min())) - margin)
    y0 = max(0, int(np.floor(points[:, 1].min())) - margin)
    x1 = min(shape[1], int(np.ceil(points[:, 0].max())) + margin + 1)
    y1 = min(shape[0], int(np.ceil(points[:, 1].max())) + margin + 1)
    if x0 >= x1 or y0 >= y1:
        raise ValueError("The polygon is outside the image")
    local = points - (x0, y0)
    fill = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(fill, [np.round(local * 256).astype(np.int32)], 255,
                 lineType=cv2.LINE_AA, shift=8)
    coverage = fill.astype(np.float32) / 255
    if box_radius:
        size = (2 * box_radius + 1, 2 * box_radius + 1)
        for _ in range(3):
            coverage = cv2.blur(coverage, size, borderType=cv2.BORDER_REPLICATE)
    return x0, y0, x1, y1, np.clip(coverage, 0, 1)


def paint_polygon_mask(project, scan: dict, layer: dict,
                       vertices: list[tuple[float, float]], radius: int, hardness: int,
                       mask: np.ndarray | None = None, persist: bool = True):
    path = project.directory / layer["mask"]
    if mask is None:
        with Image.open(path) as source:
            mask = np.asarray(source.convert("L"), dtype=np.uint8).copy()
    else:
        mask = mask.copy()
    if not layer.get("soft_mask", False):
        mask = np.round(_layer_gate(mask, int(layer["feather"])) * 255).astype(np.uint8)
    hardness = max(0, min(100, int(hardness)))
    feather = round(max(0, radius) * (100 - hardness) / 100)
    x0, y0, x1, y1, coverage = polygon_coverage(mask.shape, vertices, feather)
    region = mask[y0:y1, x0:x1]
    np.maximum(region, np.round(coverage * 255).astype(np.uint8), out=region)
    if persist:
        temporary = path.with_name(path.stem + ".tmp.png")
        Image.fromarray(mask, "L").save(temporary)
        replace_file(temporary, path)
    layer["hardness"] = hardness
    layer["soft_mask"] = True
    if layer.get("model"):  # an unfitted layer has no effect on the result yet
        scan["local_dirty"] = True
    if persist:
        project.save()
    return mask


def _layer_gate(mask: np.ndarray, feather: int):
    binary = mask > 0
    if not np.any(binary):
        return np.zeros(mask.shape, np.float32)
    if feather <= 0:
        return binary.astype(np.float32)
    distance = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)
    gate = np.clip(distance / max(1, feather), 0, 1)
    gate = gate * gate * (3 - 2 * gate)
    gate[~binary] = 0
    return gate


def _layer_patch(current: np.ndarray, mask: np.ndarray, layer: dict):
    """Pixels one layer writes on top of ``current``: (x0, y0, painted, values) or None.

    A layer only rewrites its painted pixels, and each new value depends only on
    the current pixel at the same position.
    """
    if (not layer.get("model") or not layer.get("visible", True)
            or layer.get("strength", 100) <= 0):
        return None
    box = _mask_box(mask)
    if box is None:
        return None
    margin = max(2, int(layer["feather"]))
    x0, y0 = max(0, box[0] - margin), max(0, box[1] - margin)
    x1, y1 = min(current.shape[1], box[2] + margin), min(current.shape[0], box[3] + margin)
    rgb = current[y0:y1, x0:x1].astype(np.float32) / 255
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    coeff = np.asarray(layer["model"], np.float32)
    features = np.concatenate((np.ones((*rgb.shape[:2], 1), np.float32), rgb), axis=2)
    correction = features @ coeff
    correction[:, :, 0] = np.clip(correction[:, :, 0], -6, 6)
    correction[:, :, 1:] = np.clip(correction[:, :, 1:], -12, 12)
    if layer.get("soft_mask", False):
        gate = mask[y0:y1, x0:x1].astype(np.float32) / 255
    else:
        gate = _layer_gate(mask[y0:y1, x0:x1], int(layer["feather"]))
    if layer.get("protect_lines", True):
        line_gate = np.clip((lab[:, :, 0] - 12) / 25, 0, 1)
        gate *= line_gate * line_gate * (3 - 2 * line_gate)
    gate *= float(layer["strength"]) / 100
    changed = cv2.cvtColor(lab + correction * gate[:, :, None], cv2.COLOR_LAB2RGB)
    # Do not round-trip unpainted pixels through Lab: they must stay exact.
    painted = gate > 0
    values = np.round(np.clip(changed[painted], 0, 1) * 255).astype(np.uint8)
    return x0, y0, painted, values


def _paste_patch(current: np.ndarray, patch) -> None:
    x0, y0, painted, values = patch
    current[y0:y0 + painted.shape[0], x0:x0 + painted.shape[1]][painted] = values


def _apply_layer(current: np.ndarray, mask: np.ndarray, layer: dict):
    patch = _layer_patch(current, mask, layer)
    if patch is None:
        return current
    output = current.copy()
    _paste_patch(output, patch)
    return output


def _mask_box(mask: np.ndarray):
    """Bounding box (x0, y0, x1, y1) of nonzero mask pixels, or None."""
    rows = np.flatnonzero(mask.any(axis=1))
    if not len(rows):
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


class PatchCache:
    """Size-bounded, thread-safe LRU of decoded images, masks and layer patches."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self._items: OrderedDict = OrderedDict()
        self._sizes: dict = {}
        self._total = 0
        self._lock = threading.Lock()

    @staticmethod
    def _nbytes(value) -> int:
        if isinstance(value, np.ndarray):
            return value.nbytes
        if isinstance(value, tuple):
            return 64 + sum(PatchCache._nbytes(item) for item in value)
        return 64

    def get(self, key):
        with self._lock:
            if key not in self._items:
                return None
            self._items.move_to_end(key)
            return self._items[key]

    def put(self, key, value):
        size = self._nbytes(value)
        with self._lock:
            if key in self._items:
                self._total -= self._sizes.pop(key)
                del self._items[key]
            if size > self.max_bytes:
                return value
            self._items[key] = value
            self._sizes[key] = size
            self._total += size
            while self._total > self.max_bytes:
                old, _ = self._items.popitem(last=False)
                self._total -= self._sizes.pop(old)
        return value

    def clear(self):
        with self._lock:
            self._items.clear()
            self._sizes.clear()
            self._total = 0


def _layer_state(layer: dict) -> str:
    return json.dumps((layer.get("model"), layer.get("strength", 100),
                       None if layer.get("soft_mask", False) else layer.get("feather"),
                       layer.get("soft_mask", False), layer.get("protect_lines", True)),
                      sort_keys=True)


class LayerStack:
    """Apply layers bottom to top onto a base, reusing cached per-layer patches.

    A layer's patch only depends on the pixels under its painted area, so it can be
    reused whenever the base and every lower visible layer that touches that area are
    unchanged. Layers whose masks do not overlap are therefore pure patches that can be
    switched on and off without recomputing anything. The result is identical to
    applying ``_apply_layer`` sequentially.
    """

    def __init__(self, base: np.ndarray, base_key, cache: "PatchCache | None" = None):
        self.current = base.copy()
        self.base_key = base_key
        self.cache = cache
        self._applied = []

    def apply(self, layer: dict, mask: np.ndarray, mask_key) -> None:
        if not layer.get("visible", True):
            return
        box = _mask_box(mask)
        depends_on = []
        for other_box, other_mask, other_key in self._applied:
            if (box is None or other_box is None or box[0] >= other_box[2]
                    or other_box[0] >= box[2] or box[1] >= other_box[3] or other_box[1] >= box[3]):
                continue
            x0, y0 = max(box[0], other_box[0]), max(box[1], other_box[1])
            x1, y1 = min(box[2], other_box[2]), min(box[3], other_box[3])
            if np.any((mask[y0:y1, x0:x1] > 0) & (other_mask[y0:y1, x0:x1] > 0)):
                depends_on.append(other_key)
        key = ("patch", self.base_key, layer.get("id"), _layer_state(layer), mask_key,
               tuple(depends_on))
        patch = self.cache.get(key) if self.cache is not None else None
        if patch is None:
            patch = _layer_patch(self.current, mask, layer) or ()
            if self.cache is not None:
                self.cache.put(key, patch)
        if patch:
            _paste_patch(self.current, patch)
        self._applied.append((box, mask, key))


def compose_layers(base: np.ndarray, layers: list[dict], load_mask, base_key,
                   cache: PatchCache | None = None, cancel: threading.Event | None = None,
                   progress: Progress | None = None) -> np.ndarray:
    """Apply visible layers through a ``LayerStack``.

    ``load_mask(layer)`` returns ``(mask, mask_key)`` with ``mask`` at the base size.
    """
    stack = LayerStack(base, base_key, cache)
    visible = [layer for layer in layers if layer.get("visible", True)]
    for index, layer in enumerate(visible):
        if cancel is not None:
            _check_cancel(cancel)
        stack.apply(layer, *load_mask(layer))
        if progress is not None:
            progress(index + 1, len(visible), f"Updated {layer['name']}")
    return stack.current


def _fit_local(current: np.ndarray, mask: np.ndarray, reference: np.ndarray, valid: np.ndarray, layer: dict):
    ms = _matcher()
    src = current.astype(np.float32) / 255
    target = reference.astype(np.float32) / 255
    sample_mask = mask >= 128 if layer.get("soft_mask", False) else mask > 0
    eroded = cv2.erode(sample_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    selection = eroded & valid
    if np.count_nonzero(selection) < 80:
        raise ValueError("Paint a larger area inside the aligned reference")
    edge = np.maximum(ms.gradient_strength(src), ms.gradient_strength(target))
    cutoff = max(float(np.quantile(edge[selection], .75)), .03)
    selection &= edge <= cutoff
    coordinates = np.flatnonzero(selection)
    if len(coordinates) < 60:
        raise ValueError("Not enough reliable color pixels in the painted area")
    try:
        seed = int(str(layer["id"])[:8], 16)
    except ValueError:
        seed = int.from_bytes(str(layer["id"]).encode("utf-8")[:8].ljust(8, b"\0"), "little")
    rng = np.random.default_rng(seed)
    rng.shuffle(coordinates)
    coordinates = coordinates[:min(30000, len(coordinates))]
    split = max(40, round(len(coordinates) * .8))
    train, test = coordinates[:split], coordinates[split:]
    if len(test) < 12:
        raise ValueError("Not enough validation pixels in the painted area")
    source_flat = src.reshape(-1, 3)
    target_flat = target.reshape(-1, 3)
    current_lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    x = np.column_stack((np.ones(len(train)), source_flat[train])).astype(np.float64)
    y = (target_lab[train] - current_lab[train]).astype(np.float64)
    weights = np.ones(len(train), np.float64)
    coeff = np.zeros((4, 3), np.float64)
    for _ in range(4):
        weighted = x * np.sqrt(weights)[:, None]
        coeff = np.linalg.solve(weighted.T @ weighted + 15 * np.eye(4),
                                weighted.T @ (y * np.sqrt(weights)[:, None]))
        error = np.linalg.norm(x @ coeff - y, axis=1)
        weights = np.minimum(1, 4 / np.maximum(error, 1e-5))
    x_test = np.column_stack((np.ones(len(test)), source_flat[test]))
    predicted = x_test @ coeff
    before = float(np.mean(np.linalg.norm(target_lab[test] - current_lab[test], axis=1)))
    after = float(np.mean(np.linalg.norm(target_lab[test] - current_lab[test] - predicted, axis=1)))
    if after >= before:
        median = np.median(y, axis=0)
        fallback = np.zeros((4, 3), np.float64)
        fallback[0] = median
        fallback_after = float(np.mean(np.linalg.norm(target_lab[test] - current_lab[test] - median, axis=1)))
        if fallback_after >= before:
            raise ValueError("Local correction did not improve held-out pixels")
        coeff, after = fallback, fallback_after
    return coeff.tolist(), {"before_delta_e76": round(before, 3), "after_delta_e76": round(after, 3), "pixels": len(coordinates)}


REFINE_CACHE_VERSION = 2


def _signature(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=str).encode("utf-8")).hexdigest()


def refine_quick_signature(project, scan: dict) -> str:
    """Cheap invalidation key for a fully refined, already rendered scan."""
    run = project.run_dir()
    base = (run / "base" / f"{scan['id']}.png" if scan.get("global_visible", True)
            else Path(scan["path"]))
    base_stat = base.stat() if base.exists() else None
    layers = []
    for layer in scan["layers"]:
        path = project.directory / layer["mask"]
        stat = path.stat() if path.exists() else None
        layers.append((layer["id"], layer.get("reference_id"), layer.get("visible", True),
                       layer.get("strength", 100), layer.get("feather"),
                       layer.get("protect_lines", True), layer.get("soft_mask", False),
                       layer.get("model"), layer.get("model_run"),
                       layer.get("needs_refit", False), layer.get("mask_changed_since_refine", False),
                       layer.get("fit_signature"), layer.get("auto_strength_signature"),
                       layer.get("auto_strength_best"),
                       (stat.st_mtime_ns, stat.st_size) if stat else None))
    inputs = (REFINE_CACHE_VERSION, run.name if run else None,
              scan.get("global_visible", True),
              (base_stat.st_mtime_ns, base_stat.st_size) if base_stat else None, layers)
    exclusions = [project.exclusion_signature(ref) for ref in scan["refs"]
                  if ref.get("exclude_mask")]
    return _signature((*inputs, exclusions) if exclusions else inputs)


def _auto_strength_for_stage(current: np.ndarray, mask: np.ndarray,
                             reference: np.ndarray, valid: np.ndarray,
                             layer: dict, cancel: threading.Event) -> tuple[int, float, float]:
    """Choose the scalar strength after fitting, before later layers are fitted."""
    selected = (mask > 0) & valid
    indexes = np.flatnonzero(selected)
    if len(indexes) < 50:
        raise ValueError(f"Not enough aligned pixels inside {layer['name']}'s mask")
    if len(indexes) > 30000:
        rng = np.random.default_rng(int.from_bytes(
            str(layer["id"]).encode("utf-8")[:8].ljust(8, b"\0"), "little"))
        indexes = rng.choice(indexes, 30000, replace=False)
    source = current.reshape(-1, 3)[indexes].astype(np.float32) / 255
    target = reference.reshape(-1, 3)[indexes].astype(np.float32) / 255
    source_lab = cv2.cvtColor(source.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    target_lab = cv2.cvtColor(target.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    coefficients = np.asarray(layer["model"], np.float32)
    correction = np.column_stack((np.ones(len(source), np.float32), source)) @ coefficients
    correction[:, 0] = np.clip(correction[:, 0], -6, 6)
    correction[:, 1:] = np.clip(correction[:, 1:], -12, 12)
    if layer.get("soft_mask", False):
        gate = mask.reshape(-1)[indexes].astype(np.float32) / 255
    else:
        feather = max(0, int(layer.get("feather", 0)))
        padded = np.pad(mask, 1)
        full_gate = _layer_gate(padded, feather)
        inner = full_gate[1:-1, 1:-1]
        gate = inner.reshape(-1)[indexes]
    if layer.get("protect_lines", True):
        line_gate = np.clip((source_lab[:, 0] - 12) / 25, 0, 1)
        gate *= line_gate * line_gate * (3 - 2 * line_gate)
    current_strength = max(0, min(200, int(layer.get("strength", 100))))
    tested = {}

    def measure(strength: int):
        _check_cancel(cancel)
        if strength in tested:
            return
        if strength == 0:
            lab = source_lab
        else:
            changed = cv2.cvtColor(
                (source_lab + correction * (gate * strength / 100)[:, None])
                .reshape(-1, 1, 3), cv2.COLOR_LAB2RGB).reshape(-1, 3)
            quantized = np.round(np.clip(changed, 0, 1) * 255).astype(np.uint8)
            quantized[gate <= 0] = np.round(source[gate <= 0] * 255).astype(np.uint8)
            lab = cv2.cvtColor((quantized.astype(np.float32) / 255).reshape(-1, 1, 3),
                               cv2.COLOR_RGB2LAB).reshape(-1, 3)
        tested[strength] = float(np.mean(np.linalg.norm(lab - target_lab, axis=1)))

    measure(current_strength)
    for value in range(0, 201, 20):
        measure(value)
    best = min(tested, key=tested.get)
    for value in range(max(0, best - 20), min(200, best + 20) + 1, 5):
        measure(value)
    best = min(tested, key=tested.get)
    for value in range(max(0, best - 5), min(200, best + 5) + 1):
        measure(value)
    best = min(tested, key=tested.get)
    before, after = tested[current_strength], tested[best]
    if before - after < .005:
        best, after = current_strength, before
    return best, round(before, 4), round(after, 4)


def refine_layers(project, scan: dict, cancel: threading.Event, progress: Progress,
                  cache: PatchCache | None = None):
    """Incrementally re-fit and auto-tune layers bottom-to-top, atomically."""
    run = project.run_dir()
    if run is None or not project.is_current():
        raise ValueError("Run Color Match before refining this scan")
    calibrated_path = run / "base" / f"{scan['id']}.png"
    base_path = calibrated_path if scan.get("global_visible", True) else Path(scan["path"])
    if not calibrated_path.exists() or not base_path.exists():
        raise ValueError("This scan has no calibrated result")
    cached_output = scan.get("local_output")
    if (not scan.get("local_dirty") and cached_output
            and (project.directory / cached_output).exists()
            and scan.get("refine_signature") == refine_quick_signature(project, scan)):
        progress(1, 1, "Refine All reused the current result")
        return copy.deepcopy(scan["layers"]), cached_output
    progress(20, 1000, "Preparing layers...")
    (base_rgb, alpha, info), base_key = _cached_file(cache, "image", base_path,
                                                     lambda: load_image(str(base_path)))
    stack = LayerStack(base_rgb, base_key, cache)
    current = stack.current
    layers = copy.deepcopy(scan["layers"])
    calibrated_stat = calibrated_path.stat()
    # The global result changes with its strength: layers fitted on another one refit.
    upstream = _signature((REFINE_CACHE_VERSION, run.name, scan.get("global_visible", True),
                           calibrated_stat.st_mtime_ns, calibrated_stat.st_size))
    share = 820 / max(1, len(layers))
    for index, layer in enumerate(layers):
        _check_cancel(cancel)
        start = 60 + share * index
        progress(round(start), 1000, f"Refining {layer['name']}")
        mask_path = project.directory / layer["mask"]

        def read_mask(path=mask_path):
            with Image.open(path) as image:
                return np.asarray(image.convert("L"), dtype=np.uint8).copy()
        mask, mask_key = _cached_file(cache, "mask", mask_path, read_mask)
        if mask.shape != current.shape[:2]:
            raise ValueError(f"Mask size changed for {layer['name']}")
        if not np.any(mask):
            raise ValueError(f"Paint an area for {layer['name']}")
        ref = next((item for item in scan["refs"] if item["id"] == layer["reference_id"]), None)
        if ref is None:
            raise ValueError(f"Reference is missing for {layer['name']}")
        align_path = run / "align" / f"{ref['id']}.npy"
        if not align_path.exists():
            raise ValueError(f"Reference did not align for {layer['name']}")
        mask_digest = hashlib.sha256(mask.tobytes()).hexdigest()
        fit_inputs = (REFINE_CACHE_VERSION, upstream, ref["id"], mask_digest,
                      layer.get("soft_mask", False))
        exclusion = project.exclusion_signature(ref)
        fit_key = _signature((*fit_inputs, exclusion) if exclusion else fit_inputs)
        ys, xs = np.where(mask > 0)
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
        reuse_fit = (layer.get("fit_signature") == fit_key and layer.get("model")
                     and layer.get("model_run") == run.name
                     and not layer.get("needs_refit", False))
        auto_key = _signature((REFINE_CACHE_VERSION, fit_key, layer.get("feather", 0),
                               layer.get("protect_lines", True)))
        reuse_strength = (reuse_fit and layer.get("auto_strength_signature") == auto_key
                          and layer.get("strength", 100) == layer.get("auto_strength_best"))
        if not reuse_fit or not reuse_strength:
            ref_rgb, ref_alpha, _ = load_image(ref["path"])
            homography = np.load(align_path)
            translate = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
            transform = translate @ homography
            size = (x1 - x0, y1 - y0)
            warped = cv2.warpPerspective(ref_rgb, transform, size)
            valid = cv2.warpPerspective(ref_alpha, transform, size,
                                        flags=cv2.INTER_NEAREST) > 0
            valid = reference_valid(valid, project.directory, ref, transform)
            if not reuse_fit:
                try:
                    model, metrics = _fit_local(current[y0:y1, x0:x1], mask[y0:y1, x0:x1],
                                                warped, valid, layer)
                except ValueError as exc:
                    if ref.get("exclude_mask") and (str(exc).startswith("Not enough") or
                                                    str(exc).startswith("Paint a larger")):
                        raise ValueError(f"{exc} after the reference sampling mask; adjust the sampling mask or enlarge the painted area") from exc
                    raise
                layer["model"] = model
                layer["metrics"] = metrics
                layer["model_run"] = run.name
                layer["needs_refit"] = False
                progress(round(start + share * .6), 1000, f"Choosing the strength of {layer['name']}")
            if not reuse_strength:
                try:
                    strength, before, after = _auto_strength_for_stage(
                        current[y0:y1, x0:x1], mask[y0:y1, x0:x1], warped,
                        valid & (alpha[y0:y1, x0:x1] >= 250), layer, cancel)
                except ValueError as exc:
                    if ref.get("exclude_mask") and str(exc).startswith("Not enough"):
                        raise ValueError(f"{exc} after the reference sampling mask; adjust the sampling mask or enlarge the painted area") from exc
                    raise
                layer["strength"] = strength
                layer["auto_strength_metrics"] = {"before_delta_e76": before,
                                                   "after_delta_e76": after}
        layer["fit_signature"] = fit_key
        layer["auto_strength_signature"] = auto_key
        layer["auto_strength_best"] = layer.get("strength", 100)
        layer["mask_changed_since_refine"] = False
        stack.apply(layer, mask, mask_key)
        if layer.get("visible", True) and layer.get("strength", 100) > 0:
            upstream = _signature((upstream, fit_key, layer["model"],
                                   layer.get("strength", 100), layer.get("feather", 0),
                                   layer.get("protect_lines", True)))
        progress(round(start + share), 1000,
                 f"{'Reused' if reuse_fit and reuse_strength else 'Refined'} {layer['name']}")
    progress(900, 1000, "Saving the result...")
    destination = project.directory / "cache" / "local" / f"{scan['id']}-{uuid.uuid4().hex[:8]}.png"
    _save_png(destination, current, alpha, info, compress_level=0)  # cache file: speed over size
    progress(1000, 1000, "Refine All finished")
    return layers, str(destination.relative_to(project.directory))


def _cached_file(cache: PatchCache | None, kind: str, path: Path, loader):
    stat = path.stat()
    key = (kind, str(path), stat.st_mtime_ns, stat.st_size)
    value = cache.get(key) if cache is not None else None
    if value is None:
        value = loader()
        if cache is not None:
            cache.put(key, value)
    return value, key


def render_layers(project, scan: dict, cancel: threading.Event, progress: Progress,
                  cache: PatchCache | None = None) -> str | None:
    """Recompose existing fitted local models after visibility/strength changes."""
    run = project.run_dir()
    if run is None or not project.is_current():
        raise ValueError("Run Color Match before updating local layers")
    calibrated_path = run / "base" / f"{scan['id']}.png"
    base_path = calibrated_path if scan.get("global_visible", True) else Path(scan["path"])
    if not calibrated_path.exists() or not base_path.exists():
        raise ValueError("This scan has no calibrated result")
    visible = [layer for layer in scan["layers"] if layer.get("visible", True)]
    if not visible:
        return (str(base_path.relative_to(project.directory)) if scan.get("global_visible", True)
                else None)
    # Layers that were never fitted have no effect yet and are simply skipped.
    visible = [layer for layer in visible if layer.get("model")]
    for layer in visible:
        if layer.get("model_run") != run.name or layer.get("needs_refit", False):
            raise ValueError(f"{layer['name']} needs Refine All before it can be shown")
    (rgb, alpha, info), base_key = _cached_file(cache, "image", base_path,
                                                lambda: load_image(str(base_path)))

    def load_mask(layer):
        path = project.directory / layer["mask"]

        def read():
            with Image.open(path) as image:
                return np.asarray(image.convert("L"), dtype=np.uint8).copy()
        mask, key = _cached_file(cache, "mask", path, read)
        if mask.shape != rgb.shape[:2]:
            raise ValueError(f"Mask size changed for {layer['name']}")
        return mask, key

    progress(80, 1000, "Updating local layers...")
    current = compose_layers(
        rgb, visible, load_mask, base_key, cache, cancel,
        lambda index, total, message: progress(80 + round(720 * index / max(1, total)), 1000, message))
    progress(820, 1000, "Saving the result...")
    destination = project.directory / "cache" / "local" / f"{scan['id']}-{uuid.uuid4().hex[:8]}.png"
    _save_png(destination, current, alpha, info, compress_level=0)  # cache file: speed over size
    progress(1000, 1000, "Local layers updated")
    return str(destination.relative_to(project.directory))


def evaluate_result(project, scan: dict, result_path: Path) -> dict | None:
    """Compare base and final on the same stable, aligned reference pixels."""
    run = project.run_dir()
    if run is None or not result_path.exists():
        return None
    base_path = (run / "base" / f"{scan['id']}.png" if scan.get("global_visible", True)
                 else Path(scan["path"]))
    with Image.open(base_path) as source:
        width, height = source.size
        scale = min(1.0, 1200 / max(width, height))
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        base = np.asarray(source.convert("RGBA").resize(size, Image.Resampling.BILINEAR),
                          dtype=np.uint8)
    with Image.open(result_path) as source:
        if source.size != (width, height):
            raise ValueError("Result size differs from the scan")
        final = np.asarray(source.convert("RGBA").resize(size, Image.Resampling.BILINEAR),
                           dtype=np.uint8)
    base_rgb = base[:, :, :3].astype(np.float32) / 255
    final_rgb = final[:, :, :3].astype(np.float32) / 255
    base_gray = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2GRAY)
    base_edge = np.hypot(cv2.Sobel(base_gray, cv2.CV_32F, 1, 0),
                         cv2.Sobel(base_gray, cv2.CV_32F, 0, 1))
    before, after, count = [], [], 0
    output_scale = np.diag([size[0] / width, size[1] / height, 1.0])
    for ref in scan["refs"]:
        align_path = run / "align" / f"{ref['id']}.npy"
        if not ref.get("enabled", True) or not align_path.exists() or not Path(ref["path"]).exists():
            continue
        with Image.open(ref["path"]) as source:
            ref_width, ref_height = source.size
            reference = source.convert("RGBA")
            reference.thumbnail((1600, 1600), Image.Resampling.BILINEAR)
            reference = np.asarray(reference, dtype=np.uint8)
        source_scale = np.diag([ref_width / reference.shape[1],
                                ref_height / reference.shape[0], 1.0])
        homography = np.load(align_path)
        warped = cv2.warpPerspective(reference, output_scale @ homography @ source_scale,
                                     size, flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0, 0))
        valid = (warped[:, :, 3] >= 250) & (base[:, :, 3] >= 250)
        valid = reference_valid(valid, project.directory, ref, output_scale @ homography,
                                 (reference.shape[1], reference.shape[0]))
        if np.count_nonzero(valid) < 100:
            continue
        target_rgb = warped[:, :, :3].astype(np.float32) / 255
        target_gray = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2GRAY)
        target_edge = np.hypot(cv2.Sobel(target_gray, cv2.CV_32F, 1, 0),
                               cv2.Sobel(target_gray, cv2.CV_32F, 0, 1))
        edge = np.maximum(base_edge, target_edge)
        valid &= edge <= np.quantile(edge[valid], .7)
        indexes = np.flatnonzero(valid)
        if len(indexes) < 100:
            continue
        # Keep samples stable even for projects with non-UUID reference IDs.
        rng = np.random.default_rng(int.from_bytes(
            str(ref["id"]).encode("utf-8")[:8].ljust(8, b"\0"), "little"))
        indexes = rng.choice(indexes, min(12000, len(indexes)), replace=False)
        target = target_rgb.reshape(-1, 3)[indexes]
        before.append(_metric(base_rgb.reshape(-1, 3)[indexes], target))
        after.append(_metric(final_rgb.reshape(-1, 3)[indexes], target))
        count += len(indexes)
    if not before:
        return None
    return {"base_delta_e76": round(float(np.mean(before)), 3),
            "final_delta_e76": round(float(np.mean(after)), 3),
            "pixels": count, "references": len(before),
            "baseline": "Global LUT + spatial" if scan.get("global_visible", True) else "Original scan"}


def evaluate_layer_effects(project, scan: dict) -> dict:
    """Measure each fitted layer off/on in its own mask, with all other layers unchanged."""
    run = project.run_dir()
    if run is None:
        return {}
    base_path = (run / "base" / f"{scan['id']}.png" if scan.get("global_visible", True)
                 else Path(scan["path"]))
    if not base_path.exists():
        return {}
    with Image.open(base_path) as image:
        full_size = image.size
    masks = {}
    for layer in scan["layers"]:
        path = project.directory / layer["mask"]
        if path.exists():
            with Image.open(path) as image:
                if image.size == full_size:
                    masks[layer["id"]] = np.asarray(image.convert("L"), dtype=np.uint8)
    scores = {}
    for selected in scan["layers"]:
        mask = masks.get(selected["id"])
        if mask is None or not np.any(mask) or not selected.get("model"):
            continue
        ref = next((item for item in scan["refs"]
                    if item["id"] == selected.get("reference_id")), None)
        align_path = run / "align" / f"{ref['id']}.npy" if ref else None
        if ref is None or not Path(ref["path"]).exists() or not align_path.exists():
            continue
        ys, xs = np.nonzero(mask)
        margin = max(3, *(int(layer.get("feather", 0)) + 3 for layer in scan["layers"]))
        x0, y0 = max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin)
        x1, y1 = min(full_size[0], int(xs.max()) + margin + 1), min(full_size[1], int(ys.max()) + margin + 1)
        bounds = (x0, y0, x1, y1)
        with Image.open(base_path) as image:
            base = np.asarray(image.crop(bounds).convert("RGBA"), dtype=np.uint8)
        off = base[:, :, :3].copy()
        on = off.copy()
        for layer in scan["layers"]:
            layer_mask = masks.get(layer["id"])
            if layer_mask is None:
                continue
            cropped_mask = layer_mask[y0:y1, x0:x1]
            if not np.any(cropped_mask) or not layer.get("model"):
                continue
            if layer["id"] == selected["id"]:
                enabled = dict(layer, visible=True)
                on = _apply_layer(on, cropped_mask, enabled)
            elif layer.get("visible", True):
                off = _apply_layer(off, cropped_mask, layer)
                on = _apply_layer(on, cropped_mask, layer)
        with Image.open(ref["path"]) as image:
            reference = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        translate = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
        warped = cv2.warpPerspective(reference, translate @ np.load(align_path),
                                     (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0, 0))
        selected_mask = mask[y0:y1, x0:x1]
        valid = ((selected_mask > 0) & (base[:, :, 3] >= 250)
                 & (warped[:, :, 3] >= 250))
        valid = reference_valid(valid, project.directory, ref, translate @ np.load(align_path))
        if np.count_nonzero(valid) < 50:
            continue
        # A fixed, layer-local sample avoids the whole-page dilution of small edits.
        indexes = np.flatnonzero(valid)
        if len(indexes) > 30000:
            rng = np.random.default_rng(int.from_bytes(
                str(selected["id"]).encode("utf-8")[:8].ljust(8, b"\0"), "little"))
            indexes = rng.choice(indexes, 30000, replace=False)
        target = warped[:, :, :3].reshape(-1, 3)[indexes].astype(np.float32) / 255
        source_off = off.reshape(-1, 3)[indexes].astype(np.float32) / 255
        source_on = on.reshape(-1, 3)[indexes].astype(np.float32) / 255
        scores[selected["id"]] = {
            "off_delta_e76": _metric(source_off, target),
            "on_delta_e76": _metric(source_on, target),
            "pixels": len(indexes),
        }
    return scores


def optimize_layer_strength(project, scan: dict, layer_id: str, cancel: threading.Event,
                            progress: Progress) -> tuple[int, float, float]:
    """Search 0–200% against the selected reference on fixed pixels inside its mask."""
    run = project.run_dir()
    layer = next((item for item in scan["layers"] if item["id"] == layer_id), None)
    if (run is None or layer is None or not layer.get("model")
            or layer.get("needs_refit", False)
            or layer.get("model_run") != run.name):
        raise ValueError("Use Refine All before Auto Strength")
    ref = next((item for item in scan["refs"] if item["id"] == layer.get("reference_id")), None)
    align_path = run / "align" / f"{ref['id']}.npy" if ref else None
    if ref is None or not align_path.exists() or not Path(ref["path"]).exists():
        raise ValueError("The selected layer has no aligned reference")
    base_path = (run / "base" / f"{scan['id']}.png" if scan.get("global_visible", True)
                 else Path(scan["path"]))
    with Image.open(base_path) as image:
        full_size = image.size
    masks = {}
    for item in scan["layers"]:
        if not item.get("visible", True) and item["id"] != layer_id:
            continue
        if (not item.get("model") or item.get("needs_refit", False)
                or item.get("model_run") != run.name):
            raise ValueError(f"{item['name']} needs Refine All before Auto Strength")
        with Image.open(project.directory / item["mask"]) as image:
            if image.size != full_size:
                raise ValueError(f"Mask size changed for {item['name']}")
            masks[item["id"]] = np.asarray(image.convert("L"), dtype=np.uint8)
    selected_mask = masks[layer_id]
    ys, xs = np.nonzero(selected_mask)
    if not len(xs):
        raise ValueError("Paint the selected layer before using Auto Strength")
    margin = max(3, *(int(item.get("feather", 0)) + 3 for item in scan["layers"]))
    x0, y0 = max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin)
    x1, y1 = min(full_size[0], int(xs.max()) + margin + 1), min(full_size[1], int(ys.max()) + margin + 1)
    with Image.open(base_path) as image:
        base = np.asarray(image.crop((x0, y0, x1, y1)).convert("RGBA"), dtype=np.uint8)
    with Image.open(ref["path"]) as image:
        reference = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    translate = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
    warped = cv2.warpPerspective(reference, translate @ np.load(align_path),
                                 (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))
    valid = ((selected_mask[y0:y1, x0:x1] > 0) & (base[:, :, 3] >= 250)
             & (warped[:, :, 3] >= 250))
    valid = reference_valid(valid, project.directory, ref, translate @ np.load(align_path))
    indexes = np.flatnonzero(valid)
    if len(indexes) < 50:
        if ref.get("exclude_mask"):
            raise ValueError("Not enough aligned pixels allowed by the reference sampling mask inside the selected mask")
        raise ValueError("Not enough aligned pixels inside the selected mask")
    if len(indexes) > 30000:
        rng = np.random.default_rng(int.from_bytes(
            str(layer_id).encode("utf-8")[:8].ljust(8, b"\0"), "little"))
        indexes = rng.choice(indexes, 30000, replace=False)
    base_samples = base[:, :, :3].reshape(-1, 3)[indexes]
    target = warped[:, :, :3].reshape(-1, 3)[indexes].astype(np.float32) / 255
    target_lab = cv2.cvtColor(target.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    stages = []
    for item in scan["layers"]:
        full_mask = masks.get(item["id"])
        if full_mask is None:
            continue
        cropped = full_mask[y0:y1, x0:x1]
        painted = cropped.reshape(-1)[indexes] > 0
        if not np.any(painted):
            continue
        gate = (cropped.astype(np.float32) / 255 if item.get("soft_mask", False)
                else _layer_gate(cropped, int(item.get("feather", 0))))
        stages.append((item, painted, gate.reshape(-1)[indexes][painted],
                       np.asarray(item["model"], np.float32)))

    def score(strength: int) -> float:
        colors = base_samples.copy()
        for item, painted, base_gate, coefficients in stages:
            amount = strength if item["id"] == layer_id else item.get("strength", 100)
            if amount <= 0:
                continue
            rgb = colors[painted].astype(np.float32) / 255
            lab = cv2.cvtColor(rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
            features = np.column_stack((np.ones(len(rgb), np.float32), rgb))
            correction = features @ coefficients
            correction[:, 0] = np.clip(correction[:, 0], -6, 6)
            correction[:, 1:] = np.clip(correction[:, 1:], -12, 12)
            gate = base_gate.copy()
            if item.get("protect_lines", True):
                line_gate = np.clip((lab[:, 0] - 12) / 25, 0, 1)
                gate *= line_gate * line_gate * (3 - 2 * line_gate)
            gate *= amount / 100
            active = gate > 0
            if not np.any(active):
                continue
            changed = cv2.cvtColor((lab[active] + correction[active] * gate[active, None])
                                       .reshape(-1, 1, 3), cv2.COLOR_LAB2RGB).reshape(-1, 3)
            colors[np.flatnonzero(painted)[active]] = np.round(
                np.clip(changed, 0, 1) * 255).astype(np.uint8)
        lab = cv2.cvtColor((colors.astype(np.float32) / 255).reshape(-1, 1, 3),
                           cv2.COLOR_RGB2LAB).reshape(-1, 3)
        return float(np.mean(np.linalg.norm(lab - target_lab, axis=1)))

    current_strength = max(0, min(200, int(layer.get("strength", 100))))
    tested = {}

    def measure(value: int):
        _check_cancel(cancel)
        if value not in tested:
            tested[value] = score(value)
            progress(len(tested), 32, f"Testing strength {value}%")

    measure(current_strength)
    for value in range(0, 201, 20):
        measure(value)
    best = min(tested, key=tested.get)
    for value in range(max(0, best - 20), min(200, best + 20) + 1, 5):
        measure(value)
    best = min(tested, key=tested.get)
    for value in range(max(0, best - 5), min(200, best + 5) + 1):
        measure(value)
    best = min(tested, key=tested.get)
    before, after = tested[current_strength], tested[best]
    if before - after < .005:
        best, after = current_strength, before
    progress(32, 32, "Auto Strength complete")
    return best, round(before, 4), round(after, 4)


def _reference_composite(project, scan: dict) -> Image.Image | None:
    """Every aligned, enabled reference in scan coordinates, combined into one image.

    References that cover different parts of the page fill their own areas; where they
    overlap the earlier one in the list is shown (blending two different pictures would
    leave ghosts). Areas no reference covers stay dark.
    """
    run = project.run_dir()
    total = weight = None
    count = 0
    for ref in scan["refs"]:
        preview_path = run / "reference-preview" / f"{ref['id']}.png"
        align_path = run / "align" / f"{ref['id']}.npy"
        if not ref.get("enabled", True) or not preview_path.exists() or not align_path.exists():
            continue
        with Image.open(preview_path) as image:
            preview = np.asarray(image.convert("RGB"), dtype=np.float32)
        height, width = preview.shape[:2]
        scan_width, _ = validate_image(scan["path"])
        ref_width, ref_height = validate_image(ref["path"])
        scale = width / scan_width
        transform = np.diag([scale, scale, 1.0]) @ np.load(align_path)
        footprint = cv2.warpPerspective(np.ones((ref_height, ref_width), np.float32), transform,
                                        (width, height), flags=cv2.INTER_NEAREST)
        if total is None:
            total = np.zeros_like(preview)
            weight = np.zeros((height, width), np.float32)
        if preview.shape != total.shape:
            continue
        fresh = (footprint > 0) & (weight == 0)
        total[fresh] = preview[fresh]
        weight[fresh] = 1
        count += 1
    if not count:
        return None
    combined = np.full_like(total, (37, 40, 46))
    covered = weight > 0
    combined[covered] = total[covered]
    return Image.fromarray(np.clip(np.round(combined), 0, 255).astype(np.uint8), "RGB")


def _export_preview(project, scan: dict, output: Path, result_path: Path) -> None:
    """A side-by-side sheet: original scan, aligned reference(s), corrected result."""
    panels = [("Original scan", Path(scan["path"]))]
    reference = _reference_composite(project, scan)
    if reference is not None:
        enabled = sum(1 for ref in scan["refs"] if ref.get("enabled", True))
        panels.append((f"Aligned references ({enabled})" if enabled > 1 else "Aligned reference", reference))
    panels.append(("Corrected result", result_path))
    panel_width, panel_height = 450, 660
    sheet = Image.new("RGB", (panel_width * len(panels), panel_height), (37, 40, 46))
    draw = ImageDraw.Draw(sheet)
    for index, (label, source) in enumerate(panels):
        if isinstance(source, Image.Image):
            thumbnail = source.copy()
        else:
            with Image.open(source) as image:
                thumbnail = image.convert("RGB")
        thumbnail.thumbnail((panel_width - 16, panel_height - 42), Image.Resampling.LANCZOS)
        x = index * panel_width + (panel_width - thumbnail.width) // 2
        y = 28 + (panel_height - 42 - thumbnail.height) // 2
        sheet.paste(thumbnail, (x, y))
        draw.text((index * panel_width + 12, 8), label, fill=(240, 240, 240))
    sheet.save(output.with_name(output.stem + "-preview.jpg"), quality=92)


def export_scan(project, scan: dict, destination: Path, image_format: str,
                include_report: bool = False, include_preview: bool = False) -> Path:
    if not project.is_current() or scan.get("local_dirty"):
        raise ValueError("The result is out of date. Run matching or refine the painted layers first")
    run = project.run_dir()
    calibrated_path = run / "base" / f"{scan['id']}.png"
    if not calibrated_path.exists():
        raise ValueError("This scan has no calibrated result")
    local = scan.get("local_output")
    source = (project.directory / local if local else
              calibrated_path if scan.get("global_visible", True)
              else Path(scan["path"]))
    if not source.exists():
        raise ValueError("This scan has no calibrated result")
    destination.mkdir(parents=True, exist_ok=True)
    extension = ".webp" if image_format.lower() == "webp" else ".png"
    stem = Path(scan["path"]).stem + "-color-matched"
    output = destination / f"{stem}{extension}"
    suffix = 2
    while (output.exists()
           or (include_report and output.with_suffix(".json").exists())
           or (include_preview and output.with_name(output.stem + "-preview.jpg").exists())):
        output = destination / f"{stem}-{suffix}{extension}"
        suffix += 1
    # Cached results are stored uncompressed for speed: always encode the export properly.
    with Image.open(source) as image:
        kwargs = ({"lossless": True, "method": 6} if extension == ".webp" else {"compress_level": 6})
        for key in ("icc_profile", "dpi"):
            if image.info.get(key):
                kwargs[key] = image.info[key]
        image.save(output, format="WEBP" if extension == ".webp" else "PNG", **kwargs)
    if include_report:
        report_path = run / "report.json"
        if report_path.exists():
            shutil.copy2(report_path, output.with_suffix(".json"))
    if include_preview:
        _export_preview(project, scan, output, source)
    return output

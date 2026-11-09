#!/usr/bin/env python3
"""Align a printed scan to digital artwork and build global/spatial color corrections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", help="Output prefix; defaults to the scan stem")
    parser.add_argument("--spatial-strength", type=float, default=1.0)
    parser.add_argument("--no-spatial", action="store_true")
    parser.add_argument("--seed", type=int, default=3703)
    return parser.parse_args()


def resize_for_features(rgb: np.ndarray, max_dimension: int = 1800):
    h, w = rgb.shape[:2]
    scale = min(1.0, max_dimension / max(h, w))
    if scale == 1.0:
        return rgb, scale
    resized = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def feature_gray(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def estimate_homography(scan: np.ndarray, reference: np.ndarray):
    scan_small, scan_scale = resize_for_features(scan)
    ref_small, ref_scale = resize_for_features(reference)
    sift = cv2.SIFT_create(nfeatures=14000, contrastThreshold=0.018, edgeThreshold=12)
    kp_scan, desc_scan = sift.detectAndCompute(feature_gray(scan_small), None)
    kp_ref, desc_ref = sift.detectAndCompute(feature_gray(ref_small), None)
    if desc_scan is None or desc_ref is None:
        raise RuntimeError("SIFT could not find descriptors in both images")

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(desc_ref, desc_scan, k=2)
    good = [first for first, second in pairs if first.distance < 0.74 * second.distance]
    if len(good) < 40:
        raise RuntimeError(f"Only {len(good)} reliable feature matches; need at least 40")

    ref_points = np.float32([kp_ref[m.queryIdx].pt for m in good])
    scan_points = np.float32([kp_scan[m.trainIdx].pt for m in good])
    h_small, inlier_mask = cv2.findHomography(ref_points, scan_points, cv2.RANSAC, 4.0)
    if h_small is None or inlier_mask is None:
        raise RuntimeError("Homography estimation failed")
    inliers = int(inlier_mask.sum())
    ratio = inliers / len(good)
    if inliers < 25 or ratio < 0.25:
        raise RuntimeError(f"Weak homography: {inliers}/{len(good)} inliers ({ratio:.3f})")

    scale_ref = np.diag([ref_scale, ref_scale, 1.0])
    scale_scan_inv = np.diag([1.0 / scan_scale, 1.0 / scan_scale, 1.0])
    h_full = scale_scan_inv @ h_small @ scale_ref
    h_full /= h_full[2, 2]
    return h_full, len(good), inliers, ratio


def features(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    return np.column_stack(
        [
            np.ones(len(rgb)), r, g, b,
            r * r, g * g, b * b, r * g, r * b, g * b,
            r**3, g**3, b**3,
            r * r * g, r * r * b, g * g * r, g * g * b,
            b * b * r, b * b * g, r * g * b,
        ]
    )


def identity_weights() -> np.ndarray:
    weights = np.zeros((20, 3), dtype=np.float64)
    weights[1, 0] = 1.0
    weights[2, 1] = 1.0
    weights[3, 2] = 1.0
    return weights


def solve_weighted(x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, ridge: float = 18.0):
    root_weight = np.sqrt(sample_weight)[:, None]
    xw = x * root_weight
    yw = y * root_weight
    lhs = xw.T @ xw + ridge * np.eye(x.shape[1])
    rhs = xw.T @ yw + ridge * identity_weights()
    return np.linalg.solve(lhs, rhs)


def fit_mapping(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    x = features(source.astype(np.float64))
    y = target.astype(np.float64)
    weight = np.ones(len(source), dtype=np.float64)
    mapping = identity_weights()
    for _ in range(4):
        mapping = solve_weighted(x, y, weight)
        residual = np.mean(np.abs(np.clip(x @ mapping, 0, 1) - y), axis=1)
        weight = np.minimum(1.0, 0.035 / np.maximum(residual, 1e-6))
    return mapping


def _lab_with_jacobian(rgb: np.ndarray):
    """Smooth D65 sRGB-to-Lab conversion and its analytic RGB derivative."""
    clipped = np.clip(rgb, 0, 1)
    linear = np.where(clipped <= .04045, clipped / 12.92,
                      ((clipped + .055) / 1.055) ** 2.4)
    slope = np.where(clipped <= .04045, 1 / 12.92,
                     2.4 / 1.055 * ((clipped + .055) / 1.055) ** 1.4)
    slope *= (rgb >= 0) & (rgb <= 1)
    matrix = np.array([[.4124564, .3575761, .1804375],
                       [.2126729, .7151522, .0721750],
                       [.0193339, .1191920, .9503041]])
    matrix /= np.array([.95047, 1, 1.08883])[:, None]
    xyz = linear @ matrix.T
    epsilon = (6 / 29) ** 3
    f = np.where(xyz > epsilon, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    derivative = np.where(xyz > epsilon, 1 / (3 * np.maximum(xyz, epsilon) ** (2 / 3)),
                          1 / (3 * (6 / 29) ** 2))
    lab_matrix = np.array([[0, 116, 0], [500, -500, 0], [0, 200, -200]])
    lab = f @ lab_matrix.T - np.array([16, 0, 0])
    jacobian = np.einsum('ij,nj,jk,nk->nik', lab_matrix, derivative, matrix, slope)
    return lab, jacobian


def fit_perceptual_mapping(source: np.ndarray, target: np.ndarray, iterations: int = 40) -> np.ndarray:
    """Fit the polynomial to the mean Delta E76 used by held-out validation.

    Gauss-Newton on the Lab residuals, reweighted by 1/distance so that it minimizes
    the mean distance rather than its square. Pure numpy: thread-safe (scipy's
    L-BFGS-B before 1.15 is not), and it reaches a lower objective in fewer steps.
    Only training samples enter this optimizer. Callers must independently
    validate the resulting mapping before applying or exporting it.
    """
    x = features(source.astype(np.float64))
    identity = identity_weights()
    target_lab, _ = _lab_with_jacobian(target.astype(np.float64))
    count = len(x)

    def objective(mapping):
        lab, jacobian = _lab_with_jacobian(x @ mapping)
        residual = lab - target_lab
        distance = np.sqrt(np.sum(residual * residual, axis=1) + .01)
        value = float(distance.mean() + .01 * np.sum((mapping - identity) ** 2))
        return value, residual, distance, jacobian

    mapping = fit_mapping(source, target)
    value, residual, distance, jacobian = objective(mapping)
    for _ in range(iterations):
        design = np.einsum("nik,nj->nijk", jacobian, x).reshape(count * 3, 60)
        weighted = design.T * np.repeat(1 / distance, 3)
        lhs = weighted @ design / count + .02 * np.eye(60)
        rhs = -(weighted @ residual.reshape(-1) / count + .02 * (mapping - identity).reshape(-1))
        step = np.linalg.solve(lhs, rhs).reshape(20, 3)
        for scale in (1.0, .5, .25, .125):
            candidate = mapping + scale * step
            trial = objective(candidate)
            if np.isfinite(trial[0]) and trial[0] < value:
                break
        else:
            break
        improvement = value - trial[0]
        mapping = candidate
        value, residual, distance, jacobian = trial
        if improvement < 1e-6 * value:
            break
    return mapping


def apply_mapping(rgb: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    """Evaluate the polynomial in float32, one contiguous plane per term.

    About 4x faster than a per-pixel feature matrix, < 0.001/255 from float64.
    """
    flat = np.asarray(rgb, dtype=np.float32).reshape(-1, 3)
    r, g, b = (np.ascontiguousarray(flat[:, channel]) for channel in range(3))
    x = np.empty((20, len(flat)), np.float32)
    x[0] = 1
    x[1], x[2], x[3] = r, g, b
    np.multiply(r, r, out=x[4])
    np.multiply(g, g, out=x[5])
    np.multiply(b, b, out=x[6])
    np.multiply(r, g, out=x[7])
    np.multiply(r, b, out=x[8])
    np.multiply(g, b, out=x[9])
    np.multiply(x[4], r, out=x[10])
    np.multiply(x[5], g, out=x[11])
    np.multiply(x[6], b, out=x[12])
    np.multiply(x[4], g, out=x[13])
    np.multiply(x[4], b, out=x[14])
    np.multiply(x[5], r, out=x[15])
    np.multiply(x[5], b, out=x[16])
    np.multiply(x[6], r, out=x[17])
    np.multiply(x[6], g, out=x[18])
    np.multiply(x[7], b, out=x[19])
    out = mapping.astype(np.float32).T @ x
    np.clip(out, 0, 1, out=out)
    return np.ascontiguousarray(out.T).reshape(np.shape(rgb))


def write_cube(path: Path, mapping: np.ndarray, size: int = 33) -> None:
    grid = np.linspace(0.0, 1.0, size)
    samples = np.array([(r, g, b) for b in grid for g in grid for r in grid])
    mapped = np.clip(features(samples) @ mapping, 0, 1)
    lines = [
        f'TITLE "{path.stem}"',
        f"LUT_3D_SIZE {size}",
        "DOMAIN_MIN 0.0 0.0 0.0",
        "DOMAIN_MAX 1.0 1.0 1.0",
    ]
    lines.extend(f"{r:.7f} {g:.7f} {b:.7f}" for r, g, b in mapped)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def gradient_strength(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def smooth_residual(values: np.ndarray, weight: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    denominator = cv2.GaussianBlur(weight, (0, 0), sigmaX=sigma, sigmaY=sigma)
    channels = []
    for channel in range(3):
        numerator = cv2.GaussianBlur(values[..., channel] * weight, (0, 0), sigmaX=sigma, sigmaY=sigma)
        channels.append(numerator / np.maximum(denominator, 1e-5))
    return np.stack(channels, axis=2), denominator


def lab_error(reference_lab: np.ndarray, candidate_lab: np.ndarray, mask: np.ndarray) -> list[float]:
    value = np.mean(np.abs(reference_lab - candidate_lab)[mask], axis=0)
    return [round(float(item), 4) for item in value]


def make_comparison(path: Path, panels: list[tuple[str, np.ndarray]], valid: np.ndarray) -> None:
    where = np.where(valid > 0)
    y0, y1 = int(where[0].min()), int(where[0].max()) + 1
    x0, x1 = int(where[1].min()), int(where[1].max()) + 1
    rendered = []
    for label, rgb in panels:
        crop = np.round(np.clip(rgb[y0:y1, x0:x1], 0, 1) * 255).astype(np.uint8)
        width = 480
        height = round(crop.shape[0] * width / crop.shape[1])
        crop = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)
        image = Image.fromarray(crop)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, width, 30), fill=(0, 0, 0))
        draw.text((9, 8), label, fill=(255, 255, 255))
        rendered.append(image)
    sheet = Image.new("RGB", (sum(p.width for p in rendered), max(p.height for p in rendered)), "#222")
    x = 0
    for panel in rendered:
        sheet.paste(panel, (x, 0))
        x += panel.width
    sheet.save(path, quality=94)


def main() -> None:
    args = parse_args()
    if not 0.25 <= args.spatial_strength <= 2.0:
        raise ValueError("--spatial-strength must be between 0.25 and 2.0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.name or args.scan.stem

    scan_u8 = np.asarray(Image.open(args.scan).convert("RGB"), dtype=np.uint8)
    ref_u8 = np.asarray(Image.open(args.reference).convert("RGB"), dtype=np.uint8)
    h, w = scan_u8.shape[:2]
    homography, matches, inliers, inlier_ratio = estimate_homography(scan_u8, ref_u8)

    work_scale = min(0.32, 1700 / max(h, w))
    sw, sh = round(w * work_scale), round(h * work_scale)
    scan_small = cv2.resize(scan_u8, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    ref_float = ref_u8.astype(np.float32) / 255.0
    target_scale = np.diag([work_scale, work_scale, 1.0])
    h_work = target_scale @ homography
    ref_small = cv2.warpPerspective(ref_float, h_work, (sw, sh), flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective(
        np.ones(ref_u8.shape[:2], dtype=np.uint8), h_work, (sw, sh), flags=cv2.INTER_NEAREST
    ).astype(np.float32)
    valid = cv2.erode(valid, np.ones((7, 7), np.uint8), iterations=1)

    edge = np.maximum(gradient_strength(scan_small), gradient_strength(ref_small))
    edge_limit = float(np.quantile(edge[valid > 0], 0.68))
    candidate = (valid > 0) & (edge <= edge_limit)
    coordinates = np.flatnonzero(candidate)
    if len(coordinates) < 10000:
        raise RuntimeError("Too few aligned low-gradient pixels for calibration")
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(coordinates, size=min(225000, len(coordinates)), replace=False)
    source_samples = scan_small.reshape(-1, 3)[chosen]
    target_samples = ref_small.reshape(-1, 3)[chosen]
    mapping = fit_mapping(source_samples, target_samples)

    cube_path = args.output_dir / f"{prefix}-global.cube"
    homography_path = args.output_dir / f"{prefix}-homography.npy"
    comparison_path = args.output_dir / f"{prefix}-comparison.jpg"
    report_path = args.output_dir / f"{prefix}-report.json"
    spatial_path = args.output_dir / f"{prefix}-spatial.png"
    write_cube(cube_path, mapping)
    np.save(homography_path, homography)

    global_small = apply_mapping(scan_small, mapping)
    ref_lab = cv2.cvtColor(np.clip(ref_small, 0, 1), cv2.COLOR_RGB2LAB)
    scan_lab = cv2.cvtColor(scan_small, cv2.COLOR_RGB2LAB)
    global_lab = cv2.cvtColor(global_small, cv2.COLOR_RGB2LAB)
    evaluation_mask = candidate
    spatial_small = global_small

    if not args.no_spatial:
        residual = ref_lab - global_lab
        residual[..., 0] = np.clip(residual[..., 0], -7.0, 7.0)
        residual[..., 1:] = np.clip(residual[..., 1:], -6.0, 6.0)
        magnitude = np.sqrt(
            np.square(residual[..., 0] / 8.0)
            + np.square(residual[..., 1] / 7.0)
            + np.square(residual[..., 2] / 7.0)
        )
        weight = valid * np.exp(-np.square(edge / max(edge_limit, 1e-4)))
        weight *= np.exp(-np.square(magnitude / 2.2))
        sigma = max(24.0, 230.0 * work_scale)
        field, support = smooth_residual(residual, weight, sigma)
        field[..., 0] = np.clip(field[..., 0] * 0.55 * args.spatial_strength, -3.8, 3.8)
        field[..., 1:] = np.clip(field[..., 1:] * args.spatial_strength, -4.5, 4.5)
        support_scale = max(float(np.percentile(support[valid > 0], 75)), 1e-5)
        alpha_small = np.clip(support / support_scale, 0, 1) * valid
        alpha_small = cv2.GaussianBlur(alpha_small, (0, 0), sigmaX=5, sigmaY=5)
        spatial_lab = global_lab + field
        spatial_small = np.clip(cv2.cvtColor(spatial_lab, cv2.COLOR_LAB2RGB), 0, 1)

        field_full = cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC)
        alpha_full = cv2.resize(alpha_small, (w, h), interpolation=cv2.INTER_LINEAR)
        spatial_full = np.empty((h, w, 3), dtype=np.float32)
        for y0 in range(0, h, 192):
            y1 = min(y0 + 192, h)
            source = scan_u8[y0:y1].astype(np.float32) / 255.0
            current = apply_mapping(source, mapping)
            lab = cv2.cvtColor(current, cv2.COLOR_RGB2LAB) + field_full[y0:y1]
            spatial_full[y0:y1] = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1)
        rgba = np.dstack(
            [
                np.round(spatial_full * 255).astype(np.uint8),
                np.round(np.clip(alpha_full, 0, 1) * 255).astype(np.uint8),
            ]
        )
        Image.fromarray(rgba, "RGBA").save(spatial_path, compress_level=5)

    make_comparison(
        comparison_path,
        [("scan", scan_small), ("global LUT", global_small), ("spatial", spatial_small), ("reference", ref_small)],
        valid,
    )

    spatial_lab_for_error = cv2.cvtColor(spatial_small, cv2.COLOR_RGB2LAB)
    report = {
        "scan": str(args.scan),
        "reference": str(args.reference),
        "dimensions": {"scan": [w, h], "reference": [int(ref_u8.shape[1]), int(ref_u8.shape[0])]},
        "alignment": {"good_matches": matches, "inliers": inliers, "inlier_ratio": round(inlier_ratio, 4)},
        "samples": int(len(chosen)),
        "spatial_strength": None if args.no_spatial else args.spatial_strength,
        "mean_abs_lab": {
            "scan": lab_error(ref_lab, scan_lab, evaluation_mask),
            "global": lab_error(ref_lab, global_lab, evaluation_mask),
            "spatial": None if args.no_spatial else lab_error(ref_lab, spatial_lab_for_error, evaluation_mask),
        },
        "outputs": {
            "cube": str(cube_path),
            "spatial": None if args.no_spatial else str(spatial_path),
            "comparison": str(comparison_path),
            "homography": str(homography_path),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

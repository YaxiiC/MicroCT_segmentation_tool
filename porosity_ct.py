"""
Lunar-soil micro-CT porosity pipeline (threshold only, no deep learning).

Inside the particle (rolling-ball envelope, then drop the outer edge band):
    phi_closed = V_closed / (V_solid + V_closed + V_open)
    phi_open   = V_open   / (V_solid + V_closed + V_open)
    phi_total  = phi_closed + phi_open

Open voxels within --exclude-edge-um of the envelope surface are treated as
wrapping, not porosity. Closed vesicles are kept even if they sit near the edge.

Cupping-bright vesicle cores (above air threshold but darker than the local
neighborhood, and touching an air-dark pore) are reclassified from solid to pore.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from skimage.morphology import ball, disk

ROOT = Path(__file__).resolve().parent
# Preferred display order when these folders exist; extras are auto-discovered.
DATASETS = [
    "5-19",
    "5-19 High Resolution",
    "5-21",
    "6-11",
    "6-16",
    "MN Lunar Soil",
]

CLOSED_RGB = np.array([0.00, 0.45, 0.70])
OPEN_RGB = np.array([0.90, 0.62, 0.00])
RIM_RGB = np.array([0.80, 0.12, 0.55])
ENVELOPE_RGB = np.array([0.20, 0.70, 0.35])

# Manual-edit label codes (uint8 volume)
L_BG = 0
L_SOLID = 1
L_CLOSED = 2
L_OPEN = 3
L_WRAP = 4  # surface wrap / ignore — not counted in porosity
LABEL_NAMES = {
    L_BG: "background",
    L_SOLID: "solid",
    L_CLOSED: "closed_pore",
    L_OPEN: "open_pore",
    L_WRAP: "wrap_ignore",
}


def parse_header(folder: Path) -> dict:
    header = folder / "Header.txt"
    info = {}
    if not header.exists():
        return info
    text = header.read_text(encoding="utf-8", errors="ignore")
    for key, cast in [
        ("Pixel Size", float),
        ("Image Width", int),
        ("Image Height", int),
        ("Images Taken", int),
        ("Voltage", float),
        ("Optical Magnification", float),
    ]:
        m = re.search(rf"{re.escape(key)}\s*=\s*([0-9.]+)", text)
        if m:
            info[key] = cast(m.group(1))
    return info


def list_tiffs(folder: Path) -> list[Path]:
    files = sorted(folder.glob("*.tiff")) + sorted(folder.glob("*.tif"))
    return [p for p in files if p.is_file()]


def path_relative_to_root(path: Path | str, root: Path | None = None) -> str:
    """Store portable project-relative paths (forward slashes) in JSON/meta."""
    root = (root or ROOT).resolve()
    path = Path(path).resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        # Outside the project tree — keep absolute (rare)
        return str(path)


def resolve_project_path(
    path_str: str | None,
    root: Path | None = None,
    fallback_name: str | None = None,
) -> Path | None:
    """Resolve a path that may be relative or an absolute path from another PC."""
    root = (root or ROOT).resolve()
    if path_str:
        p = Path(path_str)
        if p.exists():
            return p
        cand = root / path_str
        if cand.exists():
            return cand
        # Absolute path from another machine: try folder name under ROOT
        cand = root / p.name
        if cand.exists():
            return cand
    if fallback_name:
        cand = root / fallback_name
        if cand.exists():
            return cand
    return None


def discover_datasets(root: Path | None = None) -> list[str]:
    """Dataset folders = subdirs of the project that contain TIFF slices."""
    root = root or ROOT
    found: list[str] = []
    if not root.is_dir():
        return list(DATASETS)
    for p in sorted(root.iterdir(), key=lambda x: x.name.lower()):
        if not p.is_dir() or p.name.startswith(("_", ".")):
            continue
        if list_tiffs(p):
            found.append(p.name)
    if not found:
        return list(DATASETS)
    ordered = [d for d in DATASETS if d in found]
    extras = [d for d in found if d not in DATASETS]
    return ordered + extras


def fov_disk(shape_hw: tuple[int, int], margin: float = 0.98) -> np.ndarray:
    h, w = shape_hw
    yy, xx = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = margin * min(cy, cx)
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def _crop_or_pad(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=arr.dtype)
    hh, ww = arr.shape[:2]
    y = min(hh, h)
    x = min(ww, w)
    out[:y, :x] = arr[:y, :x]
    return out


def load_volume(files: list[Path], downsample: int) -> np.ndarray:
    files = files[::downsample]
    heights = []
    widths = []
    probe = files[:: max(1, len(files) // 8)][:9]
    for path in probe:
        arr = np.array(Image.open(path))
        heights.append(arr.shape[0])
        widths.append(arr.shape[1])
    h = int(np.median(heights))
    w = int(np.median(widths))
    out_h, out_w = h // downsample, w // downsample
    vol = np.empty((len(files), out_h, out_w), dtype=np.uint16)
    for i, path in enumerate(files):
        arr = np.array(Image.open(path), dtype=np.uint16)
        arr = _crop_or_pad(arr, h, w)
        vol[i] = arr[::downsample, ::downsample][:out_h, :out_w]
        if (i + 1) % 100 == 0 or i + 1 == len(files):
            print(f"  loaded {i + 1}/{len(files)} slices", flush=True)
    return vol


def pick_otsu(vol: np.ndarray, fov2d: np.ndarray) -> int:
    sampled = vol[::2, ::2, ::2]
    fov_s = fov2d[::2, ::2]
    vals = sampled[:, fov_s]
    vals = vals[vals > 0]
    if vals.size == 0:
        raise RuntimeError("No positive voxels inside FOV.")
    if vals.size > 2_000_000:
        rng = np.random.default_rng(0)
        vals = rng.choice(vals, 2_000_000, replace=False)
    return int(threshold_otsu(vals))


def exterior_air_mask(vol: np.ndarray, fov2d: np.ndarray, otsu_t: int) -> np.ndarray:
    """Air well away from the particle, so partial-volume rims are excluded."""
    union = np.zeros(vol.shape[1:], dtype=bool)
    for z in range(0, vol.shape[0], 4):
        union |= (vol[z] > otsu_t) & fov2d
    halo = ndi.binary_dilation(union, iterations=25)
    return np.broadcast_to(fov2d & ~halo, vol.shape) & (vol > 0)


def pick_air_threshold(vol: np.ndarray, air_mask: np.ndarray, k_std: float = 2.5) -> int:
    vals = vol[air_mask]
    if vals.size < 1000:
        raise RuntimeError("Not enough exterior-air voxels to set threshold.")
    t = float(vals.mean() + k_std * vals.std())
    return int(round(t))


def remove_small(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    if min_voxels <= 1 or not mask.any():
        return mask
    lab, _n = ndi.label(mask)
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_voxels
    keep[0] = False
    return keep[lab]


def _class_stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"n": 0, "mean": None, "median": None, "std": None, "p5": None, "p95": None}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "p5": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def _sample_vals(vol: np.ndarray, mask: np.ndarray, cap: int = 400_000) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return np.array([], dtype=np.float32)
    if idx.size > cap:
        rng = np.random.default_rng(0)
        idx = rng.choice(idx, cap, replace=False)
    return vol.ravel()[idx].astype(np.float32)


def rolling_ball_envelope(solid: np.ndarray, radius_px: float) -> np.ndarray:
    r = max(1, int(round(radius_px)))
    print(f"  rolling-ball envelope R={r} vx ...", flush=True)
    closed = ndi.binary_closing(solid, structure=ball(r))
    return ndi.binary_fill_holes(closed)


def recover_cupping_cores(
    vol: np.ndarray,
    solid: np.ndarray,
    envelope: np.ndarray,
    pore_seeds: np.ndarray,
    voxel_um: float,
    window_um: float = 30.0,
    delta: float = 1500.0,
    min_voxels: int = 27,
) -> np.ndarray:
    """Reclassify cupping-bright vesicle cores currently labeled as solid.

    CT beam-hardening raises gray values inside large vesicles above the air
    threshold, so only a dark rim is segmented as pore.  A voxel is recovered
    if it is solid, inside the envelope, darker than the local mean by `delta`,
    and belongs to a blob that touches an existing pore seed.
    """
    if window_um <= 0 or delta <= 0:
        return np.zeros(solid.shape, dtype=bool)
    size = int(round(window_um / voxel_um))
    size = max(5, size | 1)  # odd kernel
    print(
        f"  cupping recovery: local window={window_um:.0f} um ({size} px), "
        f"delta={delta:.0f} gray ...",
        flush=True,
    )
    cup = np.zeros(solid.shape, dtype=bool)
    for z in range(vol.shape[0]):
        if not solid[z].any():
            continue
        sl = vol[z].astype(np.float32)
        local = ndi.uniform_filter(sl, size=size)
        cup[z] = ((local - sl) >= delta) & solid[z] & envelope[z]
    cup = remove_small(cup, min_voxels)
    if not cup.any() or not pore_seeds.any():
        return cup
    lab, n = ndi.label(cup)
    if n == 0:
        return cup
    touch = ndi.binary_dilation(pore_seeds)
    keep = np.zeros(n + 1, dtype=bool)
    hit = np.unique(lab[touch & (lab > 0)])
    keep[hit] = True
    keep[0] = False
    return keep[lab]


def filter_elongated(
    mask: np.ndarray,
    min_elongation: float = 2.5,
    min_voxels: int = 8,
) -> np.ndarray:
    """Keep only elongated connected components (crack-like), drop round speckles."""
    if not mask.any() or min_elongation <= 1.0:
        return remove_small(mask, min_voxels)
    # Slice-wise: cracks are elongated in-plane; avoids expensive full-3D props
    out = np.zeros(mask.shape, dtype=bool)
    for z in range(mask.shape[0]):
        sl = mask[z]
        if not sl.any():
            continue
        lab, n = ndi.label(sl)
        if n == 0:
            continue
        keep = np.zeros(n + 1, dtype=bool)
        for p in regionprops(lab):
            if p.area < min_voxels:
                continue
            minor = max(p.minor_axis_length, 1e-3)
            if (p.major_axis_length / minor) >= min_elongation:
                keep[p.label] = True
        out[z] = keep[lab]
    return out


def recover_dark_cracks(
    vol: np.ndarray,
    solid: np.ndarray,
    envelope: np.ndarray,
    voxel_um: float,
    window_um: float = 12.0,
    delta: float = 1100.0,
    min_voxels: int = 8,
    min_elongation: float = 2.5,
) -> np.ndarray:
    """Reclassify thin crack / fissure voxels that sit above the global T.

    Partial-volume cracks are often darker than neighboring mineral but still
    brighter than the air (vesicle) threshold. Round noise speckles are rejected
    by an elongation filter so vesicle detection stays separate.
    """
    if window_um <= 0 or delta <= 0:
        return np.zeros(solid.shape, dtype=bool)
    size = int(round(window_um / voxel_um))
    size = max(3, size | 1)
    print(
        f"  crack recovery: local window={window_um:.0f} um ({size} px), "
        f"delta={delta:.0f} gray, min_elong={min_elongation:.1f} ...",
        flush=True,
    )
    cracks = np.zeros(solid.shape, dtype=bool)
    for z in range(vol.shape[0]):
        if not solid[z].any():
            continue
        sl = vol[z].astype(np.float32)
        local = ndi.uniform_filter(sl, size=size)
        cracks[z] = ((local - sl) >= delta) & solid[z] & envelope[z]
    return filter_elongated(cracks, min_elongation=min_elongation, min_voxels=min_voxels)


def analyze_volume(
    vol: np.ndarray,
    pixel_um: float,
    downsample: int,
    thresh: int | None,
    air_k_std: float = 2.5,
    min_pore_voxels: int = 8,
    wall_close_iter: int = 0,
    ball_um: float = 15.0,
    exclude_edge_um: float | None = None,
    cupping_um: float = 40.0,
    cupping_delta: float = 1000.0,
    thresh_mix: float = 0.50,
    open_thresh: int | None = None,
    open_thresh_mix: float | None = None,
    crack_um: float = 12.0,
    crack_delta: float = 0.0,
    crack_elongation: float = 2.5,
) -> dict:
    fov2d = fov_disk(vol.shape[1:])
    voxel_um = pixel_um * downsample
    otsu_t = pick_otsu(vol, fov2d)
    air_mask = exterior_air_mask(vol, fov2d, otsu_t)
    try:
        air_t = pick_air_threshold(vol, air_mask, k_std=air_k_std)
    except RuntimeError:
        air_t = otsu_t
        print("  warning: too little exterior air; falling back to Otsu")
    mix = float(np.clip(thresh_mix, 0.0, 1.0))
    if thresh is None:
        # Blend air and Otsu so cupping-bright vesicle cores fall below T
        # (pure air T leaves bright pore interiors as solid)
        thresh = int(round(air_t + mix * (otsu_t - air_t)))
        print(
            f"  Otsu={otsu_t}  air={air_t} (mean+{air_k_std}*std)  "
            f"closed T={thresh} (mix={mix:.2f})"
        )
    else:
        print(f"  manual closed T={thresh}  (Otsu={otsu_t}, air-calibrated={air_t})")

    if open_thresh is not None:
        t_open = int(open_thresh)
        open_mix_used = None
    else:
        open_mix = mix if open_thresh_mix is None else float(np.clip(open_thresh_mix, 0.0, 1.0))
        t_open = int(round(air_t + open_mix * (otsu_t - air_t)))
        open_mix_used = open_mix
    if t_open == thresh:
        print(f"  open T={t_open} (same as closed)")
    else:
        print(
            f"  open T={t_open}"
            + (f" (open-mix={open_mix_used:.2f})" if open_mix_used is not None else "")
        )

    solid = np.empty(vol.shape, dtype=bool)
    struct2d = ndi.generate_binary_structure(2, 1)
    for z in range(vol.shape[0]):
        sl = (vol[z] > thresh) & fov2d
        if wall_close_iter > 0:
            sl = ndi.binary_closing(sl, structure=struct2d, iterations=wall_close_iter)
        solid[z] = sl

    # Closed vesicles from full threshold solid (same as 2D preview), before
    # restricting to the largest particle for the envelope.
    closed_pore = closed_pores_slicewise(solid, min_pore_voxels)

    labeled, nlab = ndi.label(solid)
    if nlab == 0:
        raise RuntimeError("No solid voxels found. Check threshold.")
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    keep = int(np.argmax(counts))
    solid = labeled == keep
    del labeled

    ball_px = ball_um / voxel_um
    envelope = rolling_ball_envelope(solid, ball_px)
    closed_pore = remove_small(closed_pore & envelope, min_pore_voxels)

    # Air-dark pores first (seeds for cupping growth)
    air_pores = envelope & ~solid
    cupping = recover_cupping_cores(
        vol,
        solid,
        envelope,
        pore_seeds=air_pores,
        voxel_um=voxel_um,
        window_um=cupping_um,
        delta=cupping_delta,
        min_voxels=max(min_pore_voxels, 27),
    )
    n_cupping = int(cupping.sum())
    if n_cupping:
        solid = solid & ~cupping
        closed_pore = closed_pore | cupping
        print(f"  cupping cores reclassified as pore: {n_cupping} voxels", flush=True)

    cracks = recover_dark_cracks(
        vol,
        solid,
        envelope,
        voxel_um=voxel_um,
        window_um=crack_um,
        delta=crack_delta,
        min_voxels=max(8, min_pore_voxels),
        min_elongation=crack_elongation,
    )
    n_cracks = int(cracks.sum())
    if n_cracks:
        solid = solid & ~cracks
        closed_pore = closed_pore | cracks
        print(f"  dark cracks reclassified as pore: {n_cracks} voxels", flush=True)

    # Complete leftover solid interiors enclosed by pore rims (2D)
    pore_now = envelope & ~solid
    ring_fill = np.zeros(solid.shape, dtype=bool)
    for z in range(solid.shape[0]):
        if pore_now[z].any():
            ring_fill[z] = ndi.binary_fill_holes(pore_now[z]) & solid[z]
    ring_fill = remove_small(ring_fill, min_pore_voxels)
    n_ring = int(ring_fill.sum())
    if n_ring:
        solid = solid & ~ring_fill
        closed_pore = closed_pore | ring_fill
        print(f"  pore-ring interiors filled: {n_ring} voxels", flush=True)

    closed_pore = remove_small(closed_pore & envelope, min_pore_voxels)
    print(
        f"  closed pores: slice-wise 2D fill-holes + extras "
        f"({int(closed_pore.sum())} voxels) — matched to setup preview",
        flush=True,
    )
    solid = solid & ~closed_pore
    # Open pores use a separate intensity gate (default: same as closed T).
    # Higher open T → more yellow (dim solid near surface counted as open).
    # Lower open T → less yellow (only darker exterior-connected voids).
    open_all = remove_small(
        envelope & (vol <= t_open) & ~closed_pore, min_pore_voxels
    )
    if exclude_edge_um is None:
        exclude_edge_um = ball_um
    if exclude_edge_um > 0:
        rim_px = max(1, int(round(exclude_edge_um / voxel_um)))
        rim = _envelope_rim(envelope, rim_px)
        open_wrap = open_all & rim
        open_pore = open_all & ~rim
    else:
        rim_px = 0
        open_wrap = np.zeros(open_all.shape, dtype=bool)
        open_pore = open_all
    # Keep class masks exclusive; voxels rejected by open T stay solid
    solid = (solid | (envelope & ~closed_pore & (vol > t_open))) & ~open_pore & ~open_wrap
    env_pore = closed_pore | open_pore
    particle = solid | closed_pore | open_pore
    exterior_air = air_mask

    vox_um3 = voxel_um ** 3
    n_solid = int(solid.sum())
    n_closed = int(closed_pore.sum())
    n_open = int(open_pore.sum())
    n_wrap = int(open_wrap.sum())
    n_open_all = int(open_all.sum())
    n_den = n_solid + n_closed + n_open
    n_den_raw = n_solid + n_closed + n_open_all

    phi_closed = n_closed / n_den if n_den else float("nan")
    phi_open = n_open / n_den if n_den else float("nan")
    phi_total = phi_closed + phi_open if n_den else float("nan")
    phi_closed_raw = n_closed / n_den_raw if n_den_raw else float("nan")
    phi_open_raw = n_open_all / n_den_raw if n_den_raw else float("nan")
    phi_total_raw = phi_closed_raw + phi_open_raw if n_den_raw else float("nan")
    frac_open_excluded = n_wrap / n_open_all if n_open_all else 0.0

    closed_lab, n_closed_cc = ndi.label(closed_pore)
    open_lab, n_open_cc = ndi.label(open_pore)
    closed_sizes = np.bincount(closed_lab.ravel())[1:] if n_closed_cc else np.array([])
    open_sizes = np.bincount(open_lab.ravel())[1:] if n_open_cc else np.array([])

    slice_phi = np.full(vol.shape[0], np.nan, dtype=np.float64)
    for z in range(vol.shape[0]):
        den = int(particle[z].sum())
        if den > 0:
            slice_phi[z] = float(env_pore[z].sum() / den)

    gray = {
        "exterior_air": _class_stats(_sample_vals(vol, exterior_air)),
        "closed_pore": _class_stats(_sample_vals(vol, closed_pore)),
        "open_pore": _class_stats(_sample_vals(vol, open_pore)),
        "open_wrap": _class_stats(_sample_vals(vol, open_wrap)),
        "solid": _class_stats(_sample_vals(vol, solid)),
    }
    print(
        f"  ball={ball_um:.1f} um ({ball_px:.1f} vx)  "
        f"exclude-edge={exclude_edge_um:.1f} um  "
        f"dropped {100 * frac_open_excluded:.1f}% of raw open voxels"
    )
    print(
        f"  counted  closed={100 * phi_closed:.2f}%  open={100 * phi_open:.2f}%  "
        f"total={100 * phi_total:.2f}%"
    )
    print(
        f"  raw wrap-in  closed={100 * phi_closed_raw:.2f}%  "
        f"open={100 * phi_open_raw:.2f}%  total={100 * phi_total_raw:.2f}%"
    )

    return {
        "threshold": thresh,
        "open_threshold": t_open,
        "otsu_threshold": otsu_t,
        "air_threshold": air_t,
        "air_k_std": air_k_std,
        "thresh_mix": float(np.clip(thresh_mix, 0.0, 1.0)),
        "open_thresh_mix": open_mix_used,
        "min_pore_voxels": min_pore_voxels,
        "ball_um": ball_um,
        "ball_px": ball_px,
        "exclude_edge_um": exclude_edge_um,
        "exclude_edge_px": rim_px,
        "cupping_um": cupping_um,
        "cupping_delta": cupping_delta,
        "n_cupping": n_cupping,
        "crack_um": crack_um,
        "crack_delta": crack_delta,
        "crack_elongation": crack_elongation,
        "n_cracks": n_cracks,
        "n_solid": n_solid,
        "n_closed_pore": n_closed,
        "n_open_pore": n_open,
        "n_open_wrap": n_wrap,
        "n_open_all": n_open_all,
        "n_envelope": int(envelope.sum()),
        "n_particle": int(particle.sum()),
        "n_closed_components": int(n_closed_cc),
        "n_open_components": int(n_open_cc),
        "phi_closed": phi_closed,
        "phi_open": phi_open,
        "phi_total": phi_total,
        "phi_closed_raw": phi_closed_raw,
        "phi_open_raw": phi_open_raw,
        "phi_total_raw": phi_total_raw,
        "frac_open_excluded": frac_open_excluded,
        "voxel_um3": vox_um3,
        "voxel_um": voxel_um,
        "solid_mm3": n_solid * vox_um3 / 1e9,
        "closed_pore_mm3": n_closed * vox_um3 / 1e9,
        "open_pore_mm3": n_open * vox_um3 / 1e9,
        "open_wrap_mm3": n_wrap * vox_um3 / 1e9,
        "envelope_mm3": int(envelope.sum()) * vox_um3 / 1e9,
        "particle_mm3": int(particle.sum()) * vox_um3 / 1e9,
        "slice_phi": slice_phi,
        "solid": solid,
        "closed_pore": closed_pore,
        "open_pore": open_pore,
        "open_wrap": open_wrap,
        "open_all": open_all,
        "cupping": cupping,
        "envelope": envelope,
        "particle": particle,
        "env_pore": env_pore,
        "fov2d": fov2d,
        "gray": gray,
        "closed_eq_um": (6.0 * closed_sizes * vox_um3 / np.pi) ** (1.0 / 3.0)
        if closed_sizes.size
        else np.array([]),
        "open_eq_um": (6.0 * open_sizes * vox_um3 / np.pi) ** (1.0 / 3.0)
        if open_sizes.size
        else np.array([]),
    }


def masks_to_labels(
    solid: np.ndarray,
    closed_pore: np.ndarray,
    open_pore: np.ndarray,
    open_wrap: np.ndarray | None = None,
) -> np.ndarray:
    """Pack segmentation masks into a single uint8 label volume for editing."""
    labels = np.zeros(solid.shape, dtype=np.uint8)
    labels[solid] = L_SOLID
    if open_wrap is not None:
        labels[open_wrap] = L_WRAP
    labels[open_pore] = L_OPEN
    labels[closed_pore] = L_CLOSED
    return labels


def closed_pores_slicewise(solid: np.ndarray, min_pore_voxels: int = 8) -> np.ndarray:
    """Closed (vesicle) pores via per-slice 2D fill-holes — matches setup preview.

    3D fill_holes misses many vesicles that appear enclosed in XY but connect in Z;
    the interactive preview uses 2D fill-holes, so the edit session must too.
    """
    closed = np.zeros(solid.shape, dtype=bool)
    for z in range(solid.shape[0]):
        if not solid[z].any():
            continue
        filled = ndi.binary_fill_holes(solid[z])
        closed[z] = filled & ~solid[z]
    return remove_small(closed, min_pore_voxels)


def classify_slice_labels(
    ct_z: np.ndarray,
    thresh_closed: int,
    fov2d: np.ndarray,
    thresh_open: int | None = None,
    voxel_um: float = 1.0,
    ball_um: float = 15.0,
    exclude_edge_um: float | None = None,
    crack_delta: float = 0.0,
    crack_um: float = 12.0,
    crack_elongation: float = 2.5,
    min_pore_voxels: int = 8,
    show_open: bool = True,
) -> np.ndarray:
    """Single-slice classification shared by setup preview (same rules as analyze)."""
    t_open = int(thresh_closed if thresh_open is None else thresh_open)
    if exclude_edge_um is None:
        exclude_edge_um = ball_um
    ball_px = max(1, int(round(ball_um / max(voxel_um, 1e-6))))

    solid = (ct_z > thresh_closed) & fov2d
    envelope = ndi.binary_closing(solid, structure=disk(ball_px))
    envelope = ndi.binary_fill_holes(envelope) & fov2d

    closed = remove_small(ndi.binary_fill_holes(solid) & ~solid, min_pore_voxels)

    if crack_delta > 0 and solid.any():
        size = int(round(crack_um / max(voxel_um, 1e-6)))
        size = max(3, size | 1)
        local = ndi.uniform_filter(ct_z.astype(np.float32), size=size)
        raw = ((local - ct_z) >= crack_delta) & solid & envelope
        lab, n = ndi.label(raw)
        if n:
            keep = np.zeros(n + 1, dtype=bool)
            for p in regionprops(lab):
                if p.area < min_pore_voxels:
                    continue
                minor = max(p.minor_axis_length, 1e-3)
                if (p.major_axis_length / minor) >= crack_elongation:
                    keep[p.label] = True
            cracks = keep[lab]
            solid = solid & ~cracks
            closed = closed | cracks

    open_wrap = np.zeros(ct_z.shape, dtype=bool)
    open_pore = np.zeros(ct_z.shape, dtype=bool)
    if show_open:
        open_all = remove_small(
            envelope & (ct_z <= t_open) & ~closed, min_pore_voxels
        )
        if exclude_edge_um > 0:
            rim_px = max(1, int(round(exclude_edge_um / max(voxel_um, 1e-6))))
            dist = ndi.distance_transform_edt(envelope)
            rim = envelope & (dist <= rim_px)
            open_wrap = open_all & rim
            open_pore = open_all & ~rim
        else:
            open_pore = open_all

    solid = (solid | (envelope & ~closed & (ct_z > t_open))) & ~open_pore & ~open_wrap
    return masks_to_labels(solid, closed, open_pore, open_wrap)


def preview_labels_for_slice(
    ct_z: np.ndarray,
    thresh_closed: int,
    fov2d: np.ndarray,
    thresh_open: int | None = None,
    voxel_um: float = 1.0,
    ball_um: float = 15.0,
    exclude_edge_um: float | None = None,
    crack_delta: float = 0.0,
    crack_um: float = 12.0,
    crack_elongation: float = 2.5,
    min_pore_voxels: int = 8,
    show_open: bool = True,
) -> np.ndarray:
    """Setup live preview — delegates to classify_slice_labels (same as 3D closed logic)."""
    return classify_slice_labels(
        ct_z,
        thresh_closed=thresh_closed,
        fov2d=fov2d,
        thresh_open=thresh_open,
        voxel_um=voxel_um,
        ball_um=ball_um,
        exclude_edge_um=exclude_edge_um,
        crack_delta=crack_delta,
        crack_um=crack_um,
        crack_elongation=crack_elongation,
        min_pore_voxels=min_pore_voxels,
        show_open=show_open,
    )


def labels_to_masks(labels: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "solid": labels == L_SOLID,
        "closed_pore": labels == L_CLOSED,
        "open_pore": labels == L_OPEN,
        "open_wrap": labels == L_WRAP,
    }


def result_from_labels(
    vol: np.ndarray,
    labels: np.ndarray,
    meta: dict,
) -> dict:
    """Build an analyze_volume-compatible result from edited labels (no re-segmentation)."""
    masks = labels_to_masks(labels)
    solid = masks["solid"]
    closed_pore = masks["closed_pore"]
    open_pore = masks["open_pore"]
    open_wrap = masks["open_wrap"]
    env_pore = closed_pore | open_pore
    particle = solid | closed_pore | open_pore
    envelope = particle | open_wrap
    if not envelope.any():
        envelope = particle
    open_all = open_pore | open_wrap

    voxel_um = float(meta["voxel_um"])
    vox_um3 = voxel_um ** 3
    n_solid = int(solid.sum())
    n_closed = int(closed_pore.sum())
    n_open = int(open_pore.sum())
    n_wrap = int(open_wrap.sum())
    n_open_all = int(open_all.sum())
    n_den = n_solid + n_closed + n_open
    n_den_raw = n_solid + n_closed + n_open_all

    phi_closed = n_closed / n_den if n_den else float("nan")
    phi_open = n_open / n_den if n_den else float("nan")
    phi_total = phi_closed + phi_open if n_den else float("nan")
    phi_closed_raw = n_closed / n_den_raw if n_den_raw else float("nan")
    phi_open_raw = n_open_all / n_den_raw if n_den_raw else float("nan")
    phi_total_raw = phi_closed_raw + phi_open_raw if n_den_raw else float("nan")
    frac_open_excluded = n_wrap / n_open_all if n_open_all else 0.0

    closed_lab, n_closed_cc = ndi.label(closed_pore)
    open_lab, n_open_cc = ndi.label(open_pore)
    closed_sizes = np.bincount(closed_lab.ravel())[1:] if n_closed_cc else np.array([])
    open_sizes = np.bincount(open_lab.ravel())[1:] if n_open_cc else np.array([])

    slice_phi = np.full(vol.shape[0], np.nan, dtype=np.float64)
    for z in range(vol.shape[0]):
        den = int(particle[z].sum())
        if den > 0:
            slice_phi[z] = float(env_pore[z].sum() / den)

    fov2d = fov_disk(vol.shape[1:])
    exterior_air = np.broadcast_to(fov2d, vol.shape) & ~envelope & (vol > 0)
    gray = {
        "exterior_air": _class_stats(_sample_vals(vol, exterior_air)),
        "closed_pore": _class_stats(_sample_vals(vol, closed_pore)),
        "open_pore": _class_stats(_sample_vals(vol, open_pore)),
        "open_wrap": _class_stats(_sample_vals(vol, open_wrap)),
        "solid": _class_stats(_sample_vals(vol, solid)),
    }

    return {
        "threshold": meta.get("threshold"),
        "otsu_threshold": meta.get("otsu_threshold"),
        "air_threshold": meta.get("air_threshold"),
        "air_k_std": meta.get("air_k_std"),
        "min_pore_voxels": meta.get("min_pore_voxels", 8),
        "ball_um": meta.get("ball_um"),
        "ball_px": meta.get("ball_px"),
        "exclude_edge_um": meta.get("exclude_edge_um"),
        "exclude_edge_px": meta.get("exclude_edge_px", 0),
        "cupping_um": meta.get("cupping_um"),
        "cupping_delta": meta.get("cupping_delta"),
        "n_cupping": meta.get("n_cupping", 0),
        "n_solid": n_solid,
        "n_closed_pore": n_closed,
        "n_open_pore": n_open,
        "n_open_wrap": n_wrap,
        "n_open_all": n_open_all,
        "n_envelope": int(envelope.sum()),
        "n_particle": int(particle.sum()),
        "n_closed_components": int(n_closed_cc),
        "n_open_components": int(n_open_cc),
        "phi_closed": phi_closed,
        "phi_open": phi_open,
        "phi_total": phi_total,
        "phi_closed_raw": phi_closed_raw,
        "phi_open_raw": phi_open_raw,
        "phi_total_raw": phi_total_raw,
        "frac_open_excluded": frac_open_excluded,
        "voxel_um3": vox_um3,
        "voxel_um": voxel_um,
        "solid_mm3": n_solid * vox_um3 / 1e9,
        "closed_pore_mm3": n_closed * vox_um3 / 1e9,
        "open_pore_mm3": n_open * vox_um3 / 1e9,
        "open_wrap_mm3": n_wrap * vox_um3 / 1e9,
        "envelope_mm3": int(envelope.sum()) * vox_um3 / 1e9,
        "particle_mm3": int(particle.sum()) * vox_um3 / 1e9,
        "slice_phi": slice_phi,
        "solid": solid,
        "closed_pore": closed_pore,
        "open_pore": open_pore,
        "open_wrap": open_wrap,
        "open_all": open_all,
        "cupping": np.zeros(solid.shape, dtype=bool),
        "envelope": envelope,
        "particle": particle,
        "env_pore": env_pore,
        "fov2d": fov2d,
        "gray": gray,
        "closed_eq_um": (6.0 * closed_sizes * vox_um3 / np.pi) ** (1.0 / 3.0)
        if closed_sizes.size
        else np.array([]),
        "open_eq_um": (6.0 * open_sizes * vox_um3 / np.pi) ** (1.0 / 3.0)
        if open_sizes.size
        else np.array([]),
        "edited": True,
    }


def session_dir(dataset: str, root: Path | None = None) -> Path:
    stem = dataset.replace(" ", "_")
    return (root or ROOT) / "_edit_sessions" / stem


def prepare_edit_session(
    folder: Path,
    downsample: int = 2,
    thresh: int | None = None,
    air_k_std: float = 2.5,
    min_pore_voxels: int = 8,
    ball_um: float = 15.0,
    exclude_edge_um: float | None = None,
    cupping_um: float = 40.0,
    cupping_delta: float = 1000.0,
    thresh_mix: float = 0.50,
    open_thresh: int | None = None,
    open_thresh_mix: float | None = None,
    crack_um: float = 12.0,
    crack_delta: float = 0.0,
    crack_elongation: float = 2.5,
    sessions_root: Path | None = None,
) -> Path:
    """Auto-segment, then write volume + labels for manual editing."""
    name = folder.name
    print(f"\n=== prepare-edit {name} ===")
    header = parse_header(folder)
    files = list_tiffs(folder)
    if not files:
        raise FileNotFoundError(f"No TIFF slices in {folder}")
    pixel_um = float(header.get("Pixel Size", 1.0))
    vol = load_volume(files, downsample)
    result = analyze_volume(
        vol,
        pixel_um,
        downsample,
        thresh,
        air_k_std=air_k_std,
        min_pore_voxels=min_pore_voxels,
        ball_um=ball_um,
        exclude_edge_um=exclude_edge_um,
        cupping_um=cupping_um,
        cupping_delta=cupping_delta,
        thresh_mix=thresh_mix,
        open_thresh=open_thresh,
        open_thresh_mix=open_thresh_mix,
        crack_um=crack_um,
        crack_delta=crack_delta,
        crack_elongation=crack_elongation,
    )
    labels = masks_to_labels(
        result["solid"],
        result["closed_pore"],
        result["open_pore"],
        result["open_wrap"],
    )
    out = session_dir(name, sessions_root)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "volume.npy", vol)
    np.save(out / "labels.npy", labels)
    np.save(out / "labels_auto.npy", labels.copy())
    meta = {
        "dataset": name,
        "source_folder": path_relative_to_root(folder),
        "project_root_hint": ROOT.name,
        "downsample": downsample,
        "pixel_um": pixel_um,
        "voxel_um": result["voxel_um"],
        "shape_zyx": list(vol.shape),
        "threshold": result["threshold"],
        "open_threshold": result.get("open_threshold"),
        "otsu_threshold": result["otsu_threshold"],
        "air_threshold": result["air_threshold"],
        "air_k_std": result["air_k_std"],
        "thresh_mix": result.get("thresh_mix"),
        "open_thresh_mix": result.get("open_thresh_mix"),
        "min_pore_voxels": result["min_pore_voxels"],
        "ball_um": result["ball_um"],
        "ball_px": result["ball_px"],
        "exclude_edge_um": result["exclude_edge_um"],
        "exclude_edge_px": result["exclude_edge_px"],
        "cupping_um": result["cupping_um"],
        "cupping_delta": result["cupping_delta"],
        "n_cupping": result["n_cupping"],
        "crack_um": result.get("crack_um"),
        "crack_delta": result.get("crack_delta"),
        "crack_elongation": result.get("crack_elongation"),
        "n_cracks": result.get("n_cracks"),
        "phi_closed_auto_pct": 100 * result["phi_closed"],
        "phi_open_auto_pct": 100 * result["phi_open"],
        "phi_total_auto_pct": 100 * result["phi_total"],
        "label_codes": LABEL_NAMES,
        "confirmed": False,
        "created_unix": time.time(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "README_edit.txt").write_text(
        "\n".join(
            [
                f"Edit session for {name}",
                "",
                "1) Open the editor:",
                f'   python porosity_edit.py --session "_edit_sessions/{out.name}"',
                "",
                "2) Paint labels on slices (mouse wheel = Z, left-drag = paint):",
                "   blue/yellow shadows = closed/open pores",
                "   1 solid | 2 closed | 3 open | 4 wrap/ignore | 0 background",
                "",
                "3) Save anytime (Ctrl+S). Quit and reopen the same session to continue.",
                "4) When finished, Confirm & compute, or:",
                f'   python porosity_ct.py --finalize --session "_edit_sessions/{out.name}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"  session -> {out}")
    print(
        f"  auto porosity  closed={meta['phi_closed_auto_pct']:.2f}%  "
        f"open={meta['phi_open_auto_pct']:.2f}%  total={meta['phi_total_auto_pct']:.2f}%"
    )
    print(f'  next: python porosity_edit.py --session "_edit_sessions/{out.name}"')
    return out


def finalize_edit_session(
    session: Path,
    out_dir: Path | None = None,
    do_sensitivity: bool = False,
) -> dict:
    """Load human-edited labels, compute porosity, write QC figures."""
    session = Path(session)
    vol = np.load(session / "volume.npy")
    labels = np.load(session / "labels.npy")
    meta = json.loads((session / "meta.json").read_text(encoding="utf-8"))
    name = meta["dataset"]
    print(f"\n=== finalize {name} (edited labels) ===")
    result = result_from_labels(vol, labels, meta)
    out_dir = Path(out_dir) if out_dir else (ROOT / "_porosity_results")
    out_dir.mkdir(exist_ok=True)
    save_qc_evidence(out_dir, name, vol, result, do_sensitivity=do_sensitivity)

    row = {
        "dataset": name,
        "edited": True,
        "session": path_relative_to_root(session),
        "shape_zyx": f"{vol.shape[0]}x{vol.shape[1]}x{vol.shape[2]}",
        "pixel_um": meta.get("pixel_um"),
        "voxel_um_after_ds": result["voxel_um"],
        "downsample": meta.get("downsample"),
        "threshold": result["threshold"],
        "ball_um": result["ball_um"],
        "exclude_edge_um": result["exclude_edge_um"],
        "phi_closed": result["phi_closed"],
        "phi_closed_pct": 100 * result["phi_closed"],
        "phi_open": result["phi_open"],
        "phi_open_pct": 100 * result["phi_open"],
        "phi_total": result["phi_total"],
        "phi_total_pct": 100 * result["phi_total"],
        "phi_closed_auto_pct": meta.get("phi_closed_auto_pct"),
        "phi_open_auto_pct": meta.get("phi_open_auto_pct"),
        "phi_total_auto_pct": meta.get("phi_total_auto_pct"),
        "n_closed_components": result["n_closed_components"],
        "n_open_components": result["n_open_components"],
        "solid_mm3": result["solid_mm3"],
        "closed_pore_mm3": result["closed_pore_mm3"],
        "open_pore_mm3": result["open_pore_mm3"],
        "open_wrap_mm3": result["open_wrap_mm3"],
        "particle_mm3": result["particle_mm3"],
        "gray_air_mean": result["gray"]["exterior_air"]["mean"],
        "gray_closed_mean": result["gray"]["closed_pore"]["mean"],
        "gray_open_mean": result["gray"]["open_pore"]["mean"],
        "gray_solid_mean": result["gray"]["solid"]["mean"],
        "qc_dir": path_relative_to_root(out_dir / f"{name.replace(' ', '_')}_qc"),
    }
    # Keep meta portable if it still has an absolute source_folder from another PC
    src = resolve_project_path(meta.get("source_folder"), fallback_name=name)
    if src is not None:
        meta["source_folder"] = path_relative_to_root(src)
    meta["confirmed"] = True
    meta["confirmed_unix"] = time.time()
    meta["phi_closed_edited_pct"] = row["phi_closed_pct"]
    meta["phi_open_edited_pct"] = row["phi_open_pct"]
    meta["phi_total_edited_pct"] = row["phi_total_pct"]
    (session / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (session / "porosity_edited.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

    summary_path = out_dir / "porosity_summary.json"
    rows: list[dict] = []
    if summary_path.exists():
        try:
            rows = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            rows = []
    rows = [r for r in rows if r.get("dataset") != name]
    rows.append(row)
    write_csv(out_dir / "porosity_summary.csv", rows)
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(
        f"  edited  closed={row['phi_closed_pct']:.2f}%  "
        f"open={row['phi_open_pct']:.2f}%  total={row['phi_total_pct']:.2f}%"
    )
    print(f"  QC -> {row['qc_dir']}")
    return row


def _stretch(img: np.ndarray) -> tuple[np.ndarray, float, float]:
    nz = img[img > 0]
    lo, hi = (np.percentile(nz, [1, 99]) if nz.size else (0.0, 1.0))
    vis = np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1.0), 0, 1)
    return vis, float(lo), float(hi)


def _overlay(vis: np.ndarray, closed: np.ndarray, opened: np.ndarray) -> np.ndarray:
    rgb = np.stack([vis, vis, vis], axis=-1)
    rgb[opened] = rgb[opened] * 0.30 + OPEN_RGB * 0.70
    rgb[closed] = rgb[closed] * 0.25 + CLOSED_RGB * 0.75
    return np.clip(rgb, 0, 1)


def _overlay_rim(
    vis: np.ndarray,
    closed: np.ndarray,
    interior_open: np.ndarray,
    rim_open: np.ndarray,
) -> np.ndarray:
    rgb = np.stack([vis, vis, vis], axis=-1)
    rgb[interior_open] = rgb[interior_open] * 0.30 + OPEN_RGB * 0.70
    rgb[rim_open] = rgb[rim_open] * 0.22 + RIM_RGB * 0.78
    rgb[closed] = rgb[closed] * 0.25 + CLOSED_RGB * 0.75
    return np.clip(rgb, 0, 1)


def _add_scalebar(ax, voxel_um: float, width_px: int, height_px: int) -> None:
    target_um = 100.0 if voxel_um * 80 < 120 else 50.0
    bar_px = target_um / voxel_um
    if bar_px > width_px * 0.35:
        target_um = 20.0
        bar_px = target_um / voxel_um
    x0 = width_px * 0.06
    y0 = height_px * 0.92
    ax.add_patch(Rectangle((x0, y0), bar_px, max(3, height_px * 0.012), color="white", zorder=5))
    ax.text(
        x0 + bar_px / 2,
        y0 - 8,
        f"{target_um:.0f} um",
        color="white",
        ha="center",
        va="top",
        fontsize=8,
        zorder=5,
    )


def _qc_slices(solid: np.ndarray, closed_pore: np.ndarray, open_pore: np.ndarray) -> list[int]:
    """Vesicular slice, max-solid slice, particle end — not just first/last Z."""
    area = solid.sum(axis=(1, 2))
    pore_area = (closed_pore | open_pore).sum(axis=(1, 2))
    keep = np.flatnonzero(area > 0.05 * area.max() if area.max() else 0)
    if keep.size == 0:
        return [solid.shape[0] // 2]
    z_ves = int(keep[np.argmax(pore_area[keep])])
    z_mid = int(keep[np.argmax(area[keep])])
    z_end = int(keep[int(0.85 * (len(keep) - 1))])
    picks = []
    for z in (z_ves, z_mid, z_end):
        if z not in picks:
            picks.append(z)
    return picks


def _qc_slices_span(solid: np.ndarray, n: int = 8) -> list[int]:
    area = solid.sum(axis=(1, 2))
    keep = np.flatnonzero(area > 0.05 * (area.max() if area.max() else 0))
    if keep.size == 0:
        return [solid.shape[0] // 2]
    n = min(n, int(keep.size))
    idx = np.linspace(0, keep.size - 1, n).astype(int)
    return [int(z) for z in keep[idx]]


def _envelope_rim(envelope: np.ndarray, rim_px: int) -> np.ndarray:
    dist = ndi.distance_transform_edt(envelope)
    return envelope & (dist <= max(1, rim_px))


def _zoom_boxes(mask: np.ndarray, n: int = 3, half: int = 72, min_vox: int = 12):
    if not mask.any():
        return []
    lab, _nlab = ndi.label(mask)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    boxes = []
    h, w = mask.shape
    for k in np.argsort(sizes)[::-1]:
        if sizes[k] < min_vox or len(boxes) >= n:
            break
        ys, xs = np.where(lab == k)
        cy, cx = int(np.median(ys)), int(np.median(xs))
        y0, y1 = max(0, cy - half), min(h, cy + half)
        x0, x1 = max(0, cx - half), min(w, cx + half)
        boxes.append((y0, y1, x0, x1, int(sizes[k])))
    return boxes


def _draw_envelope_contour(ax, envelope_z: np.ndarray) -> None:
    if envelope_z.any():
        ax.contour(envelope_z.astype(float), levels=[0.5], colors=["#00E676"], linewidths=0.7)


def _safe_savefig(fig, path: Path, dpi: int) -> Path:
    """Windows often locks previously viewed QC PNGs; retry then use an alternate name."""
    path = Path(path)
    last_err = None
    names = [path] + [path.with_name(f"{path.stem}_v{i}{path.suffix}") for i in range(2, 6)]
    for dest in names:
        for _ in range(3):
            try:
                fig.savefig(dest, dpi=dpi)
                return dest
            except OSError as e:
                last_err = e
                time.sleep(0.4)
    raise OSError(f"Could not save figure {path}: {last_err}") from last_err


def _line_profile_coords(closed_z: np.ndarray, open_z: np.ndarray, h: int, w: int):
    pore = closed_z | open_z
    if not pore.any():
        return h // 2, 0, w - 1
    lab, n = ndi.label(pore)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    k = int(np.argmax(sizes))
    ys, xs = np.where(lab == k)
    y = int(np.median(ys))
    x0, x1 = int(xs.min()), int(xs.max())
    pad = max(10, (x1 - x0) // 4)
    return y, max(0, x0 - pad), min(w - 1, x1 + pad)


def _save_edge_overlays(
    qc_dir: Path,
    name: str,
    vol: np.ndarray,
    result: dict,
) -> dict:
    """Extra overlays: magenta = wrapping dropped from porosity."""
    voxel_um = result["voxel_um"]
    rim_um = float(result.get("exclude_edge_um", result["ball_um"]))
    rim_px = int(result.get("exclude_edge_px", max(1, round(rim_um / voxel_um))))
    rim_open = result["open_wrap"]
    interior_open = result["open_pore"]
    n_open_all = int(result.get("n_open_all", interior_open.sum() + rim_open.sum()))
    n_rim_open = int(result.get("n_open_wrap", rim_open.sum()))
    frac_open_rim = result.get(
        "frac_open_excluded",
        n_rim_open / n_open_all if n_open_all else 0.0,
    )
    print(
        f"  edge-band ({rim_um:.0f} um / {rim_px} vx): "
        f"excluded {100 * frac_open_rim:.1f}% of raw open voxels from porosity",
        flush=True,
    )

    zs = _qc_slices_span(result["solid"], n=8)
    for z in _qc_slices(
        result["solid"], result["closed_pore"], result["open_pore"] | result["open_wrap"]
    ):
        if z not in zs:
            zs.append(z)
    zs = sorted(zs)

    fig, axes = plt.subplots(len(zs), 3, figsize=(12.5, 2.9 * len(zs)), squeeze=False)
    fig.suptitle(
        f"{name}  magenta = EXCLUDED wrapping (not in porosity)  |  "
        f"orange = counted open  |  blue = closed\n"
        f"lime = envelope  |  dropped {100 * frac_open_rim:.1f}% of raw open voxels",
        fontsize=10,
    )
    col_titles = [
        "CT + envelope contour",
        "pores (counted only)",
        "magenta = dropped wrap",
    ]
    for i, z in enumerate(zs):
        vis, lo, hi = _stretch(vol[z])
        axes[i, 0].imshow(vol[z], cmap="gray", vmin=lo, vmax=hi)
        _draw_envelope_contour(axes[i, 0], result["envelope"][z])
        axes[i, 1].imshow(_overlay(vis, result["closed_pore"][z], result["open_pore"][z]))
        _draw_envelope_contour(axes[i, 1], result["envelope"][z])
        axes[i, 2].imshow(
            _overlay_rim(vis, result["closed_pore"][z], interior_open[z], rim_open[z])
        )
        _draw_envelope_contour(axes[i, 2], result["envelope"][z])
        axes[i, 0].set_ylabel(f"z={z}", fontsize=8)
        _add_scalebar(axes[i, 0], voxel_um, vol[z].shape[1], vol[z].shape[0])
        for j, ax in enumerate(axes[i]):
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(col_titles[j], fontsize=8)
    fig.tight_layout()
    more_path = qc_dir / "slices_overlay_more.png"
    more_path = _safe_savefig(fig, more_path, dpi=160)
    plt.close(fig)

    rim_by_z = [(int(z), int(rim_open[z].sum())) for z in zs]
    rim_by_z.sort(key=lambda t: t[1], reverse=True)
    zoom_rows = []
    for z, _n in rim_by_z[:3]:
        boxes = _zoom_boxes(rim_open[z], n=2, half=80)
        if not boxes:
            boxes = _zoom_boxes(result["open_pore"][z], n=1, half=80)
        for box in boxes:
            zoom_rows.append((z, box))
    if not zoom_rows:
        z = zs[len(zs) // 2]
        h, w = vol[z].shape
        zoom_rows = [(z, (max(0, h // 2 - 80), min(h, h // 2 + 80),
                          max(0, w // 2 - 80), min(w, w // 2 + 80), 0))]

    fig, axes = plt.subplots(len(zoom_rows), 3, figsize=(11.5, 3.3 * len(zoom_rows)), squeeze=False)
    fig.suptitle(
        f"Edge zooms: magenta is dropped from porosity (wrapping).\n"
        "Orange+blue are counted. A thin magenta halo outside the grain is not a vesicle.",
        fontsize=10,
    )
    for i, (z, (y0, y1, x0, x1, nvox)) in enumerate(zoom_rows):
        vis, lo, hi = _stretch(vol[z])
        sl = (slice(y0, y1), slice(x0, x1))
        axes[i, 0].imshow(vol[z][sl], cmap="gray", vmin=lo, vmax=hi)
        _draw_envelope_contour(axes[i, 0], result["envelope"][z][sl])
        axes[i, 1].imshow(
            _overlay(vis, result["closed_pore"][z], result["open_pore"][z])[sl]
        )
        axes[i, 2].imshow(
            _overlay_rim(vis, result["closed_pore"][z], interior_open[z], rim_open[z])[sl]
        )
        axes[i, 0].set_ylabel(f"z={z}\n{nvox} rim vx", fontsize=8)
        if i == 0:
            axes[i, 0].set_title("CT zoom + envelope", fontsize=9)
            axes[i, 1].set_title("counted pores", fontsize=9)
            axes[i, 2].set_title("magenta = excluded", fontsize=9)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    zoom_path = _safe_savefig(fig, qc_dir / "edge_zooms.png", dpi=180)
    plt.close(fig)

    if rim_open.any():
        lab, nlab = ndi.label(rim_open)
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        k = int(np.argmax(sizes))
        zs_p, ys_p, xs_p = np.where(lab == k)
        zc, yc, xc = int(np.median(zs_p)), int(np.median(ys_p)), int(np.median(xs_p))
    else:
        zc = zs[len(zs) // 2]
        yc, xc = vol.shape[1] // 2, vol.shape[2] // 2

    planes = [
        ("XY", vol[zc], result["closed_pore"][zc], interior_open[zc], rim_open[zc], result["envelope"][zc]),
        ("XZ", vol[:, yc, :], result["closed_pore"][:, yc, :], interior_open[:, yc, :], rim_open[:, yc, :], result["envelope"][:, yc, :]),
        ("YZ", vol[:, :, xc], result["closed_pore"][:, :, xc], interior_open[:, :, xc], rim_open[:, :, xc], result["envelope"][:, :, xc]),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 11.5))
    fig.suptitle(
        f"Orthogonal overlays through largest edge-band blob  (z={zc}, y={yc}, x={xc})",
        fontsize=11,
    )
    for i, (title, img, closed, iopen, ropen, env) in enumerate(planes):
        vis, lo, hi = _stretch(img)
        axes[i, 0].imshow(img, cmap="gray", vmin=lo, vmax=hi)
        _draw_envelope_contour(axes[i, 0], env)
        axes[i, 1].imshow(_overlay(vis, closed, iopen | ropen))
        axes[i, 2].imshow(_overlay_rim(vis, closed, iopen, ropen))
        axes[i, 0].set_ylabel(title, fontsize=10)
        if i == 0:
            axes[i, 0].set_title("CT + envelope")
            axes[i, 1].set_title("pores")
            axes[i, 2].set_title("edge split")
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    ortho_path = _safe_savefig(fig, qc_dir / "orthogonal_overlay.png", dpi=160)
    plt.close(fig)

    return {
        "files": {
            "slices_overlay_more": more_path.name,
            "edge_zooms": zoom_path.name,
            "orthogonal_overlay": ortho_path.name,
        },
        "rim_um": rim_um,
        "rim_px": rim_px,
        "frac_open_in_edge_band": frac_open_rim,
        "n_open_edge_band": n_rim_open,
    }


def save_qc_evidence(
    out_dir: Path,
    name: str,
    vol: np.ndarray,
    result: dict,
    do_sensitivity: bool = True,
) -> dict:
    stem = name.replace(" ", "_")
    qc_dir = out_dir / f"{stem}_qc"
    qc_dir.mkdir(exist_ok=True)
    voxel_um = result["voxel_um"]
    zs = _qc_slices(
        result["solid"],
        result["closed_pore"],
        result["open_pore"] | result["open_wrap"],
    )

    fig, axes = plt.subplots(len(zs), 4, figsize=(14, 3.4 * len(zs)), squeeze=False)
    fig.suptitle(
        f"{name}  T={result['threshold']}  ball={result['ball_um']:.0f} um  "
        f"edge-wrap excluded  |  "
        f"closed={100 * result['phi_closed']:.2f}%  "
        f"open={100 * result['phi_open']:.2f}%  "
        f"total={100 * result['phi_total']:.2f}%",
        fontsize=11,
    )
    col_titles = ["CT slice", "solid", "3D envelope", "counted pores (wrap excluded)"]
    for i, z in enumerate(zs):
        img = vol[z]
        vis, lo, hi = _stretch(img)
        closed = result["closed_pore"][z]
        opened = result["open_pore"][z]
        axes[i, 0].imshow(img, cmap="gray", vmin=lo, vmax=hi)
        axes[i, 1].imshow(result["solid"][z], cmap="gray")
        env_rgb = np.stack([vis, vis, vis], axis=-1)
        env = result["envelope"][z]
        env_rgb[env & ~result["solid"][z]] = env_rgb[env & ~result["solid"][z]] * 0.4 + ENVELOPE_RGB * 0.6
        axes[i, 2].imshow(env_rgb)
        axes[i, 3].imshow(_overlay(vis, closed, opened))
        axes[i, 0].set_ylabel(f"z={z}", fontsize=9)
        _add_scalebar(axes[i, 0], voxel_um, img.shape[1], img.shape[0])
        for j, ax in enumerate(axes[i]):
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(col_titles[j], fontsize=9)
    fig.tight_layout()
    slice_path = _safe_savefig(fig, qc_dir / "slices_overlay.png", dpi=180)
    plt.close(fig)

    edge_info = _save_edge_overlays(qc_dir, name, vol, result)

    # Rolling-ball radius sensitivity on the two most informative slices
    ball_path = qc_dir / "ball_radius_sensitivity.png"
    if do_sensitivity:
        radii_um = [8.0, 15.0, 25.0]
        show_z = zs[:2]
        fig, axes = plt.subplots(len(show_z), len(radii_um), figsize=(12, 3.6 * len(show_z)), squeeze=False)
        fig.suptitle("QC: rolling-ball envelope radius (green = envelope minus solid)", fontsize=11)
        for j, r_um in enumerate(radii_um):
            if abs(r_um - result["ball_um"]) < 0.1:
                env_r = result["envelope"]
            else:
                env_r = rolling_ball_envelope(result["solid"], r_um / voxel_um)
            for i, z in enumerate(show_z):
                vis, lo, hi = _stretch(vol[z])
                rgb = np.stack([vis, vis, vis], axis=-1)
                extra = env_r[z] & ~result["solid"][z]
                rgb[extra] = rgb[extra] * 0.35 + ENVELOPE_RGB * 0.65
                axes[i, j].imshow(rgb)
                axes[i, j].axis("off")
                if i == 0:
                    axes[i, j].set_title(f"R={r_um:.0f} um")
                if j == 0:
                    axes[i, j].set_ylabel(f"z={z}")
        fig.tight_layout()
        _safe_savefig(fig, ball_path, dpi=160)
        plt.close(fig)

    z = zs[0]
    vis, lo, hi = _stretch(vol[z])
    pore = result["closed_pore"][z] | result["open_pore"][z]
    if pore.any():
        ys, xs = np.where(pore)
        cy, cx = int(np.median(ys)), int(np.median(xs))
    else:
        cy, cx = vol[z].shape[0] // 2, vol[z].shape[1] // 2
    rad = 90
    y0, y1 = max(0, cy - rad), min(vol[z].shape[0], cy + rad)
    x0, x1 = max(0, cx - rad), min(vol[z].shape[1], cx + rad)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    fig.suptitle(f"QC zoom z={z}", fontsize=11)
    axes[0].imshow(vol[z][y0:y1, x0:x1], cmap="gray", vmin=lo, vmax=hi)
    axes[0].set_title("CT zoom")
    axes[1].imshow(result["envelope"][z][y0:y1, x0:x1], cmap="gray")
    axes[1].set_title("envelope")
    axes[2].imshow(_overlay(vis, result["closed_pore"][z], result["open_pore"][z])[y0:y1, x0:x1])
    axes[2].set_title("pores")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    zoom_path = _safe_savefig(fig, qc_dir / "zoom_envelope.png", dpi=180)
    plt.close(fig)

    # Cupping recovery QC: show recovered cores in yellow on a vesicular slice
    cup = result.get("cupping")
    cupping_path = qc_dir / "cupping_recovery.png"
    if cup is not None and np.any(cup):
        zc = zs[0]
        for zz in zs:
            if int(cup[zz].sum()) > int(cup[zc].sum()):
                zc = zz
        vis, lo, hi = _stretch(vol[zc])
        rgb = _overlay(vis, result["closed_pore"][zc], result["open_pore"][zc])
        rgb_y = rgb.copy()
        rgb_y[cup[zc]] = np.array([1.0, 1.0, 0.0])
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
        fig.suptitle(
            f"Cupping recovery z={zc}: yellow was solid (above air T) but darker than local walls\n"
            f"recovered {result.get('n_cupping', int(cup.sum()))} voxels  |  "
            f"window={result.get('cupping_um', 0):.0f} um  delta={result.get('cupping_delta', 0):.0f}",
            fontsize=10,
        )
        axes[0].imshow(vol[zc], cmap="gray", vmin=lo, vmax=hi)
        axes[0].set_title("CT")
        axes[1].imshow(rgb_y)
        axes[1].set_title("yellow = recovered cores")
        axes[2].imshow(rgb)
        axes[2].set_title("final pores")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        cupping_path = _safe_savefig(fig, cupping_path, dpi=170)
        plt.close(fig)

    gray = result["gray"]
    samples = {
        "exterior_air": _sample_vals(
            vol,
            np.broadcast_to(result["fov2d"], vol.shape) & ~result["envelope"] & (vol > 0),
            250_000,
        ),
        "closed_pore": _sample_vals(vol, result["closed_pore"], 250_000),
        "open_pore": _sample_vals(vol, result["open_pore"], 250_000),
        "solid": _sample_vals(vol, result["solid"], 250_000),
    }
    colors = {
        "exterior_air": "#7f7f7f",
        "closed_pore": "#0072B2",
        "open_pore": "#E69F00",
        "solid": "#222222",
    }
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    bins = np.linspace(0, max(int(vol.max()), 1), 256)
    for key, color in colors.items():
        v = samples[key]
        if v.size == 0:
            continue
        ax.hist(
            v,
            bins=bins,
            density=True,
            histtype="step",
            lw=1.6,
            color=color,
            label=key.replace("_", " "),
        )
    ax.axvline(result["threshold"], color="crimson", lw=1.4, label=f"T={result['threshold']}")
    ax.set_xlabel("gray value")
    ax.set_ylabel("density")
    ax.set_title("QC: class gray-value overlap (pores should match exterior air)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    hist_path = _safe_savefig(fig, qc_dir / "class_histogram.png", dpi=170)
    plt.close(fig)

    z = zs[len(zs) // 2]
    h, w = vol[z].shape
    y, x0, x1 = _line_profile_coords(result["closed_pore"][z], result["open_pore"][z], h, w)
    xs = np.arange(x0, x1 + 1)
    profile = vol[z, y, x0 : x1 + 1]
    vis, lo, hi = _stretch(vol[z])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.05, 1.2]})
    axes[0].imshow(vol[z], cmap="gray", vmin=lo, vmax=hi)
    axes[0].imshow(_overlay(vis, result["closed_pore"][z], result["open_pore"][z]), alpha=0.55)
    axes[0].plot([x0, x1], [y, y], color="lime", lw=1.4)
    axes[0].set_title(f"line through largest pore  z={z}")
    axes[0].axis("off")
    axes[1].plot(xs * voxel_um, profile, color="k", lw=1.0)
    axes[1].axhline(result["threshold"], color="crimson", lw=1.2, label=f"T={result['threshold']}")
    axes[1].set_xlabel("distance (um)")
    axes[1].set_ylabel("gray value")
    axes[1].set_title("QC: gray values drop to air inside pores")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    profile_path = _safe_savefig(fig, qc_dir / "line_profile.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    if result["closed_eq_um"].size:
        ax.hist(result["closed_eq_um"], bins=30, color="#0072B2", alpha=0.85, label="closed")
    if result["open_eq_um"].size:
        ax.hist(result["open_eq_um"], bins=30, color="#E69F00", alpha=0.65, label="open")
    ax.set_xlabel("equivalent sphere diameter (um)")
    ax.set_ylabel("count")
    ax.set_title("QC: pore size distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    psd_path = _safe_savefig(fig, qc_dir / "pore_size.png", dpi=160)
    plt.close(fig)

    air_mean = gray["exterior_air"]["mean"]
    closed_mean = gray["closed_pore"]["mean"]
    open_mean = gray["open_pore"]["mean"]
    solid_mean = gray["solid"]["mean"]
    sep = None if air_mean is None or solid_mean is None else (solid_mean - air_mean)
    evidence = {
        "dataset": name,
        "otsu_threshold": result["otsu_threshold"],
        "air_threshold": result["air_threshold"],
        "threshold": result["threshold"],
        "ball_um": result["ball_um"],
        "ball_px": result["ball_px"],
        "phi_closed_pct": 100 * result["phi_closed"],
        "phi_open_pct": 100 * result["phi_open"],
        "phi_total_pct": 100 * result["phi_total"],
        "phi_closed_raw_pct": 100 * result["phi_closed_raw"],
        "phi_open_raw_pct": 100 * result["phi_open_raw"],
        "phi_total_raw_pct": 100 * result["phi_total_raw"],
        "exclude_edge_um": result["exclude_edge_um"],
        "n_cupping": result.get("n_cupping", 0),
        "cupping_um": result.get("cupping_um"),
        "cupping_delta": result.get("cupping_delta"),
        "n_closed_components": result["n_closed_components"],
        "n_open_components": result["n_open_components"],
        "edge_band": {
            "rim_um": edge_info["rim_um"],
            "rim_px": edge_info["rim_px"],
            "frac_open_in_edge_band": edge_info["frac_open_in_edge_band"],
            "n_open_edge_band": edge_info["n_open_edge_band"],
        },
        "gray_by_class": gray,
        "qc_checks": {
            "closed_pore_near_exterior_air": (
                None
                if sep is None or closed_mean is None
                else abs(closed_mean - air_mean) < 0.25 * sep
            ),
            "open_pore_near_exterior_air": (
                None
                if sep is None or open_mean is None
                else abs(open_mean - air_mean) < 0.25 * sep
            ),
            "solid_well_above_threshold": (
                None if solid_mean is None else solid_mean > result["threshold"]
            ),
            "air_well_below_threshold": (
                None if air_mean is None else air_mean < result["threshold"]
            ),
        },
        "files": {
            "slices_overlay": slice_path.name,
            "slices_overlay_more": edge_info["files"]["slices_overlay_more"],
            "edge_zooms": edge_info["files"]["edge_zooms"],
            "orthogonal_overlay": edge_info["files"]["orthogonal_overlay"],
            "ball_radius_sensitivity": ball_path.name,
            "zoom_envelope": zoom_path.name,
            "cupping_recovery": cupping_path.name,
            "class_histogram": hist_path.name,
            "line_profile": profile_path.name,
            "pore_size": psd_path.name,
        },
    }
    (qc_dir / "qc_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(f"  QC evidence -> {qc_dir}")
    return evidence


def run_one(
    folder: Path,
    out_dir: Path,
    downsample: int,
    thresh: int | None,
    air_k_std: float = 2.5,
    min_pore_voxels: int = 8,
    ball_um: float = 15.0,
    exclude_edge_um: float | None = None,
    cupping_um: float = 40.0,
    cupping_delta: float = 1000.0,
    thresh_mix: float = 0.50,
    open_thresh: int | None = None,
    open_thresh_mix: float | None = None,
    do_sensitivity: bool = True,
) -> dict:
    name = folder.name
    print(f"\n=== {name} ===")
    t0 = time.time()
    header = parse_header(folder)
    files = list_tiffs(folder)
    if not files:
        raise FileNotFoundError(f"No TIFF slices in {folder}")
    pixel_um = float(header.get("Pixel Size", 1.0))
    print(f"  slices={len(files)}  pixel={pixel_um} um  ds={downsample}")
    vol = load_volume(files, downsample)
    result = analyze_volume(
        vol,
        pixel_um,
        downsample,
        thresh,
        air_k_std=air_k_std,
        min_pore_voxels=min_pore_voxels,
        ball_um=ball_um,
        exclude_edge_um=exclude_edge_um,
        cupping_um=cupping_um,
        cupping_delta=cupping_delta,
        thresh_mix=thresh_mix,
        open_thresh=open_thresh,
        open_thresh_mix=open_thresh_mix,
    )
    save_qc_evidence(out_dir, name, vol, result, do_sensitivity=do_sensitivity)

    row = {
        "dataset": name,
        "n_slices_raw": len(files),
        "n_slices_used": int(vol.shape[0]),
        "shape_zyx": f"{vol.shape[0]}x{vol.shape[1]}x{vol.shape[2]}",
        "pixel_um": pixel_um,
        "voxel_um_after_ds": pixel_um * downsample,
        "downsample": downsample,
        "otsu_threshold": result["otsu_threshold"],
        "air_threshold": result["air_threshold"],
        "threshold": result["threshold"],
        "open_threshold": result.get("open_threshold"),
        "air_k_std": result["air_k_std"],
        "thresh_mix": result.get("thresh_mix"),
        "open_thresh_mix": result.get("open_thresh_mix"),
        "min_pore_voxels": result["min_pore_voxels"],
        "ball_um": result["ball_um"],
        "ball_px": result["ball_px"],
        "exclude_edge_um": result["exclude_edge_um"],
        "frac_open_excluded": result["frac_open_excluded"],
        "cupping_um": result["cupping_um"],
        "cupping_delta": result["cupping_delta"],
        "n_cupping": result["n_cupping"],
        "phi_closed": result["phi_closed"],
        "phi_closed_pct": 100 * result["phi_closed"],
        "phi_open": result["phi_open"],
        "phi_open_pct": 100 * result["phi_open"],
        "phi_total": result["phi_total"],
        "phi_total_pct": 100 * result["phi_total"],
        "phi_closed_raw_pct": 100 * result["phi_closed_raw"],
        "phi_open_raw_pct": 100 * result["phi_open_raw"],
        "phi_total_raw_pct": 100 * result["phi_total_raw"],
        "n_closed_components": result["n_closed_components"],
        "n_open_components": result["n_open_components"],
        "solid_mm3": result["solid_mm3"],
        "closed_pore_mm3": result["closed_pore_mm3"],
        "open_pore_mm3": result["open_pore_mm3"],
        "open_wrap_mm3": result["open_wrap_mm3"],
        "envelope_mm3": result["envelope_mm3"],
        "particle_mm3": result["particle_mm3"],
        "gray_air_mean": result["gray"]["exterior_air"]["mean"],
        "gray_closed_mean": result["gray"]["closed_pore"]["mean"],
        "gray_open_mean": result["gray"]["open_pore"]["mean"],
        "gray_solid_mean": result["gray"]["solid"]["mean"],
        "seconds": round(time.time() - t0, 1),
        "qc_dir": path_relative_to_root(Path(out_dir) / f"{name.replace(' ', '_')}_qc"),
    }
    print(
        f"  closed={row['phi_closed_pct']:.2f}%  "
        f"open={row['phi_open_pct']:.2f}%  "
        f"total={row['phi_total_pct']:.2f}%  "
        f"({row['seconds']} s)"
    )
    print(
        f"  gray means  air={row['gray_air_mean']:.0f}  "
        f"closed={row['gray_closed_mean']:.0f}  "
        f"open={row['gray_open_mean']:.0f}  "
        f"solid={row['gray_solid_mean']:.0f}"
    )
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CT porosity of lunar-soil particles.")
    parser.add_argument("--dataset", default="", help="Folder name, or omit with --all")
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--prepare-edit",
        action="store_true",
        help="Auto-segment and write an editable session under _edit_sessions/.",
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Compute porosity + QC from a human-edited session.",
    )
    parser.add_argument(
        "--session",
        default="",
        help="Path to _edit_sessions/<name> (for --finalize, or optional with --prepare-edit).",
    )
    parser.add_argument("--downsample", type=int, default=2, help="Integer >=1. 1=full resolution.")
    parser.add_argument(
        "--ball-um",
        type=float,
        default=15.0,
        help="Rolling-ball envelope radius in micrometers (3D morphological closing).",
    )
    parser.add_argument("--threshold", type=int, default=None, help="Manual gray threshold.")
    parser.add_argument(
        "--air-k-std",
        type=float,
        default=2.5,
        help="Air-calibrated threshold = air_mean + k*air_std. Lower k keeps more dark solid.",
    )
    parser.add_argument(
        "--min-pore-voxels",
        type=int,
        default=8,
        help="Drop pore components smaller than this (noise filter).",
    )
    parser.add_argument(
        "--exclude-edge-um",
        type=float,
        default=None,
        help="Drop open voxels within this many um of the envelope surface "
        "(default: same as --ball-um). 0 keeps wrapping in open porosity.",
    )
    parser.add_argument(
        "--thresh-mix",
        type=float,
        default=0.50,
        help="Closed-pore auto threshold blend: air + mix*(Otsu-air). "
        "Higher = more complete bright vesicle cores (0=air only, 1=Otsu). Default 0.50.",
    )
    parser.add_argument(
        "--open-thresh-mix",
        type=float,
        default=None,
        help="Open-pore intensity blend (same scale as --thresh-mix). "
        "Default: same as --thresh-mix. Lower = less yellow; higher = more yellow.",
    )
    parser.add_argument(
        "--cupping-um",
        type=float,
        default=40.0,
        help="Local window (um) for cupping-bright vesicle recovery. 0 disables.",
    )
    parser.add_argument(
        "--cupping-delta",
        type=float,
        default=1000.0,
        help="Recover solid voxels darker than the local mean by this gray amount.",
    )
    parser.add_argument("--out", default="_porosity_results")
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="Skip extra rolling-ball radii (faster QC regeneration).",
    )
    args = parser.parse_args()
    out_dir = ROOT / args.out
    out_dir.mkdir(exist_ok=True)

    if args.finalize:
        sess = Path(args.session) if args.session else None
        if sess is not None and not sess.is_absolute():
            sess = ROOT / sess
        if sess is None or not sess.is_dir():
            raise SystemExit(
                "Provide --session path to an _edit_sessions folder "
                '(e.g. --session "_edit_sessions/6-11").'
            )
        finalize_edit_session(sess, out_dir=out_dir, do_sensitivity=not args.skip_sensitivity)
        return

    available = discover_datasets()
    if args.prepare_edit:
        names = available if args.all else [args.dataset] if args.dataset else []
        if not names:
            raise SystemExit(
                "Provide --dataset NAME or --all with --prepare-edit.\n"
                f"Available: {', '.join(available) or '(none found)'}"
            )
        for name in names:
            folder = ROOT / name
            if not folder.is_dir():
                print(f"skip missing {name}")
                continue
            prepare_edit_session(
                folder,
                downsample=max(1, args.downsample),
                thresh=args.threshold,
                air_k_std=args.air_k_std,
                min_pore_voxels=args.min_pore_voxels,
                ball_um=args.ball_um,
                exclude_edge_um=args.exclude_edge_um,
                cupping_um=args.cupping_um,
                cupping_delta=args.cupping_delta,
                thresh_mix=args.thresh_mix,
                open_thresh_mix=args.open_thresh_mix,
            )
        return

    names = available if args.all else [args.dataset] if args.dataset else available
    rows = []
    for name in names:
        folder = ROOT / name
        if not folder.is_dir():
            print(f"skip missing {name}")
            continue
        rows.append(            run_one(
                folder,
                out_dir,
                downsample=max(1, args.downsample),
                thresh=args.threshold,
                air_k_std=args.air_k_std,
                min_pore_voxels=args.min_pore_voxels,
                ball_um=args.ball_um,
                exclude_edge_um=args.exclude_edge_um,
                cupping_um=args.cupping_um,
                cupping_delta=args.cupping_delta,
                thresh_mix=args.thresh_mix,
                open_thresh_mix=args.open_thresh_mix,
                do_sensitivity=not args.skip_sensitivity,
            )
        )

    write_csv(out_dir / "porosity_summary.csv", rows)
    (out_dir / "porosity_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {out_dir / 'porosity_summary.csv'}")


if __name__ == "__main__":
    main()

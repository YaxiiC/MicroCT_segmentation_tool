"""One-off gray-value diagnostics for 6-11 threshold adjustment."""
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from porosity_ct import fov_disk, list_tiffs, load_volume, parse_header

root = Path(r"c:\Users\chris\Downloads\Lunar Soil-CT data")
folder = root / "6-11"
out = root / "_porosity_results" / "6-11_qc"
out.mkdir(parents=True, exist_ok=True)

header = parse_header(folder)
vol = load_volume(list_tiffs(folder), downsample=2)
fov = fov_disk(vol.shape[1:])
print("vol", vol.shape, "otsu", int(threshold_otsu(vol[:, fov][vol[:, fov] > 0])))

# Exterior air: FOV ring, skip the particle by taking low-solid slices? Use FOV AND far from bright.
bright = (vol > 26000) & fov
dist = ndi.distance_transform_edt(~bright)
air = fov & (vol > 0) & (dist > 30)
print("air mean/std/p95", air.sum(), vol[air].mean(), vol[air].std(), np.percentile(vol[air], [90, 95, 99]))

zs = [0, 80, 196, 250]
for z in zs:
    sl = vol[z]
    print(f"\n--- slice {z} ---")
    print("  min/max/mean", int(sl.min()), int(sl.max()), float(sl[sl > 0].mean()))

# Compare candidate thresholds on slice 0 and 196
cands = {
    "otsu_24789": 24789,
    "air_p99": int(np.percentile(vol[air], 99)),
    "air_mean+2std": int(vol[air].mean() + 2 * vol[air].std()),
    "air_mean+2.5std": int(vol[air].mean() + 2.5 * vol[air].std()),
    "air_mean+3std": int(vol[air].mean() + 3 * vol[air].std()),
}
print("\ncandidates", cands)

fig, axes = plt.subplots(2, len(cands) + 1, figsize=(3.2 * (len(cands) + 1), 7))
for row, z in enumerate([0, 196]):
    sl = vol[z]
    lo, hi = np.percentile(sl[sl > 0], [1, 99])
    axes[row, 0].imshow(sl, cmap="gray", vmin=lo, vmax=hi)
    axes[row, 0].set_title(f"CT z={z}")
    axes[row, 0].axis("off")
    for i, (name, t) in enumerate(cands.items(), start=1):
        solid = (sl > t) & fov
        # 2D fill to show recovered vesicles
        filled = ndi.binary_fill_holes(solid)
        pores = filled & ~solid
        rgb = np.stack([sl, sl, sl], axis=-1).astype(np.float32)
        rgb = (rgb - lo) / max(hi - lo, 1)
        rgb = np.clip(rgb, 0, 1)
        rgb[pores] = rgb[pores] * 0.25 + np.array([0.9, 0.62, 0.0]) * 0.75
        axes[row, i].imshow(rgb)
        axes[row, i].set_title(f"{name}\nT={t}", fontsize=8)
        axes[row, i].axis("off")
fig.tight_layout()
fig.savefig(out / "threshold_sweep.png", dpi=140)
plt.close()
print("wrote", out / "threshold_sweep.png")

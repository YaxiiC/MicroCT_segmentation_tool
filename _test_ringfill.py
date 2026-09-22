"""Test filling solid interiors enclosed by closed-pore rings (cupping cores)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

import porosity_ct as pc

ROOT = Path(__file__).resolve().parent
vol = pc.load_volume(pc.list_tiffs(ROOT / "6-11"), downsample=2)
pixel_um = float(pc.parse_header(ROOT / "6-11")["Pixel Size"])
result = pc.analyze_volume(vol, pixel_um, 2, None, ball_um=15.0, exclude_edge_um=15.0)

z = 158
sl = vol[z]
solid = result["solid"]
closed = result["closed_pore"]
opened = result["open_pore"]
wrap = result["open_wrap"]

# 2D ring fill
cup2 = np.zeros_like(solid)
for zi in range(solid.shape[0]):
    if closed[zi].any():
        cup2[zi] = ndi.binary_fill_holes(closed[zi]) & solid[zi]
# 3D ring fill
cup3 = ndi.binary_fill_holes(closed) & solid
cup = pc.remove_small(cup2 | cup3, 8)
print(f"cupping recovered: 2d={int(cup2.sum())} 3d={int(cup3.sum())} union={int(cup.sum())}")
print(f"on z={z}: {int(cup[z].sum())} voxels")

# Also try fill of closed|open (may overfill)
cup_open = np.zeros_like(solid)
for zi in range(solid.shape[0]):
    m = closed[zi] | opened[zi]
    if m.any():
        cup_open[zi] = ndi.binary_fill_holes(m) & solid[zi]
cup_open = pc.remove_small(cup_open, 8)
print(f"if also fill through open rims: {int(cup_open.sum())} (extra={int((cup_open & ~cup).sum())})")

vis, lo, hi = pc._stretch(sl)
rgb = pc._overlay_rim(vis, closed[z], opened[z], wrap[z])
rgb2 = rgb.copy()
rgb2[cup[z]] = [1, 1, 0]

# after reclassify preview: solid2, closed2
solid2 = solid & ~cup
filled = ndi.binary_fill_holes(solid2)
closed2 = filled & ~solid2
opened2 = result["envelope"] & ~solid2 & ~closed2 & ~wrap  # rough
# keep wrap as before approx
vis_rgb = pc._overlay(vis, closed2[z], (opened2[z] | (opened[z] & ~cup[z])))

fig, axes = plt.subplots(2, 2, figsize=(10, 9))
fig.suptitle(
    "Yellow = solid interiors enclosed by blue pore rims (cupping cores to recover)\n"
    "Bottom-right = pores after punching those cores out of solid"
)
axes[0, 0].imshow(sl, cmap="gray", vmin=lo, vmax=hi)
axes[0, 0].set_title("CT")
axes[0, 1].imshow(rgb)
axes[0, 1].set_title("current (thin blue rings)")
axes[1, 0].imshow(rgb2)
axes[1, 0].set_title("yellow = recovered cores")
axes[1, 1].imshow(vis_rgb)
axes[1, 1].set_title("after recovery preview")
for ax in axes.ravel():
    ax.axis("off")
# zoom red-circle region approx
cy, cx, r = 210, 340, 70
for ax in axes.ravel():
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy + r, cy - r)

out = ROOT / "_porosity_results" / "6-11_qc" / "cupping_ringfill_preview.png"
fig.tight_layout()
pc._safe_savefig(fig, out, dpi=170)
plt.close(fig)
print("wrote", out)

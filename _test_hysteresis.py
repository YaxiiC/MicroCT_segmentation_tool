"""Hysteresis: grow pores from air-dark seeds into below-Otsu solid (cupping)."""
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

air_t = result["threshold"]
otsu = result["otsu_threshold"]
env = result["envelope"]
solid = result["solid"]
closed = result["closed_pore"]
opened = result["open_pore"]
wrap = result["open_wrap"]

struct = ndi.generate_binary_structure(3, 1)
seed = (vol < air_t) & env & (vol > 0)
mask = (vol < otsu) & env & (vol > 0)
grown = ndi.binary_propagation(seed, structure=struct, mask=mask)
cupping = grown & solid
cupping = pc.remove_small(cupping, 8)
print(f"hysteresis cupping voxels: {int(cupping.sum())}")
print(f"  gray med={np.median(vol[cupping]) if cupping.any() else 0:.0f}")
print(f"  on z=158: {int(cupping[158].sum())}")

# porosity preview
solid2 = solid & ~cupping
filled = ndi.binary_fill_holes(solid2)
closed2 = pc.remove_small(filled & ~solid2, 8)
open_all = pc.remove_small(env & ~solid2 & ~closed2, 8)
rim = pc._envelope_rim(env, max(1, int(round(15.0 / result["voxel_um"]))))
open2 = open_all & ~rim
wrap2 = open_all & rim
n_s, n_c, n_o = int(solid2.sum()), int(closed2.sum()), int(open2.sum())
den = n_s + n_c + n_o
print(f"preview porosity closed={100*n_c/den:.2f}% open={100*n_o/den:.2f}% total={100*(n_c+n_o)/den:.2f}%")

z = 158
vis, lo, hi = pc._stretch(vol[z])
before = pc._overlay_rim(vis, closed[z], opened[z], wrap[z])
after = pc._overlay_rim(vis, closed2[z], open2[z], wrap2[z])
yellow = before.copy()
yellow[cupping[z]] = [1, 1, 0]

fig, axes = plt.subplots(2, 2, figsize=(10, 9))
fig.suptitle(
    f"Hysteresis cupping recovery (seed < air T={air_t}, grow while < Otsu={otsu})\n"
    "Yellow = newly recovered from solid"
)
axes[0, 0].imshow(vol[z], cmap="gray", vmin=lo, vmax=hi)
axes[0, 0].set_title("CT")
axes[0, 1].imshow(before)
axes[0, 1].set_title("before")
axes[1, 0].imshow(yellow)
axes[1, 0].set_title("yellow = recovered")
axes[1, 1].imshow(after)
axes[1, 1].set_title("after")
for ax in axes.ravel():
    ax.axis("off")
cy, cx, r = 200, 330, 90
for ax in axes.ravel():
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy + r, cy - r)
out = ROOT / "_porosity_results" / "6-11_qc" / "cupping_hysteresis_preview.png"
fig.tight_layout()
pc._safe_savefig(fig, out, dpi=170)
plt.close(fig)
print("wrote", out)

"""Local-depression cupping recovery (solid darker than neighborhood)."""
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

solid = result["solid"]
env = result["envelope"]
closed = result["closed_pore"]
opened = result["open_pore"]
wrap = result["open_wrap"]
voxel_um = result["voxel_um"]
z = 158

# Per-slice local mean (faster / more stable than 3D for vesicles)
def local_cupping(vol, solid, env, size_px, delta):
    cup = np.zeros(solid.shape, dtype=bool)
    for zi in range(vol.shape[0]):
        if not solid[zi].any():
            continue
        sl = vol[zi].astype(np.float32)
        local = ndi.uniform_filter(sl, size=size_px)
        cup[zi] = ((local - sl) >= delta) & solid[zi] & env[zi]
    return pc.remove_small(cup, 27)

# Require recovered blob to touch an existing pore (rim-connected cupping)
def touch_pores(cup, pores):
    lab, n = ndi.label(cup)
    if n == 0:
        return cup
    dil_p = ndi.binary_dilation(pores)
    keep = np.zeros(n + 1, dtype=bool)
    # any label that overlaps dilated pores
    hit = lab[dil_p & (lab > 0)]
    keep[np.unique(hit)] = True
    keep[0] = False
    return keep[lab]

pores0 = closed | opened | wrap
for size_um, delta in [(25, 1200), (30, 1500), (35, 1500), (40, 1800), (30, 2000)]:
    size_px = max(5, int(round(size_um / voxel_um)) | 1)  # odd
    cup = local_cupping(vol, solid, env, size_px, delta)
    cup_t = touch_pores(cup, pores0)
    solid2 = solid & ~cup_t
    filled = ndi.binary_fill_holes(solid2)
    closed2 = pc.remove_small(filled & ~solid2, 8)
    open_all = pc.remove_small(env & ~solid2 & ~closed2, 8)
    rim = pc._envelope_rim(env, max(1, int(round(15 / voxel_um))))
    open2 = open_all & ~rim
    n_s, n_c, n_o = int(solid2.sum()), int(closed2.sum()), int(open2.sum())
    den = n_s + n_c + n_o
    print(
        f"size={size_um}um({size_px}px) d={delta}: cup={int(cup.sum())} "
        f"touch={int(cup_t.sum())} z158={int(cup_t[z].sum())} "
        f"phi c/o/t={100*n_c/den:.2f}/{100*n_o/den:.2f}/{100*(n_c+n_o)/den:.2f}"
    )

# Save best candidate preview: 30um / 1500
size_px = max(5, int(round(30 / voxel_um)) | 1)
cup = touch_pores(local_cupping(vol, solid, env, size_px, 1500), pores0)
solid2 = solid & ~cup
filled = ndi.binary_fill_holes(solid2)
closed2 = pc.remove_small(filled & ~solid2, 8)
open_all = pc.remove_small(env & ~solid2 & ~closed2, 8)
rim = pc._envelope_rim(env, max(1, int(round(15 / voxel_um))))
open2 = open_all & ~rim
wrap2 = open_all & rim

vis, lo, hi = pc._stretch(vol[z])
before = pc._overlay_rim(vis, closed[z], opened[z], wrap[z])
after = pc._overlay_rim(vis, closed2[z], open2[z], wrap2[z])
yellow = before.copy()
yellow[cup[z]] = [1, 1, 0]
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
fig.suptitle("Local-depression cupping (30 um window, delta=1500, must touch existing pore)")
axes[0, 0].imshow(vol[z], cmap="gray", vmin=lo, vmax=hi); axes[0, 0].set_title("CT")
axes[0, 1].imshow(before); axes[0, 1].set_title("before")
axes[1, 0].imshow(yellow); axes[1, 0].set_title("yellow = recovered")
axes[1, 1].imshow(after); axes[1, 1].set_title("after")
for ax in axes.ravel():
    ax.axis("off")
cy, cx, r = 200, 330, 90
for ax in axes.ravel():
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy + r, cy - r)
out = ROOT / "_porosity_results" / "6-11_qc" / "cupping_local_preview.png"
fig.tight_layout()
pc._safe_savefig(fig, out, dpi=170)
plt.close(fig)
print("wrote", out)

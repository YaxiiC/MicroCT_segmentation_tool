"""Check whether circled large vesicles are above-threshold (cupping) or open-to-exterior."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

import porosity_ct as pc

ROOT = Path(__file__).resolve().parent
folder = ROOT / "6-11"
vol = pc.load_volume(pc.list_tiffs(folder), downsample=2)
pixel_um = float(pc.parse_header(folder)["Pixel Size"])
result = pc.analyze_volume(vol, pixel_um, 2, None, ball_um=15.0, exclude_edge_um=15.0)

z = 158
T = result["threshold"]
sl = vol[z]
solid = result["solid"][z]
closed = result["closed_pore"][z]
opened = result["open_pore"][z]
wrap = result["open_wrap"][z]
env = result["envelope"][z]
pore = closed | opened | wrap

# Find roundish dark-looking regions that are NOT labeled: local mean below neighbor solid
# Candidate: morphological holes in solid on this slice that are unlabeled
filled2d = ndi.binary_fill_holes(solid)
holes2d = filled2d & ~solid
unlabeled_holes = holes2d & ~pore
lab, n = ndi.label(unlabeled_holes)
sizes = np.bincount(lab.ravel())
sizes[0] = 0
print(f"2D holes in solid on z={z}: {n}")
print(f"  unlabeled hole voxels: {int(unlabeled_holes.sum())}")
print(f"  labeled hole voxels (closed/open/wrap): {int((holes2d & pore).sum())}")

print("\nLargest unlabeled 2D holes (should be the missed vesicles):")
for k in np.argsort(sizes)[::-1][:12]:
    if sizes[k] < 20:
        break
    m = lab == k
    ys, xs = np.where(m)
    cy, cx = int(ys.mean()), int(xs.mean())
    g = sl[m]
    # gray of surrounding solid ring
    dil = ndi.binary_dilation(m, iterations=2) & solid
    sg = sl[dil] if dil.any() else np.array([np.nan])
    in_env = float(env[m].mean())
    # 3D: closed fill?
    print(
        f"  size={sizes[k]:4d} cyx=({cy:3d},{cx:3d}) "
        f"hole_gray={g.mean():.0f} (med {np.median(g):.0f}) "
        f"solid_ring={np.nanmean(sg):.0f}  above_T_frac={(g > T).mean():.2f} "
        f"env={in_env:.2f} "
        f"in_closed3d={result['closed_pore'][z][m].mean():.2f} "
        f"in_open3d={result['open_pore'][z][m].mean():.2f} "
        f"in_wrap3d={result['open_wrap'][z][m].mean():.2f}"
    )

# Also: 2D holes that ARE above T (cupping) — counted as solid wrongly
pseudo = (sl < np.percentile(sl[solid], 40)) & solid & env  # dark-ish solid
labp, np_ = ndi.label(pseudo)
print(f"\nDark-solid patches (possible cupping vesicles counted as solid): comps={np_}")

# Overlay: cyan = unlabeled 2D holes (missed), blue/orange/magenta = counted
vis, lo, hi = pc._stretch(sl)
rgb = pc._overlay_rim(vis, closed, opened, wrap)
rgb[unlabeled_holes] = [0.0, 1.0, 1.0]  # cyan = missed 2D holes

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
fig.suptitle(
    "Cyan = 2D holes in solid with NO pore label (likely open-to-exterior OR cupping)\n"
    "Blue=closed  orange=open counted  magenta=excluded wrap"
)
axes[0].imshow(sl, cmap="gray", vmin=lo, vmax=hi)
axes[0].set_title("CT")
axes[1].imshow(rgb)
axes[1].set_title("labels + cyan missed holes")
# zoom on densest missed region
if unlabeled_holes.any():
    ys, xs = np.where(unlabeled_holes)
    cy, cx = int(np.median(ys)), int(np.median(xs))
else:
    cy, cx = sl.shape[0] // 2, sl.shape[1] // 2
r = 90
y0, y1 = max(0, cy - r), min(sl.shape[0], cy + r)
x0, x1 = max(0, cx - r), min(sl.shape[1], cx + r)
axes[2].imshow(rgb[y0:y1, x0:x1])
axes[2].set_title("zoom")
for ax in axes:
    ax.axis("off")
out = ROOT / "_porosity_results" / "6-11_qc" / "missed_holes_cyan.png"
fig.tight_layout()
pc._safe_savefig(fig, out, dpi=170)
plt.close(fig)
print("wrote", out)

# How many unlabeled hole voxels are actually above T vs below?
print("\nUnlabeled hole gray vs T:")
if unlabeled_holes.any():
    g = sl[unlabeled_holes]
    print(f"  n={g.size} mean={g.mean():.0f} frac_above_T={(g > T).mean():.3f}")
    print(f"  p5={np.percentile(g,5):.0f} p50={np.median(g):.0f} p95={np.percentile(g,95):.0f}")

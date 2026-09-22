"""Line profiles through user-missed vesicle centers."""
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
otsu = result["otsu_threshold"]
sl = vol[z]
solid = result["solid"][z]
closed = result["closed_pore"][z]
opened = result["open_pore"][z]
wrap = result["open_wrap"][z]
voxel_um = result["voxel_um"]

# Centers from largest holes / likely user circle region
centers = [(217, 329), (230, 383), (188, 297), (273, 311), (160, 293)]

fig, axes = plt.subplots(len(centers), 2, figsize=(11, 3.2 * len(centers)))
fig.suptitle(
    f"Why circled voids are missed (z={z})\n"
    f"If core gray > air T={T}, the interior is labeled SOLID (no blue/orange)",
    fontsize=11,
)
vis, lo, hi = pc._stretch(sl)
rgb = pc._overlay_rim(vis, closed, opened, wrap)

for i, (cy, cx) in enumerate(centers):
    r = 55
    y0, y1 = max(0, cy - r), min(sl.shape[0], cy + r)
    x0, x1 = max(0, cx - r), min(sl.shape[1], cx + r)
    axes[i, 0].imshow(rgb[y0:y1, x0:x1])
    axes[i, 0].axhline(cy - y0, color="lime", lw=0.8)
    axes[i, 0].plot(cx - x0, cy - y0, "r+", ms=10)
    axes[i, 0].set_ylabel(f"y={cy}\nx={cx}")
    axes[i, 0].set_xticks([])
    axes[i, 0].set_yticks([])
    if i == 0:
        axes[i, 0].set_title("overlay (blue/orange/magenta)")

    xL, xR = max(0, cx - 50), min(sl.shape[1], cx + 50)
    xs = np.arange(xL, xR)
    profile = sl[cy, xL:xR]
    axes[i, 1].plot(xs * voxel_um, profile, color="k", lw=1.0)
    axes[i, 1].axhline(T, color="crimson", lw=1.2, label=f"air T={T}")
    axes[i, 1].axhline(otsu, color="orange", lw=1.0, label=f"Otsu={otsu}")
    # mark where currently solid / pore
    is_solid = solid[cy, xL:xR]
    is_pore = (closed | opened | wrap)[cy, xL:xR]
    axes[i, 1].fill_between(xs * voxel_um, T - 500, T - 100, where=is_pore, color="cyan", alpha=0.35, label="labeled pore")
    axes[i, 1].fill_between(xs * voxel_um, otsu, otsu + 400, where=is_solid, color="gray", alpha=0.25, label="solid")
    core = profile[(xs >= cx - 8) & (xs <= cx + 8)]
    axes[i, 1].set_title(f"core med={np.median(core):.0f}  ({'ABOVE air T — counted solid' if np.median(core) > T else 'below air T'})")
    axes[i, 1].set_xlabel("um")
    axes[i, 1].set_ylabel("gray")
    if i == 0:
        axes[i, 1].legend(fontsize=7, loc="upper right")

out = ROOT / "_porosity_results" / "6-11_qc" / "missed_vesicle_profiles.png"
fig.tight_layout()
pc._safe_savefig(fig, out, dpi=160)
plt.close(fig)
print("wrote", out)

# Quantify how much 'should-be-pore' solid sits in local depressions
# grey closing with ball approx vesicle scale (~20 um -> ~10 vx)
from skimage.morphology import ball, disk

# 2D black tophat on this slice as a quick test
close = ndi.grey_closing(sl, footprint=disk(12))
tophat = close.astype(np.int32) - sl.astype(np.int32)
# strong depression inside current solid
extra = (tophat >= 800) & solid & result["envelope"][z]
print(f"black-tophat>=800 inside solid on z={z}: {int(extra.sum())} voxels")
print(f"  gray med of those={np.median(sl[extra]) if extra.any() else 'n/a'}")
print(f"  frac above air T={(sl[extra] > T).mean() if extra.any() else 0:.2f}")

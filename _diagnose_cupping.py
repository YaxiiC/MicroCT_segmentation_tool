"""Profile gray values across large vesicles that look unlabeled in the middle."""
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

# 2D fill holes -> candidate vesicle regions on this slice
filled = ndi.binary_fill_holes(solid)
holes = filled & ~solid
lab, n = ndi.label(holes)
sizes = np.bincount(lab.ravel())
sizes[0] = 0
print(f"T={T}  largest 2D holes on z={z}:")

# For each large hole, also look at the INTERIOR of the bounding circle including
# voxels currently called solid (cupping cores)
rows = []
for k in np.argsort(sizes)[::-1][:15]:
    if sizes[k] < 30:
        break
    m = lab == k
    ys, xs = np.where(m)
    cy, cx = int(np.round(ys.mean())), int(np.round(xs.mean()))
    # radius from hole extent
    rad = int(max(np.ptp(ys), np.ptp(xs)) // 2 + 2)
    yy, xx = np.ogrid[: sl.shape[0], : sl.shape[1]]
    disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= rad * rad
    core = disk & solid  # solid voxels inside the vesicle disk = cupping core?
    g_hole = sl[m]
    g_core = sl[core] if core.any() else np.array([])
    print(
        f"  hole={sizes[k]:4d} cyx=({cy},{cx}) R~{rad} "
        f"hole_gray med={np.median(g_hole):.0f} "
        f"core_solid_vox={int(core.sum())} "
        f"core_med={np.median(g_core) if g_core.size else float('nan'):.0f} "
        f"closed={closed[m].mean():.2f} open={opened[m].mean():.2f} wrap={wrap[m].mean():.2f}"
    )
    rows.append((cy, cx, rad, m, core))

# Mark cupping cores: solid voxels that sit inside a 2D-filled particle hull
# and are darker than typical solid - local depression
particle2d = ndi.binary_fill_holes(solid)
# distance from exterior of particle
dist = ndi.distance_transform_edt(particle2d)
interior_solid = solid & (dist >= 3)
# compare each interior solid voxel to local neighborhood max (walls are brighter)
from scipy.ndimage import maximum_filter, minimum_filter

local_min = minimum_filter(sl.astype(np.float32), size=9)
# cupping core candidate: solid, deep inside, and a local minimum region below Otsu but maybe above air T
otsu = result["otsu_threshold"]
cup = interior_solid & (sl < otsu) & (sl <= local_min + 50)
labc, nc = ndi.label(cup)
sc = np.bincount(labc.ravel())
sc[0] = 0
print(f"\nPossible cupping cores (solid but dark local min, <Otsu={otsu}): comps={nc}")
for k in np.argsort(sc)[::-1][:10]:
    if sc[k] < 40:
        break
    m = labc == k
    ys, xs = np.where(m)
    cy, cx = int(ys.mean()), int(xs.mean())
    g = sl[m]
    print(
        f"  size={sc[k]:4d} cyx=({cy},{cx}) gray med={np.median(g):.0f} "
        f"mean={g.mean():.0f} frac_above_airT={(g > T).mean():.2f}"
    )

vis, lo, hi = pc._stretch(sl)
rgb = pc._overlay_rim(vis, closed, opened, wrap)
# yellow = cupping cores (currently solid, should maybe be pore)
rgb[cup] = rgb[cup] * 0.25 + np.array([1.0, 1.0, 0.0]) * 0.75

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
fig.suptitle(
    f"Yellow = dark SOLID interiors (cupping cores above air T={T}, below Otsu={otsu})\n"
    "These look like pores but are currently counted as solid — not blue/orange"
)
axes[0].imshow(sl, cmap="gray", vmin=lo, vmax=hi)
axes[0].set_title("CT")
axes[1].imshow(rgb)
axes[1].set_title("labels + yellow cupping cores")
if sc.max() > 0:
    k = int(np.argmax(sc))
    ys, xs = np.where(labc == k)
    cy, cx = int(np.median(ys)), int(np.median(xs))
else:
    cy, cx = 220, 330
r = 80
axes[2].imshow(rgb[max(0, cy - r) : cy + r, max(0, cx - r) : cx + r])
axes[2].set_title("zoom")
for ax in axes:
    ax.axis("off")
out = ROOT / "_porosity_results" / "6-11_qc" / "cupping_cores_yellow.png"
fig.tight_layout()
pc._safe_savefig(fig, out, dpi=170)
plt.close(fig)
print("wrote", out)

# Line profile through a big cupping core if any
if sc.max() >= 40:
    k = int(np.argmax(sc))
    ys, xs = np.where(labc == k)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    x0, x1 = max(0, cx - 60), min(sl.shape[1], cx + 60)
    profile = sl[cy, x0:x1]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(np.arange(x0, x1) * result["voxel_um"], profile, color="k")
    ax.axhline(T, color="crimson", label=f"air T={T}")
    ax.axhline(otsu, color="orange", label=f"Otsu={otsu}")
    ax.set_xlabel("um")
    ax.set_ylabel("gray")
    ax.set_title(f"Line through cupping core z={z} y={cy}")
    ax.legend()
    out2 = ROOT / "_porosity_results" / "6-11_qc" / "cupping_line_profile.png"
    fig.tight_layout()
    pc._safe_savefig(fig, out2, dpi=160)
    plt.close(fig)
    print("wrote", out2)

"""Find large round dark-looking regions whose centers are SOLID (cupping)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import regionprops

import porosity_ct as pc

ROOT = Path(__file__).resolve().parent
vol = pc.load_volume(pc.list_tiffs(ROOT / "6-11"), downsample=2)
pixel_um = float(pc.parse_header(ROOT / "6-11")["Pixel Size"])
result = pc.analyze_volume(vol, pixel_um, 2, None, ball_um=15.0, exclude_edge_um=15.0)

z = 158
T = result["threshold"]
otsu = result["otsu_threshold"]
sl = vol[z]
solid = result["solid"][z]
closed = result["closed_pore"][z]
opened = result["open_pore"][z]
wrap = result["open_wrap"][z]
env = result["envelope"][z]
pore = closed | opened | wrap

# Invert: dark blobs = local basins via threshold relative to local mean of solid
# Use markers: voxels darker than (local solid median - delta)
from scipy.ndimage import uniform_filter

local = uniform_filter(sl.astype(np.float32), size=25)
# Candidate vesicle: much darker than local neighborhood, inside envelope
darker = (local - sl) >= 1500
cand = darker & env
lab, n = ndi.label(cand)
print(f"local-dark blobs: {n}")
props = regionprops(lab, intensity_image=sl)
props = sorted(props, key=lambda p: p.area, reverse=True)

print(f"{'area':>5} {'cy':>4} {'cx':>4} {'gmed':>6} {'solid%':>7} {'pore%':>6} {'aboveT%':>8} note")
missed = np.zeros_like(sl, dtype=bool)
for p in props[:25]:
    cy, cx = p.centroid
    cy, cx = int(cy), int(cx)
    m = lab == p.label
    g = sl[m]
    sfrac = solid[m].mean()
    pfrac = pore[m].mean()
    above = (g > T).mean()
    note = ""
    if sfrac > 0.4 and pfrac < 0.3:
        note = "MISSED-as-solid"
        missed |= m
    elif pfrac > 0.5:
        note = "counted"
    print(
        f"{int(p.area):5d} {cy:4d} {cx:4d} {np.median(g):6.0f} {100*sfrac:6.1f}% {100*pfrac:5.1f}% {100*above:7.1f}% {note}"
    )

vis, lo, hi = pc._stretch(sl)
rgb = pc._overlay_rim(vis, closed, opened, wrap)
rgb[missed] = [1, 1, 0]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("Yellow = locally dark blobs that are mostly SOLID (true missed vesicles)")
axes[0].imshow(sl, cmap="gray", vmin=lo, vmax=hi)
axes[1].imshow(rgb)
for ax in axes:
    ax.axis("off")
out = ROOT / "_porosity_results" / "6-11_qc" / "local_dark_missed.png"
fig.tight_layout()
pc._safe_savefig(fig, out, dpi=160)
plt.close(fig)
print("wrote", out, "missed_vox", int(missed.sum()))

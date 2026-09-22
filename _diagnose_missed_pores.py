"""Diagnose large dark voids that look like pores but are not labeled."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

import porosity_ct as pc

ROOT = Path(__file__).resolve().parent
folder = ROOT / "6-11"
header = pc.parse_header(folder)
files = pc.list_tiffs(folder)
vol = pc.load_volume(files, downsample=2)
pixel_um = float(header["Pixel Size"])
result = pc.analyze_volume(vol, pixel_um, 2, None, ball_um=15.0, exclude_edge_um=15.0)

z = 158
solid = result["solid"]
closed = result["closed_pore"]
opened = result["open_pore"]
wrap = result["open_wrap"]
env = result["envelope"]
pore_any = closed | opened | wrap
T = result["threshold"]
sl = vol[z]

ys, xs = np.where(env[z])
print("slice z=", z, "T=", T)
print("  env bbox y", ys.min(), ys.max(), "x", xs.min(), xs.max())

# Center-right of particle (approx red-circle region on XY orthogonal panel)
cx0 = int(xs.min() + 0.55 * (xs.max() - xs.min()))
cx1 = int(xs.min() + 0.88 * (xs.max() - xs.min()))
cy0 = int(ys.min() + 0.30 * (ys.max() - ys.min()))
cy1 = int(ys.min() + 0.72 * (ys.max() - ys.min()))
box = (slice(cy0, cy1), slice(cx0, cx1))
print("  probe box y", cy0, cy1, "x", cx0, cx1)

dark = (sl < T) & (sl > 0)
print("  in box mean gray", float(sl[box].mean()))
print("  solid/closed/open/wrap/env fracs:",
      float(solid[z][box].mean()),
      float(closed[z][box].mean()),
      float(opened[z][box].mean()),
      float(wrap[z][box].mean()),
      float(env[z][box].mean()))
print("  dark&~env frac", float((dark & ~env[z])[box].mean()))
print("  dark&env&~pore frac", float((dark & env[z] & ~pore_any[z])[box].mean()))

cand = dark[box] & ~solid[z][box]
lab, n = ndi.label(cand)
sizes = np.bincount(lab.ravel())
sizes[0] = 0
print("  dark non-solid components in box:", n)
for k in np.argsort(sizes)[::-1][:10]:
    if sizes[k] == 0:
        continue
    full = np.zeros_like(sl, dtype=bool)
    full[box] = lab == k
    ys2, xs2 = np.where(full)
    cy, cx = int(ys2.mean()), int(xs2.mean())
    g = sl[full]
    print(
        f"  cc size={sizes[k]:5d} cyx=({cy},{cx}) gray mean={g.mean():.0f} "
        f"med={np.median(g):.0f} env={env[z][full].mean():.2f} "
        f"closed={closed[z][full].mean():.2f} open={opened[z][full].mean():.2f} "
        f"wrap={wrap[z][full].mean():.2f}"
    )

# 3D status of the largest unlabeled dark blob in the box
unlabeled = dark & env[z] & ~pore_any[z] & ~solid[z]
# also consider dark outside env in box - those are "exterior"
outside_dark = dark & ~env[z]
print("\nOutside-envelope dark in box (would never count):", int(outside_dark[box].sum()))
print("Inside-env unlabeled dark in box:", int(unlabeled[box].sum()))

# Pick largest dark component in whole slice near probe, regardless of env
cand2 = dark & ~solid[z]
lab2, _ = ndi.label(cand2)
# restrict to box seeds
seeds = lab2[box]
keep_ids = np.unique(seeds[seeds > 0])
comp_sizes = [(int(i), int((lab2 == i).sum())) for i in keep_ids]
comp_sizes.sort(key=lambda t: t[1], reverse=True)
print("\nLargest dark non-solid comps intersecting box (full-slice size):")
for i, sz in comp_sizes[:8]:
    m = lab2 == i
    ys2, xs2 = np.where(m)
    cy, cx = int(ys2.mean()), int(xs2.mean())
    g = sl[m]
    # 3D: is this component connected to exterior air via 3D below-threshold?
    # Check fraction inside 3D envelope / closed / open
    # Need 3D component from this 2D blob - use one seed voxel
    seed = (z, cy, cx)
    print(
        f"  id={i} size2d={sz} cyx=({cy},{cx}) gray={g.mean():.0f} "
        f"env2d={env[z][m].mean():.2f} closed2d={closed[z][m].mean():.2f} "
        f"open2d={opened[z][m].mean():.2f} wrap2d={wrap[z][m].mean():.2f}"
    )
    # 3D flood of below-threshold connected to seed
    air3 = (vol < T) & (vol > 0)
    # restrict flood start
    if not air3[seed]:
        # find nearest air voxel in mask m
        zz = z
        found = False
        for yy, xx in zip(ys2[:: max(1, len(ys2)//20)], xs2[:: max(1, len(xs2)//20)]):
            if air3[zz, yy, xx]:
                seed = (zz, int(yy), int(xx))
                found = True
                break
        if not found:
            print("    no air seed")
            continue
    struct = ndi.generate_binary_structure(3, 1)
    # limited BFS via label on cropped region around particle
    # Use ndi.label on air3 then take label of seed
    # Too expensive full volume? 500^3 is ok once
    break  # label full volume once below

print("\nLabeling 3D air (vol < T) ...")
air3 = (vol < T) & (vol > 0)
lab3, n3 = ndi.label(air3)
print("  3D air components:", n3)

# exterior air reference: voxels outside envelope
ext = ~env & air3
ext_labels = set(np.unique(lab3[ext]))
ext_labels.discard(0)
print("  labels touching exterior:", len(ext_labels))

for i, sz in comp_sizes[:6]:
    m = lab2 == i
    ys2, xs2 = np.where(m)
    # sample a few voxels
    idxs = np.linspace(0, len(ys2) - 1, num=min(12, len(ys2))).astype(int)
    labels_hit = [int(lab3[z, ys2[j], xs2[j]]) for j in idxs]
    labels_hit = [L for L in labels_hit if L]
    if not labels_hit:
        print(f"  id={i}: no 3D air label")
        continue
    L = max(set(labels_hit), key=labels_hit.count)
    touches_ext = L in ext_labels
    n_env = int((lab3 == L)[env].sum())
    n_tot = int((lab3 == L).sum())
    print(
        f"  id={i} size2d={sz} 3Dlab={L} 3Dvox={n_tot} inside_env={n_env} "
        f"touches_exterior={touches_ext} "
        f"frac_in_env={n_env / n_tot if n_tot else 0:.3f}"
    )

vis, lo, hi = pc._stretch(sl)
rgb = pc._overlay_rim(vis, closed[z], opened[z], wrap[z])
unlab = env[z] & ~solid[z] & ~pore_any[z]
rgb2 = np.stack([vis, vis, vis], axis=-1)
rgb2[unlab] = [1.0, 0.0, 0.0]
rgb2[outside_dark & ~solid[z]] = rgb2[outside_dark & ~solid[z]] * 0.3 + np.array([0.2, 1.0, 0.2]) * 0.7

fig, axes = plt.subplots(2, 2, figsize=(10, 9))
fig.suptitle(f"Missed-pore diagnosis z={z}  T={T}")
axes[0, 0].imshow(sl, cmap="gray", vmin=lo, vmax=hi)
axes[0, 0].add_patch(plt.Rectangle((cx0, cy0), cx1 - cx0, cy1 - cy0, fill=False, ec="r", lw=1.5))
axes[0, 0].set_title("CT + probe box")
axes[0, 1].imshow(rgb)
axes[0, 1].add_patch(plt.Rectangle((cx0, cy0), cx1 - cx0, cy1 - cy0, fill=False, ec="r", lw=1.5))
axes[0, 1].set_title("current labels")
axes[1, 0].imshow(rgb2)
axes[1, 0].set_title("RED=in env unlabeled; GREEN=dark outside env")
axes[1, 1].imshow(sl[box], cmap="gray", vmin=lo, vmax=hi)
axes[1, 1].imshow(rgb[box], alpha=0.55)
axes[1, 1].set_title("zoom")
for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])
out = ROOT / "_porosity_results" / "6-11_qc" / "missed_pore_diag.png"
fig.tight_layout()
pc._safe_savefig(fig, out, dpi=160)
plt.close(fig)
print("wrote", out)

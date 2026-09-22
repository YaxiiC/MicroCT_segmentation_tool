"""Tune cupping recovery on solid cores, not air-ring seeds."""
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import disk

import porosity_ct as pc

ROOT = Path(__file__).resolve().parent
vol = pc.load_volume(pc.list_tiffs(ROOT / "6-11"), downsample=2)
pixel_um = float(pc.parse_header(ROOT / "6-11")["Pixel Size"])
fov2d = pc.fov_disk(vol.shape[1:])
otsu = pc.pick_otsu(vol, fov2d)
air_t = pc.pick_air_threshold(vol, pc.exterior_air_mask(vol, fov2d, otsu), 2.5)
z = 158
sl = vol[z]
voxel_um = pixel_um * 2
solid = ndi.binary_closing((sl > air_t) & fov2d, iterations=1)
lab, _ = ndi.label(solid)
sizes = np.bincount(lab.ravel()); sizes[0] = 0
solid = lab == int(np.argmax(sizes))
env = ndi.binary_fill_holes(ndi.binary_closing(solid, structure=disk(8)))

# Find solid cupping cores: solid voxels inside 2D-filled holes' bounding disks
# that are darker than local walls
filled = ndi.binary_fill_holes(solid)
holes = filled & ~solid
labh, _ = ndi.label(holes)
sh = np.bincount(labh.ravel()); sh[0] = 0
core_seeds = []
for k in np.argsort(sh)[::-1][:8]:
    if sh[k] < 80:
        break
    m = labh == k
    ys, xs = np.where(m)
    cy, cx = int(ys.mean()), int(xs.mean())
    rad = int(max(np.ptp(ys), np.ptp(xs)) // 2 + 3)
    yy, xx = np.ogrid[: sl.shape[0], : sl.shape[1]]
    diskm = (yy - cy) ** 2 + (xx - cx) ** 2 <= rad * rad
    core = diskm & solid
    if not core.any():
        continue
    # darkest solid voxel in core as seed
    vals = sl[core]
    # pick percentile 20 darkest solid in disk
    thr = np.percentile(vals, 20)
    dark_core = core & (sl <= thr)
    ys2, xs2 = np.where(dark_core)
    cy2, cx2 = int(ys2.mean()), int(xs2.mean())
    core_seeds.append((cy2, cx2, sh[k], float(np.median(sl[dark_core]))))
    print(f"hole={sh[k]} ring_center=({cy},{cx}) core_seed=({cy2},{cx2}) core_med={np.median(sl[dark_core]):.0f}")

print(f"\nair T={air_t} Otsu={otsu}")
for r_um in (16, 20, 25, 30):
    r = max(2, int(round(r_um / voxel_um)))
    closed = ndi.grey_closing(sl.astype(np.float32), footprint=disk(r))
    dep = closed - sl.astype(np.float32)
    deps = [float(dep[y, x]) for y, x, _, _ in core_seeds]
    print(f"\nR={r_um}um ({r}px) dep@cores={['%.0f'%d for d in deps]}")
    for h in (800, 1200, 1600, 2000, 2500):
        cand = (dep >= h) & solid & env
        hit = sum(1 for y, x, _, _ in core_seeds if cand[y, x])
        labc, _ = ndi.label(cand)
        sc = np.bincount(labc.ravel()); sc[0] = 0
        keep = sc >= 27
        keep[0] = False
        kept = keep[labc]
        # false positive proxy: cand overlapping bright solid ( > otsu + 2k )
        bright_fp = kept & (sl > otsu + 2000)
        print(
            f"  h={h}: kept={kept.sum():5d} comps={int(keep.sum()):3d} "
            f"cores_hit={hit}/{len(core_seeds)} bright_fp={bright_fp.sum():5d} "
            f"med={np.median(sl[kept]) if kept.any() else 0:.0f}"
        )

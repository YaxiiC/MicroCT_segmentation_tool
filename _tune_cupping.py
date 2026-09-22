"""Tune black-tophat height for cupping recovery on one slice."""
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import disk

import porosity_ct as pc

ROOT = Path(__file__).resolve().parent
folder = ROOT / "6-11"
vol = pc.load_volume(pc.list_tiffs(folder), downsample=2)
pixel_um = float(pc.parse_header(folder)["Pixel Size"])
# cheap solid without full rolling ball: air threshold only on one middle range
fov2d = pc.fov_disk(vol.shape[1:])
otsu = pc.pick_otsu(vol, fov2d)
air_mask = pc.exterior_air_mask(vol, fov2d, otsu)
air_t = pc.pick_air_threshold(vol, air_mask, 2.5)
z = 158
sl = vol[z]
solid = (sl > air_t) & fov2d
solid = ndi.binary_closing(solid, iterations=1)
# largest CC 2d
lab, n = ndi.label(solid)
sizes = np.bincount(lab.ravel()); sizes[0]=0
solid = lab == int(np.argmax(sizes))
env = ndi.binary_fill_holes(ndi.binary_closing(solid, structure=disk(8)))

voxel_um = pixel_um * 2
print(f"T={air_t} Otsu={otsu} voxel={voxel_um:.2f}")

# known cupping centers from earlier
seeds = [(217, 329), (230, 383), (160, 293), (273, 311)]
# known true air pores (should already be below T) - just for reference

for r_um in (12, 16, 20, 25):
    r = max(2, int(round(r_um / voxel_um)))
    closed = ndi.grey_closing(sl.astype(np.float32), footprint=disk(r))
    dep = closed - sl.astype(np.float32)
    print(f"\nR={r_um} um ({r} px)  dep at seeds:", [float(dep[y, x]) for y, x in seeds])
    for h in (1000, 1500, 2000, 2500, 3000):
        cand = (dep >= h) & solid & env
        # how many seeds recovered?
        hit = sum(1 for y, x in seeds if cand[y, x])
        # size filter
        labc, nc = ndi.label(cand)
        sc = np.bincount(labc.ravel()); sc[0]=0
        keep = sc >= 20
        keep[0] = False
        kept = keep[labc]
        print(f"  h={h}: cand={cand.sum():5d} kept20={kept.sum():5d} comps={int(keep.sum())} seeds_hit={hit}/4  med_gray={np.median(sl[kept]) if kept.any() else 0:.0f}")

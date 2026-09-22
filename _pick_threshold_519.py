"""Compare auto-threshold mixes on a dataset and write a QC grid."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

import porosity_ct as pc

DATASET = "5-19"
MIXES = [0.0, 0.35, 0.50, 0.65, 0.80]
DOWNSAMPLE = 2
ROOT = Path(__file__).resolve().parent


def main() -> None:
    folder = ROOT / DATASET
    header = pc.parse_header(folder)
    pixel_um = float(header["Pixel Size"])
    print(f"Loading {DATASET} ...")
    vol = pc.load_volume(pc.list_tiffs(folder), DOWNSAMPLE)
    fov2d = pc.fov_disk(vol.shape[1:])
    otsu_t = pc.pick_otsu(vol, fov2d)
    air_mask = pc.exterior_air_mask(vol, fov2d, otsu_t)
    air_t = pc.pick_air_threshold(vol, air_mask, k_std=2.5)
    print(f"air T={air_t}  Otsu={otsu_t}  shape={vol.shape}")

    # Pick informative slices from a mid mix solid
    mid_t = int(round(air_t + 0.5 * (otsu_t - air_t)))
    solid_mid = np.empty(vol.shape, dtype=bool)
    for z in range(vol.shape[0]):
        solid_mid[z] = (vol[z] > mid_t) & fov2d
    lab, n = ndi.label(solid_mid)
    if n:
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        solid_mid = lab == int(np.argmax(counts))
    area = solid_mid.sum(axis=(1, 2))
    keep = np.flatnonzero(area > 0.05 * area.max())
    zs = [
        int(keep[np.argmax(area[keep])]),
        int(keep[len(keep) // 2]),
        int(keep[int(0.8 * (len(keep) - 1))]),
    ]
    zs = list(dict.fromkeys(zs))
    print("QC slices", zs)

    rows = []
    fig, axes = plt.subplots(
        len(zs),
        1 + len(MIXES),
        figsize=(3.2 * (1 + len(MIXES)), 3.1 * len(zs)),
        squeeze=False,
    )
    fig.suptitle(
        f"{DATASET} threshold mix comparison\n"
        f"air={air_t}  Otsu={otsu_t}  |  yellow/blue-ish = pores (fill_holes - solid)  "
        f"higher mix = more complete bright cores, risk of broken thin walls",
        fontsize=11,
    )

    for j, mix in enumerate(MIXES):
        t = int(round(air_t + mix * (otsu_t - air_t)))
        solid = np.empty(vol.shape, dtype=bool)
        struct2d = ndi.generate_binary_structure(2, 1)
        for z in range(vol.shape[0]):
            sl = (vol[z] > t) & fov2d
            sl = ndi.binary_closing(sl, structure=struct2d, iterations=1)
            solid[z] = sl
        lab, n = ndi.label(solid)
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        solid = lab == int(np.argmax(counts)) if n else solid
        filled = ndi.binary_fill_holes(solid)
        closed = filled & ~solid
        # quick open proxy: not used for phi here; just show closed + dark non-solid near particle
        n_s = int(solid.sum())
        n_c = int(closed.sum())
        den = n_s + n_c
        phi_c = 100 * n_c / den if den else float("nan")
        # rough "missing core" score: solid voxels that are local depressions
        # count dark-looking solid (below otsu) as incomplete
        dark_solid = solid & (vol < otsu_t) & (vol > t)
        frac_dark_solid = float(dark_solid.sum() / max(n_s, 1))
        rows.append(
            {
                "mix": mix,
                "threshold": t,
                "phi_closed_quick_pct": phi_c,
                "n_solid": n_s,
                "n_closed": n_c,
                "frac_solid_below_otsu": frac_dark_solid,
            }
        )
        print(
            f"  mix={mix:.2f} T={t}  closed~{phi_c:.2f}%  "
            f"dark_solid_frac={100 * frac_dark_solid:.1f}%"
        )

        for i, z in enumerate(zs):
            if j == 0:
                vis, lo, hi = pc._stretch(vol[z])
                axes[i, 0].imshow(vol[z], cmap="gray", vmin=lo, vmax=hi)
                axes[i, 0].set_ylabel(f"z={z}")
                axes[i, 0].set_title("CT" if i == 0 else "")
                axes[i, 0].set_xticks([])
                axes[i, 0].set_yticks([])
            vis, lo, hi = pc._stretch(vol[z])
            rgb = np.stack([vis, vis, vis], axis=-1)
            rgb[closed[z]] = rgb[closed[z]] * 0.3 + np.array([0.0, 0.45, 0.7]) * 0.7
            # also tint remaining below-T non-solid inside filled particle lightly yellow
            openish = filled[z] & ~solid[z] & ~closed[z]
            rgb[openish] = rgb[openish] * 0.35 + np.array([0.95, 0.78, 0.1]) * 0.65
            axes[i, 1 + j].imshow(np.clip(rgb, 0, 1))
            axes[i, 1 + j].axis("off")
            if i == 0:
                axes[i, 1 + j].set_title(f"mix={mix:.2f}\nT={t}\nφc~{phi_c:.1f}%")

    out_dir = ROOT / "_porosity_results" / "5-19_threshold_pick"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    out_png = out_dir / "threshold_mix_compare.png"
    pc._safe_savefig(fig, out_png, dpi=160)
    plt.close(fig)
    (out_dir / "threshold_mix_compare.json").write_text(
        json.dumps(
            {"dataset": DATASET, "air_t": air_t, "otsu_t": otsu_t, "rows": rows},
            indent=2,
        ),
        encoding="utf-8",
    )
    print("wrote", out_png)

    # Recommend: prefer lower dark_solid_frac (fewer incomplete bright cores)
    # but not so high mix that solid collapses. Score = dark_solid_frac + penalty if phi jumps wild
    best = min(rows, key=lambda r: r["frac_solid_below_otsu"] + 0.002 * abs(r["mix"] - 0.5))
    # Prefer mix where dark_solid drops most vs mix=0, without going to 0.8 unless needed
    base = rows[0]["frac_solid_below_otsu"]
    scored = []
    for r in rows:
        gain = base - r["frac_solid_below_otsu"]
        # penalize extreme mixes slightly
        score = gain - 0.15 * abs(r["mix"] - 0.5)
        scored.append((score, r))
    scored.sort(key=lambda t: t[0], reverse=True)
    rec = scored[0][1]
    print(
        f"\nRECOMMEND mix={rec['mix']:.2f}  T={rec['threshold']}  "
        f"(reduces incomplete bright solid cores vs air-only)"
    )
    (out_dir / "recommendation.txt").write_text(
        f"Recommended for {DATASET}: --thresh-mix {rec['mix']:.2f}  (T={rec['threshold']})\n"
        f"air={air_t} Otsu={otsu_t}\n"
        f"Then: python porosity_ct.py --dataset \"{DATASET}\" --prepare-edit --thresh-mix {rec['mix']:.2f}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

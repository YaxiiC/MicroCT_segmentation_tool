import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(r"c:\Users\chris\Downloads\Lunar Soil-CT data")
out = root / "_analysis_preview"
out.mkdir(exist_ok=True)

datasets = [
    "5-19",
    "5-19 High Resolution",
    "5-21",
    "6-11",
    "6-16",
    "MN Lunar Soil",
]

for name in datasets:
    p = root / name
    files = sorted(p.glob("*.tiff"))
    idxs = [len(files) // 4, len(files) // 2, 3 * len(files) // 4]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    first = Image.open(files[0])
    fig.suptitle(f"{name}  n={len(files)}  {first.size}", fontsize=12)
    for i, idx in enumerate(idxs):
        arr = np.array(Image.open(files[idx]))
        nz = arr[arr > 0]
        lo, hi = np.percentile(nz, [1, 99]) if nz.size else (0, 1)
        axes[0, i].imshow(arr, cmap="gray", vmin=lo, vmax=hi)
        axes[0, i].set_title(f"slice {idx + 1}/{len(files)}")
        axes[0, i].axis("off")
        hist, bins = np.histogram(arr.ravel(), bins=256, range=(0, max(int(arr.max()), 1)))
        axes[1, i].plot(bins[:-1], hist, lw=0.8)
        axes[1, i].set_yscale("log")
        axes[1, i].set_xlabel("gray")
        axes[1, i].set_ylabel("count (log)")
        print(
            name,
            "slice",
            idx + 1,
            "min",
            int(arr.min()),
            "max",
            int(arr.max()),
            "mean",
            round(float(arr.mean()), 1),
            "zeros%",
            round(100 * float((arr == 0).mean()), 1),
        )
    plt.tight_layout()
    fig.savefig(out / f"{name.replace(' ', '_')}_preview.png", dpi=120)
    plt.close()
    print("saved", name)

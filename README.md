# Lunar soil micro-CT porosity tools (`MicroCT_segmentation_tool`)

Portable pipeline: keep `porosity_ct.py` / `porosity_edit.py` next to your dataset folders. Paths are stored **relative to this project folder**, so the same copy works on any PC.

**This repo contains code only** — raw TIFF stacks and analysis results are gitignored.

## Layout

```
MicroCT_segmentation_tool/   ← clone / project root
  porosity_ct.py
  porosity_edit.py
  requirements.txt
  README.md
  <your-dataset>/            ← local only: TIFF stack + Header.txt (not in git)
  _edit_sessions/            ← local: created by the editor
  _porosity_results/         ← local: QC + CSV outputs
```

## Setup

```bash
git clone <your-fork-or-repo-url> MicroCT_segmentation_tool
cd MicroCT_segmentation_tool
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

Or with conda:

```bash
conda create -n porosity python=3.10
conda activate porosity
pip install -r requirements.txt
```

Put each CT stack in a subfolder next to the scripts (folder name = dataset name), with `*.tif` / `*.tiff` and optional `Header.txt`.

## Run

Interactive threshold + paint editor:

```bash
python porosity_edit.py
```

Batch porosity (auto):

```bash
python porosity_ct.py --dataset "6-11"
python porosity_ct.py --all
```

Resume an edit session:

```bash
python porosity_edit.py --session "_edit_sessions/6-11"
```

## Notes

- Datasets are **auto-discovered** (any subfolder with `.tif` / `.tiff` files).
- Do not commit `C:\Users\...` absolute paths; always run from the project folder.
- Existing sessions with old absolute `source_folder` paths are remapped by dataset name when possible.

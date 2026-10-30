# Vercel deployment

This directory is a **standalone Vercel project**. It is generated, not hand-written:

```bash
python scripts/export_portable.py     # from the repository root
```

That command produces everything here:

| Path | What it is |
|---|---|
| `data/portable_model.json` | Affine surrogate of the champion pipeline (~20 KB) |
| `data/sound_stats.parquet` | Training-window sound statistics the feature builder needs |
| `data/api/*.json` | Pre-computed analytics payloads, stamped with their measurement time |
| `src/` | Mirror of the packages the function imports |
| `public/index.html` | Copy of `dashboard/index.html` |

## Why it is packaged this way

The full stack — scikit-learn, SciPy, pandas, CatBoost, LightGBM, DuckDB — is about **478 MB**
installed. Vercel's serverless limit is 250 MB unzipped, so the application as written cannot be
deployed there directly.

Two substitutions make it fit:

1. **Prediction stays live.** The champion is a linear pipeline, so its exact behaviour is
   exported into a small affine surrogate and evaluated with NumPy. `tests/test_portable.py`
   asserts parity with the real fitted model at `1e-7` across varied inputs, including unseen
   categories.
2. **Analytics become a snapshot.** The 341 MB sample corpus cannot ship to a function, so the
   read-only payloads are pre-computed and committed. Every response states when it was measured.

## What this deployment does not do

- No persistent prediction history (no database).
- No data import or schema verification against the live source.
- No early-traction engine (blocked everywhere — the source has no repeated observations).

Each of those returns an explicit `503` / `409` with an explanation rather than failing silently.

## Deploy

```bash
cd deploy/vercel
vercel --prod
```

Or from the repository root:

```bash
vercel --cwd deploy/vercel --prod
```

## Regenerating after a retrain

Retraining changes the champion and the artefacts. Re-run the export, then redeploy:

```bash
python scripts/export_portable.py
cd deploy/vercel && vercel --prod
```

If the champion is no longer a linear pipeline, `export_portable_spec` raises rather than
silently exporting an approximation — a tree model would need a different portable strategy.

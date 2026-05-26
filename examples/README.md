# Examples

Six tutorial scripts. Run them in order — each builds on the previous one.

| Script | What it shows |
|---|---|
| `01_load_model.py` | Loads each of the 6 trained checkpoints; prints class, param count, recorded val accuracy, and val-set sha. |
| `02_inspect_data.py` | Loads one paired sample; plots the log-mel spectrogram and four evenly-spaced lip frames. |
| `03_run_inference.py` | Runs 5 val samples through AV and A-only; flags AV-rescue cases (AV right, A-only wrong). |
| `04_extract_features.py` | Pulls AV, A-only, V-fair penultimate features (and the AV `v_mid=0` variant) for a batch; saves an `.npz`. |
| `05_linear_probe.py` | Loads the `.npz`, fits 5-fold LR probes, and shows the calibration thing: AV softmax cliffs down to ~chance when `v_mid=0`, but the linear probe still decodes the word from the same features. |
| `06_gate_visualization.py` | Prints the trained gate scalar α; plots the per-channel gain (annotating the integration channels ch12 + ch22) and the spatio-temporal gate map. |

## Prereqs

You need `processed/` and `models/` populated — see the top-level README quickstart. At minimum:

```bash
python preprocess.py        # → processed/dataset.pt
python paired_dataset.py    # → processed/dataset_av.pt + the video memmap
python train_filtered.py    # → models/audio_only_filtered.pt
python train_av.py          # → models/av_fused.pt
python train_v_only_fair.py # → models/video_only_fair.pt
```

The Tier-1 variant checkpoints (`av_fused_additive.pt`, `av_fused_late.pt`, `av_fused_early.pt`) only get listed by `01_load_model.py`; any missing ones just get skipped.

## Output

All generated figures and `.npz` files land under `examples/output/` (gitignored).

## Runtimes (RTX A6000)

- `01`, `03`, `06`: under 10 seconds each.
- `02`: under 5 seconds.
- `04`: ~30 seconds (extracts 4 conditions × 1000 samples).
- `05`: ~10 seconds (probes load the cached `.npz` from `04`).

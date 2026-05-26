# AV deep-dive — the plan

44 sub-experiments across 5 directions. 39 of them are inference-only on existing checkpoints (Tier-0, about 6–7 h on GPU); 5 are single-variant retrains (Tier-1, ~10 h each). Shared val partition (sha `03c5a87a…`, N=5,244, 180 classes).

## What we already had going in

| Asset | Path | Acc |
|---|---|---:|
| A-only | `models/audio_only_filtered.pt` | 92.70% |
| A-only rawnoise | `models/audio_only_rawnoise_filtered.pt` | 93.25% |
| V-only fair | `models/video_only_fair.pt` | 86.59% |
| AV clean | `models/av_fused.pt` (α=5.20) | 95.80% |
| AV rawnoise | `models/av_fused_rawnoise.pt` (α=6.03) | 95.84% |

Plus the existing MSI battery outputs in `analysis/msi/` and the phonetic clustering CSVs in `analysis/phonetic_clustering_av/`.

## Direction 1 — noise iso-degradation (Tier-0)

| ID | Experiment | Status |
|---|---|---|
| D1.1 | 3×3 model × modality accuracy matrix | done |
| D1.2 | σ_v sweep, V-only and AV | done |
| D1.3 | Frame-drop sweep | done |
| D1.4 | σ_a × σ_v iso-performance grid | done |
| D1.5 | Iso-performance rendezvous (3.4× σ_a, 2.7× σ_v) | done |
| D1.6 | Cross-trained sanity (no leakage) | done |
| D1.7 | Heterogeneous noise (σ_a vs σ_v asymmetry) | done |

## Direction 2 — geometry of penultimate representations (Tier-0)

| ID | Experiment | Status |
|---|---|---|
| D2.1 | 3-condition UMAP from AV-only | done |
| D2.2 | Joint UMAP across A / V / AV | done |
| D2.3 | Integration-driver regression (RF on word features) | done |
| D2.4 | Per-class case studies | done |
| D2.5 | Three-condition viseme probe | done |
| D2.6 | Spread / silhouette by viseme group | done |
| D2.7 | Class-mean dendrograms | done |

## Direction 3 — architectural mechanism

Tier-0 (lesion / sweep on AV-fused and AV-rawnoise):

| ID | Experiment | Status |
|---|---|---|
| D3.5 | Per-channel W_v lesion (mid-mult) | done |
| D3.6 | Tertile lesion | done |
| D3.7 | α inference sweep | done |
| D3.8 | Gate-off ablation | done |
| D3.9 | Per-channel W_a lesion (sanity for D3.5) | done |

Tier-1 (single-variant retrains, ~10 h GPU each):

| ID | Variant | Status |
|---|---|---|
| D3.2 | Late-fusion | done (95.50%) |
| D3.1 | Early-fusion | done (94.22%) |
| D3.10 | Additive gate | done (95.80%) |
| D3.3 | Multi-stage fusion | skipped — Tier-0 lesion + D3.2 / D3.10 made it redundant |
| D3.4 | Recurrent | skipped — E11 binding window already tight, ~10 h cost not worth it |

## Direction 4 — cross-model interpretation (Tier-0)

| ID | Experiment | Status |
|---|---|---|
| D4.1 | Late ensemble vs AV-fused | done |
| D4.2 | Layer-wise linear CKA (A-only ↔ AV) | done |
| D4.3 | RSA per-class | done |
| D4.4 | McGurk-on-A sanity audit | done |
| D4.5 | Linear-probe class accuracy (5-fold CV) | done |
| D4.6–D4.9 | Per-class delta, probe transfer, distribution shape | done |

## Direction 5 — layer-wise information flow (Tier-0)

| ID | Experiment | Status |
|---|---|---|
| D5.1 | Layer-wise linear probes (word / onset / viseme) | done |
| D5.2 | Within-AV CKA across layers | done |
| D5.3 | Information plane (KSG MI) — LOOSE | done, binned estimator saturated |
| D5.4–D5.8 | MEI rank, temporal + spatial saliency, GradCAM | done |
| D5.9 | Cross-model AV × A-only CKA | done |
| D5.10 | Layer-wise saliency map | done |
| D5.11 | Per-block2-channel R² on v_mid features | done |
| D5.12 | RSA layer trajectory | done |

## How it ran

Single GPU. Tier-0 ran in one batched driver (`run_deepdive_tier0.py`) in around 7 hours. Tier-1 retrains queued up sequentially after Tier-0 (3 × ~10 h). Total about 37 h of GPU.

Tier-2 (the "retrain all four variants with modality-dropout regularisation", ~40 h GPU) got skipped — Tier-0 + Tier-1 answered the architecture question without needing it.

## Outputs

All Tier-0 CSVs and PNGs land in `analysis/deepdive/` (parallel to `analysis/msi/`). Tier-1 cross-variant summary in `AV_INTEGRATION_TIER1_CROSS_VARIANT.md`. Results writeup in `AV_INTEGRATION_DEEP_DIVE_RESULTS.md`. Synthesis in `AV_INTEGRATION_DEEP_DIVE_SYNTHESIS.md`.

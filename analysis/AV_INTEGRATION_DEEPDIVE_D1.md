# D1 — cross-modality accuracy & noise iso-degradation

Status: done. Seven sub-experiments. All on val (sha `03c5a87a…`, N=5,244, 180 classes), `.eval()`, seed 0. CSVs and PNGs in `analysis/deepdive/`.

## D1.1 — 3×3 accuracy matrix (clean)

| | A input | V input | AV input |
|---|---:|---:|---:|
| A-trained | 92.70% | — | — |
| V-trained | — | 86.50% | — |
| AV-trained | 0.84% | 44.47% | 95.67% |

AV-trained on AV is +2.97 pp over A-only and +9.17 pp over V-only. The 0.84% cell (AV with v_mid=0) is what we keep calling the v_mid=0 cliff — the gate at α=5.20 saturates the moment v_mid goes to zero. Pretty much everything else in the deep-dive is picking at this. AV with audio=0 holds 44.47% because the visual stream still drives the gate. So basically: the AV model is not a drop-in unisensory classifier.

## D1.2 — σ_v sweep on the lip ROI

V-only collapses fast — 86.5% at σ_v=0, 20.2% at σ_v=0.40, 2.0% at σ_v=1.0. AV-clean degrades way more gently: 95.7% → 77.4% → 22.9%. The AV − V gap peaks at +57.23 pp at σ_v=0.40, which is the visual-side inverse effectiveness. AV-rawnoise (trained with only audio noise) actually generalises better than AV-clean at extreme σ_v (41% vs 32% at σ_v=0.80) — so the benefit kinda transfers across noise channels.

## D1.3 — Frame-drop sweep

AV-clean drops from 98.4% to 70.3% with just one frame zeroed, 13.3% with 5 frames, 1.2% with 10. Architecture issue, not a noise one: `VisualEncoder`'s 3D-conv stem has a (5, 7, 7) kernel, so each zero frame contaminates 5 output positions, and downstream BatchNorm never saw zero frames during training, so the sparse zero hits OOD and gets clamped through ReLU. So treat the frame-drop curve separately from the noise sweeps.

## D1.4 — σ_a × σ_v iso-performance grid

The 2D surface isn't separable. AV-clean is way more sensitive to σ_a than σ_v at low-mid σ (σ_a=0.05 alone → 60.9%; σ_v=0.20 alone → 94.7%). AV-rawnoise is σ_a-invariant up to σ_a≈0.10 (93.8%), and the degradation there is mostly driven by σ_v. The (σ_a, σ_v) = (0.01, 0.00) cell of AV-rawnoise hits 95.96%, which is slightly higher than its (0, 0) cell — basically the model fits its trained noise distribution a bit better than clean. At (0.5, 0.8), AV-clean drops to 7.06% and AV-rawnoise to 17.73%.

## D1.5 — Iso-performance integration premium

At any target accuracy 50–85%, AV-clean takes 3.1–3.8× more σ_a than A-only (e.g. 0.0734 vs 0.0238 at the 50% target) and 2.0–2.7× more σ_v than V-only (e.g. 0.598 vs 0.298 at 50% target). The premium holds at every accuracy level we tested, so it's not a noise-floor artefact.

## D1.6 — Cross-trained sanity

A-only matches its recorded best_val_acc bit-exact (92.70%). V-only-fair matches within 0.06 pp. The plan originally asked "does A-only use video if you force-feed it?" — turns out the question doesn't really apply, because A-only's `WordResNet` has no video input pathway at all. The honest cross-modal evaluation is the D1.1 AV-trained row.

## D1.7 — Heterogeneous noise

σ_a is about 5× more damaging than σ_v at matched relative levels: σ_a=0.05 alone → 60.93%; σ_v=0.20 alone → 94.70%. The symmetric (σ_a + σ_v) trajectory tracks the σ_a-only trajectory until σ_a≥0.10 / σ_v≥0.40, after which the visual stream itself starts to degrade.

## Scripts

`analyze_av_deepdive.py` (D1.1, D1.6, D1.7), `eval_av_visual_noise.py` (D1.2–D1.4), `iso_perf_rendezvous.py` (D1.5), `build_d1_figures.py` (figures). Composed by `run_deepdive_tier0.py`.

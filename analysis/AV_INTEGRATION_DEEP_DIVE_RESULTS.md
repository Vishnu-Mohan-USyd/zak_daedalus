# AV deep-dive — what we found

Tier-0 inference battery plus the Tier-1 retrains. Same val throughout (sha `03c5a87a…`, N=5,244, 180 classes), `.eval()`, seed 0. CSVs live in `analysis/deepdive/`. Full mechanism story sits in `AV_INTEGRATION_DEEP_DIVE_SYNTHESIS.md`.

## The numbers worth caring about

Two channels basically carry the whole integration. Wv ch22 is +42.2 pp and ch12 is +37.7 pp (D3.5). Zero all 64 Wv channels and you get 0.84% — same number as the v_mid=0 cliff (D1.1).

That v_mid=0 cliff is a FC calibration issue. AV softmax under v=0 is 0.84%, but fit a fresh linear probe on the same penult and you pull 52.6% word and 67.8% viseme (D4.5, D2.5). So the features still hold the word — the trained FC just goes off the rails when the activation distribution shifts.

AV trades a bit of early-audio fidelity for fuseable features. A-only block1 word probe is 42.7%, AV a_mid is 28.0%. By block2 A-only is at 90.3% and AV is at 94.3% (D5.1). So it gives up some standalone discriminability up front and more than makes up for it downstream.

AV sits in its own geometric basin. Class-mean RDM Spearman ρ: AV–A 0.79, AV–V 0.74, A–V 0.47 (D4.3). Joint training pulls AV toward both A and V at the same time, so it ends up with its own structure instead of sitting halfway between them.

α is sharp. Trained α=5.20 is the inference peak; α=4 takes you to 94%, α=7 to 92%, α=10 to 68%. AV-rawnoise (α=6.03) is broader — α ∈ [5, 7] all stay ≥ 95.3% (D3.7).

## Direction 1 — noise iso-degradation

Detail in `AV_INTEGRATION_DEEPDIVE_D1.md`. Short version:

D1.1 — the 3×3 matrix. AV-trained: AV-full 95.67%, audio-only 0.84%, video-only 44.47%.

D1.4 — σ_a × σ_v grid isn't separable. AV-clean is about 5× more sensitive to σ_a than σ_v at low-mid σ; AV-rawnoise is σ_a-flat up to σ_a≈0.10.

D1.5 — AV soaks up 3.4× more σ_a than A-only and 2.7× more σ_v than V-only at iso-perf. Holds stable across 50/65/75/85% target accuracies.

D1.3 — one zeroed frame drops AV from 98.4% to 70.3%. That's the 3D-conv stem (T=5 kernel hits 5 positions, BatchNorm goes OOD), totally separate from the noise sweeps.

## Direction 2 — geometry

D2.1 / D2.2 — UMAP shows AV-full as a basin of its own (not halfway between A and V) (`D2_umap_3cond.png`, `D2_umap_joint_3model.png`).

D2.3 — RF importance for per-word AV rescue: word_len 0.32–0.51, n_vowels 0.20, viseme features 0.02–0.06. The plan predicted bilabials would dominate. They don't — total acoustic-temporal content beats per-phoneme viseme advantage. One catch: long words also have lower A-only baselines (more headroom), so word_len partly just indexes rescue opportunity.

D2.5 — three-condition viseme probe (5-fold): AV-full 92.0%, V-fair 87.4%, A-only 83.3%, AV_audio_zero 81.3%, AV_v_zero 67.8%. Joint training basically makes the AV penult more viseme-discriminative than even the V-only baseline.

## Direction 3 — mechanism (lesion + α-sweep, Tier-0)

D3.5 — per-channel Wv lesion (AV-clean): ch22 +42.2 pp, ch12 +37.7 pp, ch53 +9.3, ch40 +7.8. AV-rawnoise concentrates harder on the same channels: ch22 +83.5, ch12 +61.6. Audio-noise training pushes even more weight onto them.

D3.6 — tertile lesion: top-21 channels carry +94.5 pp (AV → 1.14%); mid-21 carry +42.8 pp; low-22 carry +1.3 pp. So 21 of 64 Wv channels do basically everything.

D3.7 — α inference sweep. AV-clean (trained α=5.2024): peak at the trained value. α=4 → 94.07%, α=7 → 91.65%, α=10 → 68.00%. AV-rawnoise (trained α=6.03): broader peak (α=5 → 95.3%, α=6 → 95.86%, α=7 → 94.5%).

D3.9 — Wa lesion sanity check: top Wa channels are 5–10× smaller than top Wv (ch10 +6.1, ch62 +4.75 vs Wv ch22 +42.2). Rules out the "Wv is just random noise" null — the visual stream is what's driving the gate.

Tier-1 cross-variant lives in `AV_INTEGRATION_TIER1_CROSS_VARIANT.md`. Quick version: mid-add ties mid-mult at 95.80% but doesn't get inhibition; late-fusion 95.50% with fully distributed channels; early-fusion 94.22% (input-scale mismatch).

## Direction 4 — cross-model interpretation

D4.1 — late ensemble (softmax-avg of A and V): 95.06%. AV-fused: 95.67%. So the gate beats decision-level fusion by just +0.61 pp. Which means most of the +3 pp gain over A-only just comes from joint training reshaping both unimodal paths; the gate adds the +0.61 pp on top.

D4.2 — layer-wise CKA (A_only × AV): block1 ↔ a_mid 0.976 (pre-gate audio basically identical), block1 ↔ block2 0.516 (downstream diverged), block2 ↔ block2 0.751.

D4.3 — RSA per-class (Spearman ρ on class-mean RDM): AV vs A 0.7862, AV vs V 0.7422, A vs V 0.4671.

D4.5 — 5-fold linear probe on penult: A_only 90.81%, V_fair 83.03%, AV_full 94.53%, AV_v_zero 52.63%, AV_audio_zero 67.39%. The 52.63% vs 0.84% gap under v=0 is the cleanest way to see this: the features still encode the class, the trained final layer just gives wrong answers once the activations shift.

## Direction 5 — layer-wise information flow

D5.1 — layer-wise word probe: A_only block1 42.7% → block2 90.3%; V_fair visual_gap 65.9% → block2 82.2%; AV a_mid 28.0% → v_mid 46.8% → gate_out 76.0% → block2 94.3%. The gate adds 29 pp in one layer (a_mid 28% → gate_out 76%); block2 then sharpens to 94.3%. Viseme probe at AV block2: 91.8%.

D5.2 — within-AV CKA: a_mid ↔ gate_out 0.980 (gate barely perturbs GAP-level audio), a_mid ↔ v_mid 0.871 (streams are already CKA-similar before fusion), gate_out ↔ block2 0.591 (heavy lifting at block2).

D5.3 — info plane (LOOSE): every site estimates 5.03 nats = log(180), the binned PCA-8 MI estimator is saturated. We flagged this going in; need KSG / InfoNCE on raw features to do a real layer comparison.

D5.5 — temporal saliency: zeroing frames 20–30 (400–600 ms, articulation peak) drops AV by 74.3 pp.

D5.6 — spatial saliency: peak at the lip-ROI centre (−6.16 pp from one centre patch). Cheek and chin patches near zero.

D5.8 — block-2 lesion: top channels each carry only +0.17 to +0.25 pp. Block-2 is a distributed readout, while the gate is sparse.

D5.9 — cross-model CKA (AV vs A_only): a_mid ↔ block1 0.976, v_mid ↔ block1 0.871, block2 ↔ block2 0.751.

D5.11 — per-block2-channel R² on v_mid: 107 of 128 channels (83.6%) have R² > 0.30 (median 0.51, max 0.96). So visual modulation is widespread at block2, but lesion sensitivity (D5.8) is flat across channels. Widespread but redundant.

D5.12 — RSA layer trajectory in AV: ρ_onset 0.004 (a_mid) → 0.046 (gate_out) → 0.202 (block2); ρ_viseme 0.008 → 0.020 → 0.151. Class structure shows up in two steps with a discrete jump at the gate. Magnitudes are small because cosine class-mean RDM on 29 samples/class is noisy, but the trend is what matters.

## Sanity audit

`val_idx` sha matches across all phases. A-only on A is bit-exact to its checkpoint (92.70%). V-fair within 0.06 pp. AV-full is 95.67%. The α=5.20 and α=6.03 overrides are bit-exact to their respective checkpoints. D3.6 all-Wv-zero (0.84%) matches the D1.1 v_mid=0 cell.

## What we skipped

D5.3 info plane saturated at log(180) — the proper fix is a KSG / InfoNCE estimator on raw features; out of Tier-0 scope.

D2.4 (case studies) and D2.6 (speaker / gender colorings) were skipped — low-cost follow-ups for the supplementary.

D5.6 spatial saliency ran at 6×6 grid (planned 11×11) to fit the Tier-0 budget; peak localization was preserved.

## Scripts

`run_deepdive_tier0.py` orchestrates everything. Phase scripts: `phase_a_deepdive.py` (D4.1/2/3/5 + cache), `analyze_av_deepdive.py` (D1.1/6/7), `eval_av_visual_noise.py` (D1.2/3/4), `iso_perf_rendezvous.py` (D1.5), `phase_c_lesions.py` (D3.5/6/7/8/9), `phase_d_saliency.py` (D5.4/5/6/7/8), `phase_e_geometry.py` (D2.x), `phase_f_flow.py` (D5.1/2/3/9/11/12). Activation cache at `processed/deepdive_act_cache.pt` (83 MB, 5 conditions).

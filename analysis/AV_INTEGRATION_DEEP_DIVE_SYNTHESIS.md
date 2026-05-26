# AV integration — putting it all together

So this is the AV ResNet (`AVWordResNet`, 95.80% val) probed against two matched unisensory baselines (`WordResNet` at 92.70%, `VOnlyFairWordResNet` at 86.50%). 39 inference-only sub-experiments across five directions, plus three Tier-1 architectural retrains. Same val set throughout (sha `03c5a87a…`, N=5244, 180 classes), `.eval()`, seed 0. Per-direction detail in the companion docs.

Integration is sparse and gated, sits at a multiplicative mid-block site, and falls apart once you go off-distribution. Two of the 64 gate channels carry at least 80 pp of the cross-modal gain. The gate is sharply tuned around α≈5.20. AV soaks up 3× more audio noise than A-only at iso-performance but breaks if you take video away — softmax drops to 0.84%. The penult features still support a 52.6% linear probe though, so the cliff is FC calibration; the features themselves still hold the info. Mid_mult, mid_add, late and early all reach 94–96% val, so the AV accuracy advantage just comes from having two input streams that carry complementary info. The gate is how mid_mult pulls that info out. Two things going on here that you can split apart: mid-fusion gating in general gives you the sparsity and the sharp α; multiplicative coupling on top of that gives you the inhibition.

---

## Methods

| Role | Checkpoint | Val acc | α | Params |
|---|---|---:|---:|---:|
| A-only | `audio_only_filtered.pt` | 92.70% | n/a | 291k |
| V-fair | `video_only_fair.pt` | 86.50% | n/a | 476k |
| AV-clean | `av_fused.pt` | 95.80% | 5.20 | 522k |
| AV-rawnoise | `av_fused_rawnoise.pt` | 95.84% | 6.03 | 522k |

`AVWordResNet` audio path is 1×80×99 log-mel → `audio_block1` → `a_mid` (64 ch). Visual path is 50×88×88 lip-ROI → 3D-conv `VisualEncoder` → `v_mid` (64 ch). They meet at the cross-modal gate:

```
a_fused = a_mid * (1 + α * σ(W_a * a_mid + W_v * v_mid))
```

After that you've got `audio_block2` (64→128, stride 2) → GAP → linear head. All inference is deterministic (seeds + `cudnn.deterministic`). The probes are 5-fold stratified LR with `StandardScaler`. CKA is the linear Kornblith 2019 form. RSA uses cosine class-mean RDMs (16,110 pairs). We confirmed bit-exact reproduction for all the anchor numbers. AV-clean reads as 95.80% (epoch-186 best); some Tier-0 inference CSVs report 95.67% from the final-weights checkpoint instead, that's a ~10-sample / 0.13 pp gap.

---

## Direction 1 — noise robustness

The 3×3 modality matrix: AV-trained gets 95.67% on AV, 44.47% on visual-only (audio zeroed), and just 0.84% on audio-only (v_mid zeroed). That 0.84% cell is what we keep calling the v_mid=0 cliff — when v_mid goes to zero the gate saturates and the audio values coming out are basically in a range `audio_block2` never saw during training. Full detail in `AV_INTEGRATION_DEEPDIVE_D1.md`.

At iso-performance (50–85% target), AV-clean takes 3.1–3.8× more σ_a than A-only and 2.0–2.7× more σ_v than V-fair. That premium stays stable across the whole operating range. The σ_a × σ_v surface isn't separable either: at low σ_a the σ_v effect is strong, at high σ_a it compresses — which is what you'd expect from a multiplicative gain. AV-rawnoise (trained with audio noise) has a broader σ_a plateau and holds up better at extreme σ_v than AV-clean (41% vs 32% at σ_v=0.80), so audio-noise training kinda transfers across noise channels.

Frame-drop is the brittleness side. A single zeroed frame drops AV from 98.4% to 70.3%; five frames takes it to 13.3%. The 3D-conv stem's (5,7,7) kernel plus frozen BatchNorm stats push sparse zero-frames into OOD; the Gaussian σ_v sweep preserves the per-frame intensity distribution and is recoverable. So treat the frame-drop curve and the noise sweeps as different things.

---

## Direction 2 — geometry

Joint UMAP across A-only, V-fair, and AV-full penultimates shows AV sitting in its own basin. RSA numbers: AV↔A 0.79, AV↔V 0.74, A↔V 0.47. Joint training pulls AV's class structure toward both unimodal structures at once. A three-condition viseme probe on the AV penult: AV-full 92.0%, V-fair 87.4%, A-only 83.3%, AV_audio_zero 81.3%, AV_v_zero 67.8%. So the AV penult ends up carrying more viseme info than V-fair's does — same kind of thing Ghazanfar & Schroeder (2006) saw in unisensory cortex.

RF feature importance for per-word AV rescue (D2.3) puts `word_len` first (0.32–0.51 across three rescue targets), `n_vowels` second (0.20–0.23), and viseme features below 0.06. That contradicts the plan's bilabial-dominance prediction. Two plausible explanations: longer words just give the model more time to integrate, and longer words also have lower A-only accuracy so there's more headroom to rescue. Can't separate them with what we have here; a re-analysis that regresses out A-only baseline first would do it, that's a one-script follow-up.

---

## Direction 3 — mechanism (Tier-0 + Tier-1)

Per-channel Wv lesion on AV-clean: ch22 is +42.16 pp, ch12 is +37.70 pp, then a long tail (ch53 +9.3, ch40 +7.8, etc.). The tertile lesion confirms 21 of 64 channels do basically the entire integration (top-21 drop = +94.5 pp, bottom-22 drop = +1.3 pp). AV-rawnoise concentrates even harder on the same two channels (ch22 +83.5, ch12 +61.6). Wa lesions are 5–10× smaller than Wv (top Wa is ch10 at +6.10), which rules out the "any gate perturbation just collapses AV" null.

α inference sweep on AV-clean (trained α=5.20): peak at the trained value, bell-shaped collapse outside α ∈ [4, 7] (α=4 → 94.07%, α=7 → 91.65%, α=10 → 68%). AV-rawnoise (trained 6.03) holds ≥95.3% across α ∈ [5, 7]. So audio-noise training widens the usable α range by about 3×.

Tier-1 retrained three more variants under identical conditions. mid_add ties mid_mult (95.80%), late drops 0.30 pp, early drops 1.58 pp. Cross-variant reads (full tables in `AV_INTEGRATION_TIER1_CROSS_VARIANT.md`):

Channel concentration is a mid-fusion thing — mid_mult's top-4 share is 68.7%, mid_add's is 42.3%, late spreads it all out (14.2%, max single channel +0.15 pp). The extreme winner-take-all pattern (top-2 ~80 pp) is multiplicative-specific.

Inhibitory regulation is multiplicative-only. mid_mult's mean|αg| goes 1.00 (AV-full) → 1.70 (audio-only) → 0.86 (video-only): so 70% louder under modality loss, with about 40% of residuals coming out negative. mid_add's mean|αg| stays at 0.61 across all three conditions, with residuals strictly positive because of how the math works. The additive form can't exhibit inhibition.

Sharp α tuning shows up in mid_add too (trained α=1.31, bell-shape, 0.59% at α=0, 0.48% at α=3). So sharp tuning holds for both gain-modulation variants.

Late ends up worse than A-only under heavy V noise (−24.1 pp at σ_v=1.6) — if you only combine the streams at the decision layer, you can't ignore one stream when it goes bad. Early's 1.58 pp penalty traces back to the ~200× input magnitude mismatch (audio mel std 4.46 vs video pixel std 0.11) at the input concat layer — it's the input scaling that hurts. Early integration on its own would be fine.

---

## Direction 4 — cross-model interpretation

Late ensemble (softmax avg of A-only and V-fair predictions) gets 95.06%. AV-fused gets 95.67%. So the gate only beats the best late ensemble by +0.61 pp, but α=0 collapses AV to 1.49% — meaning the gate is necessary even though its combining role looks small. Most of the AV gain over A-only just comes from joint training reshaping both unimodal paths, and the gate adds the +0.61 pp on top.

Layer-wise CKA (A_only × AV): `block1` ↔ `a_mid` = 0.976 (pre-gate audio basically identical), `block2` ↔ `block2` = 0.751 (post-gate diverged). RSA on class-mean cosine RDMs: AV vs A 0.786, AV vs V 0.742, A vs V 0.467 — joint training pulls both unimodal structures toward a shared basin.

The trained readout is calibrated to AV-full (D4.5): we ran a 5-fold linear probe on the AV penult features. AV_full: probe 94.53%, softmax 95.67%. AV_v_zero (audio only): probe 52.63%, softmax 0.84% — that's a 51.8 pp recovery from a fresh probe. AV_audio_zero (video only): probe 67.39%, softmax 44.47% — 22.9 pp recovery. The viseme probe at AV_v_zero (D2.5) reaches 67.8%, so the same pattern shows up across three independent probe targets. The class info is sitting right there in the features; the trained final layer was just calibrated against AV-full activations and gives wrong answers when v_mid is removed. Tier-1 shows the cliff only happens when V feeds gain into a shared A → classifier pathway (late hits 40.5%, early hits 82.9% under v_zero — neither one has the cliff).

If you extend this to the brain: sudden modality loss would probably produce a sharp accuracy drop followed by rapid readout re-calibration on a seconds timescale, with the underlying representation keeping most of the info throughout.

---

## Direction 5 — layer-wise information flow

Layer-wise word probe:

| Site | A_only | V_fair | AV_full |
|---|---:|---:|---:|
| block1 / a_mid_gap | 42.7% | n/a | 28.0% |
| visual_gap / v_mid_gap | n/a | 65.9% | 46.8% |
| gate_out_gap | n/a | n/a | 76.0% |
| block2_gap | 90.3% | 82.2% | 94.3% |

AV's `a_mid` decodes word at 28% vs A-only's `block1` at 43% — so joint training has delayed word-discriminating structure along the audio path. By block2 though, AV reaches 94.3%, highest of the three. CKA(A_only.block1, AV.a_mid) = 0.976 — so high CKA, but with a 14.7 pp probe gap. The overall geometry is similar, but joint training has rotated the features into directions the gate and `audio_block2` can use, even when a flat linear probe can't see them.

Integration adds 30 pp in one step at the gate (a_mid 28% → gate_out 76%) and another 18 pp at block2. The viseme probe traces the same staircase (V_fair visual_gap 79% → AV block2 91.8%); onset (10-way) peaks at AV block2 90.8%. Per-block2-channel R² on v_mid: 107 of 128 channels (83.6%) have R² > 0.30. Per-channel block2 lesion impact is small (top-8 each +0.17 to +0.25 pp); Spearman ρ between R²_v_mid and lesion Δpp is +0.04 (n.s.). So block2 is widely visually modulated but redundantly encoded. Sparse at the gate, distributed at the readout — kinda mirrors the pSTS → STG hierarchy.

Temporal saliency peaks at frames [20, 30) (400–600 ms post-onset, right around the articulation peak, −74.3 pp). Spatial saliency peaks at the center-lip patch (−6.16 pp); edges are about 10× less informative. GradCAM agrees. So basically the model is reading lips. RSA layer trajectory (Spearman ρ on cosine RDM vs categorical RDM): a_mid ρ_onset 0.004 → gate_out 0.046 → block2 0.202; viseme 0.008 → 0.020 → 0.151. Numbers are small because cosine RDMs on ~29 samples/class average out the within-class variation, but the trend is what we care about. The info-plane stuff (D5.3) saturated at log(180) ≈ 5.03 nats at every site — we flagged this as LOOSE going in, would need KSG / InfoNCE on raw features to do it properly.

---

## Source vs mechanism

Tier-1 splits the question into two — where the AV accuracy comes from, and what mid_mult specifically does to deliver it.

| signature | mid_mult | mid_add | late | early |
|---|:---:|:---:|:---:|:---:|
| ≥95% ceiling | ✓ 95.80 | ✓ 95.80 | ✓ 95.50 | ✗ 94.22 |
| σ_a robustness at σ_a ∈ [0.02, 0.5] | ✓ 60.9 | ✓ 56.0 | ✗ 28.0 | ✗ 22.6 |
| Sharp α tuning | ✓ | ✓ | n/a | n/a |
| Calibration cliff (v_mid=0 → ~0.8%) | ✓ | ✓ | ✗ 40.5 | ✗ 82.9 |
| Channel concentration (top-4 ≥ 42%) | ✓ 68.7 | ✓ 42.3 | ✗ 14.2 | n/a |
| Extreme sparsity (top-2 ~80 pp) | ✓ | ✗ | ✗ | n/a |
| Gate-magnitude compensation under modality loss | ✓ +70% | ✗ flat | n/a | n/a |
| Sign-flipped inhibitory residuals | ✓ 40% neg | ✗ 0% (math won't allow it) | n/a | n/a |
| Rich viseme code at penult (> V-fair 87.2) | ✓ 91.8 | ✓ 92.4 | ✓ 89.7 | ✗ 83.4 |
| AV gain positive under heavy V noise | ✓ | ✓ | ✗ (−24 pp at σ_v=1.6) | ✓ |

These signatures break into three groups.

The 94–96% ceiling shows up in every architecture. That just comes from the two streams carrying complementary information about the input — audio confusables like TEN/TURN are visually distinguishable; visual confusables like BAT/PAT/MAT are audio-distinguishable. You'll see this anywhere the streams get joined.

σ_a robustness, channel concentration, sharp α tuning, the v_mid=0 cliff, and the rich viseme code at penult all show up in both mid_mult and mid_add — but not in late or early. They come from having gain modulation at a mid-block site, same kind of thing Beauchamp 2008 saw in pSTS (sparse multisensory population) and Murray et al. 2016 saw in joint-training-shaped unisensory cortex.

Then there's a multiplicative-only batch that can't show up in mid_add at all. The gate gets louder under modality loss (multiplication needs a bigger α·σ(·) to make up for v_mid going to zero) and residuals can flip sign (`a_fused − a_mid = a_mid · α · g` inherits sign from `a_mid`). Lakatos 2009 saw similar inhibition from cross-modal phase reset; Arnal & Giraud 2012 saw it from predictive-coding error suppression. So the cleanest biology-testable difference between the architectures: multiplicative predicts +70% louder gate under modality loss and mixed-sign residuals; additive predicts invariant magnitude and strictly excitatory residuals.

Two extra failure modes sit outside the three-group pattern. Late ends up worse than A-only under heavy V noise because combining only at the decision layer means you can't ignore a corrupted stream. Early takes a 1.58 pp ceiling penalty from un-normalised input concat — once again, the input scales are what hurt, early integration itself isn't the problem.

---

## Limitations and follow-ups

D5.3 info-plane saturated at log(180); KSG / InfoNCE on raw features is what would actually work. D5.6 spatial saliency ran at 6×6 grid (planned 11×11) to fit Tier-0 budget; peak localisation was preserved though. D2.3 `word_len` finding needs a re-analysis that controls for A-only baseline (long words have lower baselines, so word_len partly indexes rescue headroom). E3 McGurk used word-distinct rather than phoneme-distinct conflict pairs. Tier-2 follow-ups still on the table: BN-normalised early fusion (predict ~95.5%, would tell us if the ceiling penalty comes from input scales or from early integration itself), recovery-dynamics on v_mid=0 penult (how many samples does a re-fit probe need to recover from 0.8% to ≥50%?), and the recurrent variant (could show pre-articulation visual saliency). Things this study doesn't cover: generalisation to natural AV speech, biological plausibility of the multiplicative gate itself (real pSTS uses long-range cortical projections, not an explicit `(1 + α·σ(·))` modulation — the gate is a computational abstraction), and developmental dynamics. The α-evolution data (5.20 clean → 5.29 mel-noisy → 6.03 raw-noisy) hints the model adapts gain with audio-noise experience, but it's purely observational.

---

Companion docs: `AV_INTEGRATION_DEEP_DIVE_PLAN.md` (protocol), `AV_INTEGRATION_DEEP_DIVE_RESULTS.md` (per-direction raw results), `AV_INTEGRATION_DEEPDIVE_D1.md` (Direction 1 detail), `AV_INTEGRATION_TIER1_CROSS_VARIANT.md` (cross-variant tables), `MSI_RESULTS.md` (the 11-experiment MSI battery).

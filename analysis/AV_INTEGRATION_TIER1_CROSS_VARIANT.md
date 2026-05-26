# Tier-1 cross-variant comparison

Same val set (sha `03c5a87a…`). Raw CSVs in `analysis/deepdive/`.

## Models

| variant | flavour | params | best val |
|---|---|---:|---:|
| AV_fused | mid_mult | 522,509 | 95.80% |
| D3_2_late | late | 525,836 | 95.50% |
| D3_10_add | mid_add | 522,509 | 95.80% |
| D3_1_early | early | 530,996 | 94.22% |

## 1. AV gain across audio noise (σ_v = 0)

AV gain here is just AV_acc minus that variant's own A-only path (`model(audio, video=None)`).

| variant | σ_a=0 | σ_a=0.05 | σ_a=0.5 |
|---|---|---|---|
| AV_fused | 95.7% (+94.8) | 60.9% (+60.5) | 12.1% (+11.6) |
| D3_2_late | 95.4% (+54.9) | 28.0% (+18.4) | 7.8% (+5.2) |
| D3_10_add | 95.8% (+94.9) | 56.0% (+55.4) | 15.0% (+14.5) |
| D3_1_early | 93.9% (+11.1) | 22.6% (+2.7) | 3.5% (+0.6) |

`mid_mult` and `mid_add` track each other pretty closely. `late` keeps less AV gain. `early` basically collapses to A-only because the audio inputs are ~200× bigger than video pixels at the input layer, so the V stream just gets drowned out. Full sweep at σ_v=0 and σ_a=0 in `analysis/deepdive/tier1_cross_variant_*.csv`.

## 2. V-side channel lesions

Site: zero one channel of `gate.Wv` output (mid_mult / mid_add, 64 ch) or the V-branch GAP output (late, 64-d). Early has no isolated V stream so it's not in the table.

| variant | top-2 channels (pp) | top-4 share of positive impact | max channel impact |
|---|---|---:|---:|
| AV_fused | ch22 +42.16, ch12 +37.70 | 68.7% | +42.16 |
| D3_10_add | ch37 +22.58, ch27 +20.08 | 42.3% | +22.58 |
| D3_2_late | ch1 +0.15, ch37 +0.15 | 14.2% | +0.15 |

Mid-mult and mid-add concentrate; late spreads it all out. The specific top channels differ between mid_mult (ch22/ch12) and mid_add (ch37/ch27), so the exact neurons differ across architectures but the concentration pattern holds in both.

## 3. α-sweep, mid_add variant

Trained α = 1.31, acc 95.77%. α=0 → 0.59%, α=1.0 → 8.18%, α=2.0 → 1.96%, α=5.0 → 0.44%. So mid_add is even sharper than mid_mult's α=5.20 peak (mid_mult still scores 68% at α=10). The additive form basically breaks faster when you push α away from where it was trained.

## 4. Gate inhibition probe

|residual| = |a_fused − a_mid|. For mid_mult that's `a_mid · α·g`; for mid_add it's just `α·g` directly.

| variant | condition | mean\|αg\| | frac(g>0.5) | mean\|res\| | frac(res>0) |
|---|---|---:|---:|---:|---:|
| AV_fused | AV_full | 1.00 | 0.16 | 0.27 | 0.60 |
| AV_fused | audio_only | 1.70 | 0.26 | 0.54 | 0.60 |
| AV_fused | video_only | 0.86 | 0.12 | 0.25 | 0.43 |
| D3_10_add | AV_full | 0.61 | 0.45 | 0.61 | 1.00 |
| D3_10_add | audio_only | 0.61 | 0.40 | 0.61 | 1.00 |
| D3_10_add | video_only | 0.61 | 0.46 | 0.62 | 1.00 |

Mid-mult's gate is 70% louder under audio-only than under AV-full — it works harder when audio is alone. About 40% of residuals come out negative, which is the inhibition. Mid-add's residual is strictly positive (α > 0, g ∈ (0,1) so it can't go negative) and the magnitude doesn't budge across conditions at all.

## 5. Viseme decodability (5-fold LR on penult)

| variant | acc | balanced acc |
|---|---:|---:|
| AV_fused (mid_mult) | 91.83% ± 0.26 | 89.57% ± 1.32 |
| D3_10_add (mid_add) | 92.39% ± 0.71 | 90.30% ± 0.77 |
| D3_2_late | 89.66% ± 0.26 | 86.85% ± 0.42 |
| D3_1_early | 83.39% ± 1.18 | 81.12% ± 2.88 |

Mid-fusion (either form) beats late, late beats early. Joint training sharpens viseme structure most when the fusion happens mid-block.

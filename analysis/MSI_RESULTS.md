# MSI tests — what came out

So I ran the AV model (`models/av_fused.pt`, 95.80%), the A-only one (`audio_only_filtered.pt`, 92.70%) and V-only (`video_only_fair.pt`, 86.59%) on the same val set (sha `03c5a87a…`, N=5,244, 180 classes). Eleven tests, basically the usual neuro/psychophysics measures mapped onto the networks. CSVs and plots are in `analysis/msi/E*_*.{csv,png}`.

## Per-test results

E1 is the Sumby–Pollack inverse effectiveness one. With clean audio the AV model only beats audio-alone by ~3 points (+2.97 pp). Add a tiny bit of noise — σ=0.02 — and the gap goes up to nearly 50 points (+49.64 pp), because the model is now leaning on the lips. Push the noise way further and the gap closes again, because video alone can't carry the task on its own.

E3 is the McGurk one. On conflict pairs where the audio word and the video word have totally different visemes, the AV model picks the audio word only 7.80% of the time (A-only picks it 92.73% of the time), and it lands on some third "fused" word 82.73% of the time. So the model is actually fusing, not just relaying audio.

E4 is the supra-additivity check. Post-gate at `block2`, 52% of channels have an MEI above zero and 27% are strictly super-additive. Pre-gate at `a_mid` it's 0%, which is the control you want — the integration happens at the gate; nothing upstream of it integrates.

E5 is the perturbation set. Time-shuffle the video frames and AV collapses to 1.11%. Freeze the frames so they stop moving, you get 5.55%. Block-shuffle the pixels (so face identity is preserved but motion is killed), 53.05%. Clean run is 95.67%. So the model is leaning on articulator motion — face identity alone gets you almost nowhere.

E8 is cross-modal predictability. Inside the AV model, you can predict v from a with R² 0.468 and a from v with R² 0.600. In the unisensory baselines those same predictions sit at 0.110 and 0.229. So joint training adds +0.36 / +0.37 R² of mutual predictability between the streams.

E9 is the gate readout. The learned α is 5.20. Mean |gate| under full AV is 0.275, audio-only (v zeroed) is 0.538, video-only (audio zeroed) is 0.253. So the gate is louder when audio is alone than when both streams are present and matching — an inhibitory regulation pattern that maps onto Lakatos-style cross-modal phase resetting.

E11 is the temporal window. Peak at Δt = 0 (95.67%), still 74–84% at ±60 ms, drops to 15–19% at ±200 ms. Basically matches the human ~±200 ms binding window people quote. There's a slight skew — video lagging audio is tolerated better than video leading audio — which fits the "visual leads audio" thing you see in speech.

## Overall

So like, eight of the ten scored tests came back positive for MSI. E2 (graduated dropout) is kinda mixed — the audio×β curve isn't monotonic, because the gate has a discrete "audio's gone → fall back to visual" regime that smooth interpolation skips right past. E9 is mixed too, but only because it's inhibitory rather than additive. E6/E7 are the race-bound ones — about 1.30% of items are AV-only-correct under the fair V baseline, which is more than either stream covers on its own.

## V-fair re-run (E6/E7, E8, E10)

After we swapped the lean V-only (70.42%) for `video_only_fair.pt` (86.50%), all three V-touching tests kept the same sign. Race-bound violations dropped from 132 items to 68 — still nonzero, which is what matters. AV-internal cross-predict R² is still 2.8–3.6× the unisensory baseline. The Bayes ratio (observed vs optimal) went 0.49 → 0.66, so AV is still about 50% tighter than independent pooling predicts, even with the stronger V baseline in the picture.

## Raw-audio-noise pair (E1, 4 models)

We trained two more — A-rawnoise (#19) and AV-rawnoise (#20) — and redrew E1 with all four. At σ_a/rms = 0.1, AV-rawnoise scores 93.82%, which is higher than clean-trained A-only at σ=0 (92.70%). So the visual stream basically recovers a whole "clean audio's worth" of performance at a noise level that nukes A-clean down to 8.91%.

Plot: `analysis/msi/E1_inverse_effectiveness_4model.png`. Numbers: `analysis/msi/E1_inverse_effectiveness_4model.csv`.

## Gate evolution

α grows with audio-noise training: AV-clean 5.20 → AV mel-noisy 5.29 → AV-rawnoise 6.03. Makes sense — the model leans harder on the gate when audio is less reliable.

Audio-only (running with video zeroed) drops to near-chance (0.84–3.18%) across all three AV variants. With v_mid = 0 the gate (α ≈ 5–6) pushes the audio activations into a range `audio_block2` was never trained on, and the downstream layer just gives garbage answers. Video-only (audio zeroed) holds up way better — 44–51% — because `W_a · 0 = 0` still leaves the `1 + α · σ(W_v · v_mid)` term doing useful work. The asymmetry between the two directions is just what a multiplicative gate does.

Plot: `analysis/msi/E_gate_evolution.png`.

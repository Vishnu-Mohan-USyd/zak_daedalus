# AV integration — the five questions

180-word spoken-word recognition. Held-out val set is 5,244 clips. A-only sits at 92.7%, V-only at 86.5%, and the mid-multiplicative AV model at 95.8%. AV works by multiplicatively gating the audio with the visual:

    a_out = a_in * (1 + α * σ(W_a * a_in + W_v * v))

α is a learned scalar that ends up at 5.20. W_v projects the visual features into 64 channels that get added in to the audio.

For the architecture comparison there are three more AV variants — mid-additive at 95.8%, late-fusion at 95.5%, early-fusion at 94.2%.

The biology bits below are predictions from the model. None of it was measured in a brain.

## Q1 — How does it do under noise?

Clean AV beats A-only by just +3 pp. Add a bit of audio noise and that gap blows up to ~50 pp — the inverse effectiveness pattern Sumby and Pollack saw way back. Push noise to extreme levels and the gap closes again, because video alone can't carry the task.

If you set a target accuracy (50–85%) and ask how much noise each model can soak up to hit it, AV handles 3.4× more audio noise than A-only and 2.7× more video noise than V-only. Audio noise is also about 5× more damaging than video noise at matched levels. So basically the network defaults to audio when audio is clean, and starts leaning on video as audio dies.

Weird edge case — drop a single video frame and AV collapses from 98% to 70%. This one's an OOD artifact. The 3D-conv's T=5 temporal kernel smears the zero across 5 output positions, and BatchNorm never saw zero frames during training, so the activations end up off-distribution. So treat the frame-drop curve separately from the noise sweeps.

## Q2 — Representational geometry

Pairwise Spearman ρ between class-distance matrices: AV–A 0.79, AV–V 0.74, A–V 0.47. So AV is closer to each unimodal geometry than the two unimodals are to each other. Joint training pulled AV toward both, and it ends up sitting in its own basin in feature space.

Linear-probe viseme decodability off the last-layer features: A-only 83%, V-only 87%, AV 92%. So joint training with audio sharpened the viseme structure beyond what V-only training did. Ghazanfar & Schroeder saw the same kind of thing in unisensory cortex — sensory cortex carries traces of the other modality when both streams were around during learning.

Word length is the strongest predictor of per-word AV rescue (RF importance 0.32–0.51). One catch though: long words also have lower A-only baselines, so this might just be inverse effectiveness at the item level instead of a real length effect. A matched-length follow-up would settle it. Haven't run it yet.

## Q3 — Architecture

| Variant | Val acc |
|---|---:|
| Mid-multiplicative | 95.80% |
| Mid-additive | 95.80% |
| Late-fusion | 95.50% |
| Early-fusion | 94.22% |

Mid and late tie within seed noise. Early underperforms because at the input layer audio (std 4.5) and video pixels (std 0.1) differ by ~200× in magnitude, so the first conv either drowns the video out or has to amplify it hugely. Just an input-scale issue.

Mid-mult and mid-add tie on accuracy but get there through different mechanisms. Both have sparse channel coding (top-2 of 64 channels carry ~50–70% of the lesion impact) and both peak around their learned α. The difference: only mid-mult shows inhibition — 40% of its gate residuals are negative, so the visual stream can suppress audio channels. The mid-mult gate is also ~70% louder under audio-only than under matched AV, so it works harder when audio is alone. Mid-add can't do either of those — its residual is just α·g with α > 0 and g ∈ (0,1), strictly positive by construction.

Late fusion's channel lesions are completely spread out (no single channel moves accuracy by more than ±0.15 pp). It just averages the two streams at the decision layer.

Lesion sensitivity in mid-mult is concentrated: zeroing channel 22 drops AV to 53.5%, channel 12 to 58.0%, and the remaining 62 channels combined only add ~15 pp. α=5.20 sits on a sharp peak — α=10 takes you all the way down to 68%.

Skipped the recurrent variant — the temporal binding window from the MSI battery drops to half-max around ±60–100 ms, which is well within what feed-forward integration over the clip can handle.

## Q4 — Does joint training actually change the audio path?

Yeah — and the final classifier was trained on AV inputs, so it just gives wrong answers when video is taken away.

CKA between matched layers, A-only vs AV: pre-gate is 0.98 (basically identical), post-gate drops to 0.75 (restructured). So joint training reshaped the audio downstream of the gate and left the early audio path alone.

There's a tradeoff too. AV's pre-gate audio only decodes 28% of word identity on its own, vs A-only's 43% at the same depth. By post-gate AV catches up (94% vs 90%). So the AV audio path gives up some standalone discriminability up front in exchange for stuff that combines well with video later.

Zeroing the visual input collapses AV from 95.8% to 0.84% — below chance (0.56%). But fit a fresh linear classifier on the same penultimate features (with v=0) and you recover 52.6% word accuracy and 67.8% viseme accuracy. So the features themselves still hold the word identity — the trained FC was just calibrated to AV-full activations and gives wrong answers once the activation distribution shifts.

Extending this to the brain: a brain trained jointly on AV, under sudden visual occlusion, would probably show a sharp accuracy drop followed by a seconds-to-minutes recovery as the decision regions re-calibrate to audio-only patterns. The upstream features stay intact; only the readout has to adapt. And this only happens in mid-fusion — late fusion doesn't break under v=0.

## Q5 — Information flow layer by layer

| Layer | Word decodability |
|---|---:|
| Pre-integration audio | 28% |
| Visual features | 47% |
| Post-integration | 76% |
| Post-block-2 audio | 94% |

The integration step alone adds 29 pp — the biggest single jump anywhere in the network. The post-integration block sharpens further to 94%.

Visual contribution is concentrated in both time and space. Peak is frames 20–30 out of 50 (400–600 ms in, right around the articulation peak); zeroing that 200 ms window costs AV 74 pp. Spatially, the center-mouth patch dominates; cheek and chin are near zero.

The gate itself is sparse — 2 of 64 channels carry +80 pp combined (Gini 0.84). The post-integration block is distributed — no single channel moves accuracy by more than 0.25 pp (Gini 0.38). 84% of block-2 channels are linearly predictable from visual features, but how strongly a channel is visually modulated doesn't predict whether lesioning it hurts (r=0.04). So integration is happening at the gate, and block-2 is a redundant readout on top.

## Where it lands

The +3 pp clean gain over A-only comes from having both streams around during training. Any reasonable AV architecture captures it — mid-mult, mid-add and late fusion all hit 95.5–95.8%.

Mid-multiplicative is the interesting one because of how it delivers the accuracy: sparse cross-modal gating, inhibition, context-dependent gain. Mid-additive doesn't do the inhibition; late fusion doesn't do the sparse channels. If biology shows the inhibitory + context-dependent pattern (and Lakatos and Arnal-Giraud have seen things like this), then mid-multiplicative would be the candidate implementation.

All the numbers are in `analysis/deepdive/*.csv`. Full writeup in `AV_INTEGRATION_DEEP_DIVE_SYNTHESIS.md`. Cross-variant stuff in `AV_INTEGRATION_TIER1_CROSS_VARIANT.md`.

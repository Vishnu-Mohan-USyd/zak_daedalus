# Daedalus — audiovisual spoken-word recognition and how the two senses combine

This repository trains small neural networks to recognise **spoken words from audio and video
of the speaker's face**, and then spends most of its effort on a single scientific question:
**when you give a network both an ear and an eye, how does it actually combine them?** The task
is a stand-in for human audiovisual speech perception, and the analyses are framed as testable
*predictions* about biological multisensory integration — none of it was measured in a real brain.

If you are new here, read this file top to bottom; it is the single source of truth. Every other
document that used to live in `analysis/` has been folded into the sections below. The raw numbers
behind every claim are in `analysis/**/*.csv`; the figures are in `analysis/msi/*.svg` and
`analysis/deepdive/*.svg`.

---

## Contents

1. [The task and the data](#1--the-task-and-the-data)
2. [The models](#2--the-models)
3. [Quickstart](#3--quickstart)
4. [Headline results](#4--headline-results)
5. [The full findings](#5--the-full-findings)
6. [Tests and validation](#6--tests-and-validation)
7. [Repository layout — where everything lives](#7--repository-layout--where-everything-lives)
8. [Requirements](#8--requirements)
9. [Reproducibility notes](#9--reproducibility-notes)
10. [Caveats and limitations](#10--caveats-and-limitations)
11. [Project status and handoff](#11--project-status-and-handoff)
12. [Glossary](#12--glossary)

---

## 1 · The task and the data

**The task.** Classify a short clip into one of **180 spoken words** (a 180-way classification
problem). Every clip has two aligned streams: an **audio** recording of the word and a **video**
of the speaker's mouth. There are **12 speakers**. Performance is reported as **top-1 accuracy** —
the fraction of held-out clips where the network's single highest-scoring guess is the correct word.
Chance is 1/180 ≈ 0.56%.

**The split.** All numbers in this repo are on one fixed, held-out **validation set of 5,244 clips**
(content hash `03c5a87a`, seed 0). Pinning the exact split matters: it makes every experiment
below directly comparable and bit-for-bit reproducible.

**Where the raw data goes** (none of it is committed — it is large and gitignored):

- Audio WAVs: `~/Downloads/audio/<speaker>/`
- Lip videos: `data/visual/video_data/<speaker>/`
- Filenames look like `speaker-02_grp1-item7_TEXT-C.wav` — speaker, group, item, and the word.
- `preprocess.py` turns these into a **log-mel cache** (audio) and a **video memmap** under
  `processed/`. Audio becomes a `1 × 80 × 99` log-mel spectrogram; video becomes a `50 × 88 × 88`
  stack of grayscale lip-ROI frames (50 frames, 88×88 pixels).

Audio and video clips are **not pre-aligned on disk**; `paired_dataset.py` pairs them sequentially
within each speaker/group. `dataset_raw_noisy.py` is the same dataset with on-the-fly audio-noise
augmentation for the noise-robustness experiments.

---

## 2 · The models

Everything is plain PyTorch — no Lightning, no Hydra, no Hugging Face. The shared backbone is a
small **ResNet** (`ResBlock` lives in `train.py`, `WordResNet` is the audio body).

### The three core networks

| Network | Class / file | Clean val acc | Notes |
|---|---|---:|---|
| **Audio-only** | `WordResNet` (`train.py`) → `models/audio_only_filtered.pt` | **92.70%** | log-mel → ResNet → word |
| **Video-only (fair)** | `VOnlyFairWordResNet` (`model_v_only_fair.py`) → `models/video_only_fair.pt` | **86.50%** | lip video → 3D-conv → ResNet → word |
| **Audiovisual** | `AVWordResNet` (`model_av.py`) → `models/av_fused.pt` | **95.80%** | the two streams joined by a gate (below) |

So on clean audio, **audio is the stronger single sense** (92.70% vs 86.50%), and adding video on
top buys **about +3 percentage points** (95.80%). That +3pp headline hides a much larger effect
under noise — see [Results](#4--headline-results).

*(Accuracy convention: `95.80%` is the best-epoch value logged during training under bf16 mixed
precision; the evaluation harness re-runs everything in **eager fp32**, where the same checkpoint
scores `95.67%`. The ~0.13pp gap is precision, not a different model. All analysis numbers below are
fp32 unless noted. See [Reproducibility](#9--reproducibility-notes).)*

### How audio and video meet — two fusion families

This repo contains **two different ways** of combining the streams, studied in two phases of work.

**(a) Mid-block multiplicative gate — `AVWordResNet` (`model_av.py`), the primary model.**
The visual stream produces a **per-channel gain** on the audio features, partway through the audio
network (between the two audio ResNet blocks):

```
a_fused = a_mid · (1 + α · sigmoid(W_a · a_mid + W_v · v_mid))
```

- `a_mid` is the 64-channel audio feature map after the first audio block.
- `v_mid` is a 64-channel feature map from a small 3D-conv lip encoder (`VisualEncoder`).
- `W_a`, `W_v` are 1×1 convolutions; `α` is a single learned scalar that trains to **≈ 5.20**.
- Read it as: *vision multiplicatively modulates the audio channels.* When `v_mid = 0` the gate
  term collapses and the audio passes through (nearly) unchanged.

This is the model behind almost all of the mechanism analysis, because the multiplication gives it
properties the other variants lack (sparse channel coding, inhibition, context-dependent gain).

**(b) Late-fusion reliability gate — `AVLateFusionReliabilityWordResNet` (`model_av_latefusion.py`).**
Here the two streams stay fully **independent** all the way to their own word predictions, then a
learned gate mixes the two prediction vectors per clip:

```
logit_a = audio_fc(audio_penult)         # a pure audio word-guess
logit_v = visual_fc(visual_penult)       # a pure, independent video word-guess
[w_a, w_v] = softmax(rel_gate(...))      # per-clip reliability weights (sum to 1)
logits = w_a · logit_a + w_v · logit_v
```

The gate reads each head's **confidence** (entropy, top probability, top-1-minus-top-2 margin) plus
the two penultimate feature vectors, and learns to lean on audio when audio is clean and on video
when audio is noisy. This model was built specifically to study the *reliability-weighting* view of
integration and a subtle failure of it (the "d′" investigation in
[Section 5.7](#57--the-late-fusion-d-investigation-and-the-biology-verdict)).

**(c) Architecture-comparison variants** (`model_av_additive.py`, `model_av_early.py`,
`model_av_late.py`, `model_av_recurrent.py`): mid-**additive** fusion, **early** (input-level)
fusion, plain **late** fusion, and a **recurrent** (GRU-over-time) variant. These exist so the
mechanism findings can be attributed to *where* and *how* fusion happens rather than to fusion in
general.

---

## 3 · Quickstart

```bash
# 0. install (see §8 for the full list)
pip install "torch>=2.0" numpy scipy scikit-learn==1.8.0 matplotlib Pillow   # + ffmpeg on PATH

# 1. build the mel cache and video memmap from raw clips
python preprocess.py

# 2. train the three core networks (~1–3 h each on one modern GPU)
python train.py                 # audio-only baseline    -> models/audio_only_filtered.pt
python train_v_only_fair.py     # video-only (fair)       -> models/video_only_fair.pt
python train_av.py              # audiovisual (mid-gate)  -> models/av_fused.pt

# 3. run the analysis battery (inference-only, uses the checkpoints above)
python run_deepdive_tier0.py    # the mechanism deep-dive
python analyze_av_msi.py        # the 11-test multisensory-integration battery

# 4. reproduce the figures
bash _run_figs.sh               # multisensory-integration and validation figures
```

To get oriented by reading code, start with `model_av.py` (the network and the gate), then
`train_av.py` (the training loop), then `paired_dataset.py` (how clips are paired). The `examples/`
folder has six short, self-contained scripts meant to be run in order — load a checkpoint and print
its accuracy, plot one sample's mel + lip frames, watch AV rescue an audio-only mistake, extract
features, linear-probe them, and visualise the gate on one clip.

---

## 4 · Headline results

- **Audio is the stronger sense; video adds a little on clean speech and a lot under noise.**
  Clean: AV beats audio-only by just **+2.97pp**. Add a small amount of audio noise (σ = 0.02) and
  the AV advantage balloons to **+49.6pp**, because the network switches to leaning on the lips.
  Push noise to extremes and the gap closes again (video alone can't carry a 180-way task). This is
  **inverse effectiveness** — the textbook finding that combining senses helps most when the
  dominant sense is weak-but-not-useless (Sumby & Pollack 1954).

- **The fusion is sparse, multiplicative, and localised to lip motion.** Only **2 of the gate's
  64 channels** carry most of the cross-modal benefit; knocking out either one costs ~40pp. The
  network reads the **centre of the mouth** at the **articulation peak** (video frames 20–30, ~400–
  600 ms in); blanking that 200 ms window costs 74pp. Vision **modulates** audio, not the reverse
  (the top visual gate channel is worth ~7× the top audio one).

- **The AV network is not a better audio recogniser with an eye bolted on.** Joint training
  *reorganised* the audio pathway to lean on the gate: the AV model's mid-level audio features
  decode words *worse* than the dedicated audio model at the same depth (28% vs 43%). The genuine
  improvement is the **fusion gain itself**, delivered through the gate.

- **Where you fuse matters.** Mid-fusion (multiplicative or additive) reaches 95.80%; late fusion
  95.50%; early fusion only 94.22%. The differences explode under noise: at σ_a = 0.05, mid-mult
  holds **60.9%** while late drops to 28.0% and early to 22.6%.

- **"Fused beats the best single sense" has a real, biological exception.** In one hard regime
  (clean audio + a *visually confusable* face) the late-fusion model scores *below* the better
  single sense — and a full investigation (plus three primary sources) shows this is **correct
  biology**, not a bug: humans also fail to beat their best sense under matched conditions. See
  [Section 5.7](#57--the-late-fusion-d-investigation-and-the-biology-verdict).

---

## 5 · The full findings

The scientific work happened in three campaigns: a **mechanism deep-dive** (how the mid-mult gate
works), an **11-test multisensory-integration (MSI) battery** (neuroscience/psychophysics measures
mapped onto the network), and a **19-question modality-integration study** (a structured, evidence-
only Q&A, independently validated). Plus the standalone **late-fusion d′ investigation**. This
section is the readable synthesis; every number is reproducible from `analysis/**/*.csv`.

### 5.1 · Accuracy and the shape of the AV advantage

Audio-only **92.70%**, video-only (fair) **86.50%**, AV **95.67%** (fp32). AV beats audio by
**+2.97pp** and video by **+9.17pp**, and it beats the best 50/50 average of the two specialists by
**+0.61pp** — so the gate does a little more than just averaging two opinions. Per word: of the 261
words audio gets wrong, AV rescues them and gets them right, at the cost of 105 words that go the
other way, for a **net +156 words**.

### 5.2 · Inverse effectiveness (the core MSI signature)

The AV−audio gap goes from **+2.97pp** (clean) to **+49.6pp** at σ_a = 0.02, then shrinks at heavy
noise. The same shape appears on the video side: the AV−video gap peaks at **+57.2pp** at σ_v = 0.40.
Framed as noise tolerance at a fixed accuracy target, the AV model absorbs **~3× more audio noise**
than audio-only and **~2–2.7× more video noise** than video-only. Training with raw-waveform noise
flattens the curve — robustness gets baked in, and α grows (5.20 clean → 6.03 raw-noise-trained),
i.e. the model leans harder on the gate when audio is less trustworthy.

### 5.3 · How integration works (the mid-multiplicative mechanism)

- **Sparse gating.** Of 64 gate channels, the top two (channels 22 and 12) account for ~80pp of the
  cross-modal gain; the top 21 together account for **94.5pp** (zeroing them collapses AV), the
  bottom 22 for only 1.3pp. The *readout* downstream is the opposite — fully distributed, no single
  channel worth more than ~0.25pp. Concentrated at the gate, redundant at the readout.
- **Vision modulates audio.** Knock out the top visual gate channel → AV drops **42pp**; the top
  audio gate channel → only **6pp** (~7× asymmetry).
- **It reads lips at the articulation peak.** Temporal saliency peaks at frames 20–30 (~400–600 ms);
  spatial saliency at the centre-lip patch (edges ~10× less informative).
- **Sharp tuning.** The learned α ≈ 5.20 sits on a sharp peak — α = 4 → 94.1%, α = 7 → 91.7%,
  α = 10 → 68%. Both weaker and stronger gains hurt.
- **A word-decodability staircase.** Word information climbs with depth, with the two biggest jumps
  exactly at the gate and at the post-gate block (mid-level audio 28% → post-gate 76% → block2 94%);
  label mutual information rises 1.84 → 4.00 → 4.76 nats along the same path.
- **The v = 0 calibration cliff.** Zero the video and AV softmax accuracy crashes to **0.84%** —
  *below chance* — yet a fresh linear probe on the very same features recovers **52.6%**. The
  information is still there; the trained final layer was just calibrated to always-have-video inputs
  and breaks when the activation distribution shifts. (This cliff is unique to mid-fusion; late and
  early fusion degrade gracefully.)

### 5.4 · The architecture comparison

| Variant | Clean acc | Acc at σ_a = 0.05 | Sparse gate? | Inhibition? | v = 0 cliff? |
|---|---:|---:|:--:|:--:|:--:|
| Mid-multiplicative | 95.80% | 60.9% | ✓ (top-2 ≈ 80pp) | ✓ (~40% neg. residuals) | ✓ (0.8%) |
| Mid-additive | 95.80% | 56.0% | partial | ✗ (math forbids it) | ✓ | 
| Late fusion | 95.50% | 28.0% | ✗ (distributed) | n/a | ✗ (40.5%) |
| Early fusion | 94.22% | 22.6% | n/a | n/a | ✗ (82.9%) |

The **≥95% ceiling** is reached by every architecture — it just comes from having two complementary
streams (audio confusables like TEN/TURN are visually separable; visual confusables like BAT/PAT/MAT
are audibly separable). What's **multiplicative-specific** is the inhibition (vision can *suppress*
audio channels; ~40% of gate residuals are negative) and the context-dependent gain (the gate runs
~70% louder when audio is alone). Early fusion's penalty is just an input-scale artifact: raw audio
is ~200× larger in magnitude than video pixels, so concatenating them at the input drowns the video.

### 5.5 · The 19-question modality-integration study

A structured, evidence-only study answering 19 questions, each with a cited artifact and an
independent re-derivation by a separate validation harness. **18 of 19 landed 🟢** (independently
reproduced); Q14 (recurrence) was the long-pole retrain. Compact index:

| # | Question | One-line answer |
|---|---|---|
| Q1 | A/V/AV accuracy | 92.70 / 86.50 / 95.67 (fp32) |
| Q2 | AV gain vs A, V | +2.97pp / +9.17pp; +0.61pp over best ensemble |
| Q3 | AV fed one modality | crashes (0.84% / 44.5%) but a probe recovers (52.6% / 67.4%) — a calibration cliff |
| Q4 | Which modality drives the gain | audio is the base; the **gain is video's, via the gate** (99.2% of rescues need it) |
| Q5 | Stimulus features behind the gain | the "long words drive it" story is a **confound** with audio headroom |
| Q6 | How it uses features | reads lips at articulation peak, multiplied into audio via a sparse gate |
| Q7 | Layer-to-layer information | a clean MI staircase 1.84 → 4.00 → 4.76 nats (big lifts at the gate and block2) |
| Q8 | How the streams integrate | a learned **multiplicative** gate; vision modulates audio (~7× asymmetry) |
| Q9 | Inverse effectiveness | yes, both sides (+49.6pp at σ_a 0.02; +57.2pp at σ_v 0.40) |
| Q10 | A/V/AV geometry | AV sits in its **own** basin, closer to audio (RSA 0.79) than to video (0.74) |
| Q11 | What improves for AV | essentially everything — word, viseme and onset decodability all rise |
| Q12 | Which stimulus features aid integration | almost none survive controlling for audio headroom; but the help is real and **broad** |
| Q13 | Accuracy vs fusion stage | **mid-fusion wins**; separation shows up under noise |
| Q14 | With vs without recurrence | head-to-head recurrent-vs-matched-feedforward retrain (the temporal-binding control) |
| Q15 | Lesioning key neurons | the rescue lives in ~21 of 64 gate channels; readout is distributed |
| Q16 | Does AV reshape the audio rep | yes — same subspace (CKA 0.98) but a worse standalone word readout |
| Q17 | A-model vs AV-model on the same clip | agree 91.8%; AV is right 2.5× more often on disagreements; fuses under McGurk |
| Q18 | Is the standalone A-recogniser better | no — joint training reorganised the audio path, didn't upgrade it |
| Q19 | Scrambled audio | vision rescues **only** when audio keeps its temporal/spectral structure (phase-scramble +18.7pp; time-shuffle ≈ 0) |

### 5.6 · The MSI battery (11 neuroscience/psychophysics tests)

Mapping standard multisensory measures onto the network (`analyze_av_msi.py`, results in
`analysis/msi/E*_*.csv`). Eight of the ten scored tests came back positive for genuine integration:

- **E1 Inverse effectiveness** — as above (+2.97 clean → +49.6pp at σ_a 0.02).
- **E3 McGurk** — on audio/video conflict clips the AV model picks the audio word only 7.8% of the
  time and lands on a **fused third word 82.7%** of the time (audio-only picks the audio word 92.7%).
  Genuine fusion, not relaying.
- **E4 Supra-additivity** — 27% of post-gate channels are strictly super-additive; 0% pre-gate (the
  control — nothing upstream of the gate integrates).
- **E5 Perturbation** — time-shuffling the video frames collapses AV to 1.1%; freezing them to 5.6%;
  block-shuffling pixels (identity kept, motion killed) to 53%. The model leans on **articulator
  motion**, not face identity.
- **E8 Cross-modal predictability** — inside the AV model you can predict video features from audio
  (R² 0.47) and audio from video (R² 0.60), ~3× the unisensory baselines.
- **E9 Gate readout** — the gate is *louder* when audio is alone (mean |gate| 0.54 vs 0.28 with both
  streams) — an inhibitory-regulation pattern reminiscent of cross-modal phase resetting.
- **E11 Temporal window** — peak at Δt = 0, still 74–84% at ±60 ms, 15–19% at ±200 ms — matching the
  human ~±200 ms audiovisual binding window, with the human-like skew that video-lagging is tolerated
  better than video-leading.

### 5.7 · The late-fusion d′ investigation and the biology verdict

The **late-fusion reliability-gate** model (Section 2b) was built to test the *optimal-cue-
combination* view: an ideal observer weighting two independent, unbiased cues by reliability should
score a **sensitivity (d′)** *at least as high as the better single cue*, and √2× higher when the two
are equally good (Ernst & Banks 2002). *(d′, "d-prime", is a signal-detection measure of how
separable two classes are, in standard-deviation units — higher means more discriminable.)*

The model reproduced optimal combination almost everywhere — **except one regime**: with **clean
audio and a visually confusable face** (call it *E1d-clean*), the fused d′ came out *below* the
better single cue (gain-over-best ≈ 0.94 < 1.0). A long investigation (`analysis/deepdive/D310–D324`)
asked whether this is a defect to fix or correct behaviour:

- **It is a pair-level property, invisible per trial.** Whether the on-screen face has a look-alike
  competitor depends on *which two words are being told apart* — information the gate, which sees one
  clip at a time, structurally cannot access. Every attempt to give the gate that signal failed: at
  every network depth, every combination of features, and even when the routing supervision was
  quadrupled and *trained into* the network (it reached its target weighting and still could not clear
  the floor). The learnable gate tops out well below an oracle that is handed the true regime — a
  **structural** ceiling, not a tuning problem.
- **It is what biology does.** Three primary sources, read in full: humans in a matched congruent
  audiovisual task also **fail to beat their best single sense** (Arnold et al. 2019, n.s.), for the
  same stated reason — no per-trial insight into which sense is currently better; the brain **fuses
  mandatorily** when the conflict is too small to detect (Körding et al. 2007); and the "fused beats
  best" law only holds when both cues are unbiased and discrepancy is small (Ernst & Banks 2002). At
  E1d-clean the confusable face is biased for that word pair, so folding it in *correctly* drags the
  estimate. Forcing the number above 1.0 would make the model **super-biological**.
- **Verdict (`D324`).** No fix warranted — the shortfall is biologically correct. The alternative of
  switching to multiplicative fusion was evaluated and **rejected**: it does not clear that regime
  either, and it catastrophically fails single-cue robustness (when one stream is ablated a cue's d′
  collapses to ~8% of its standalone value), which is the exact property that motivated the reliability
  gate. The shipped model is kept as-is.

This whole arc is preserved as a worked example of *proving a mismatch is not a bug* — analyse the
measurement, research the biology under the exact conditions, and debug to a single-variable proof
before "fixing" anything.

---

## 6 · Tests and validation

This is a research codebase, so "tests" means **independent reproduction of every load-bearing
number**, not unit tests. Three mechanisms:

- **The independent-validator suite — `validator_indep_*.py` (~30 scripts).** For each question in
  the 19-question study, a *separate* harness re-derives the numbers from scratch (its own masked
  forward pass, its own probes), never importing the analysis code it checks. Most numbers reproduced
  **bit-exact**; the few sub-0.5% wobbles were debugger-proven benign (floating-point argmax ties and
  scikit-learn solver-tolerance noise). This is how "🟢" was earned per question.
- **Harness self-tests.** Diagnostic harnesses check themselves against the committed anchors before
  being trusted — e.g. the late-fusion d′ pretests (`analysis/deepdive/diag_*_pretest_*.py`) confirm
  they reproduce a known baseline row before reporting a new one; the comparison harness re-hashes the
  val split (`03c5a87a`, N = 5244) on every run.
- **Pre-registered refute-gates.** The expensive retrains (e.g. the gate-supervision and recurrence
  experiments) were gated behind cheap, kill-criteria-fixed-in-advance pre-checks whose job was to
  *kill* the hypothesis, not confirm it — so a retrain only ran once a mechanism was already shown to
  engage on a cheap substrate.

Two lightweight standalone checks live at the root: `dprime_precision_test.py` and
`reliability_match_test.py`.

---

## 7 · Repository layout — where everything lives

The Python scripts are **flat at the repository root** and many import each other
(`from model_av import ...`, `from paired_dataset import ...`), so they are meant to be run **from the
repo root** and are intentionally *not* reshuffled into packages. Group them by role:

**Data pipeline**
- `preprocess.py` — build the log-mel cache and video memmap from raw clips.
- `paired_dataset.py` — pair audio and video clips; the main `Dataset`.
- `dataset_raw_noisy.py` — same, with on-the-fly audio-noise augmentation.

**Model definitions**
- `model_av.py` — **`AVWordResNet`** (the primary mid-multiplicative model), `VisualEncoder`,
  `CrossModalGate`.
- `model_av_latefusion.py` — **`AVLateFusionReliabilityWordResNet`** (the reliability-gate model);
  `..._GO.py` / `..._candc.py` are frozen snapshots from the d′ investigation.
- `model_av_additive.py`, `model_av_early.py`, `model_av_late.py`, `model_av_recurrent.py` — the
  architecture-comparison variants.
- `model_v_only.py`, `model_v_only_fair.py` — the video-only baselines (`WordResNet` for audio lives
  in `train.py`).

**Training** (`train*.py`) — `train.py` (audio-only), `train_av.py` (mid-mult AV),
`train_v_only_fair.py` (fair video baseline), `train_av_{additive,early,late,recurrent}.py`,
`train_av_ff_baseline.py` (the matched feed-forward control for the recurrence test),
`train_av_latefusion*.py`, and the noise-trained variants (`train_noisy.py`, `train_av_noisy.py`,
`train_av_rawnoise.py`, `train_filtered*.py`).

**Evaluation and d′** — `eval_av_visual_noise.py`, `eval_av_rawnoise_sweep.py`,
`eval_rawnoise_sweep.py`, `eval_matched_av.py`, `dprime_precision_{test,balanced}.py`, and the
late-fusion harness under `analysis/deepdive/` (`dprime_latefusion.py`, `diag_*_D3xx.py`).

**Mechanism deep-dive** — `run_deepdive_tier0.py` (the driver), `phase_a_deepdive.py`,
`phase_c_lesions.py`, `phase_d_saliency.py`, `phase_e_geometry.py`, `phase_f_flow.py`,
`phase_t1_cross_variant.py`, and the `analyze_*.py` family (MSI, phonetics, noise robustness,
per-phoneme accuracy).

**The 19-question study scripts** — `q4_*.py … q19_*.py` (one per gap question) plus
`iso_perf_rendezvous.py`.

**Independent validators** — `validator_indep_*.py` (~30 scripts; see [Section 6](#6--tests-and-validation)).

**Figures** — `build_d1_figures.py`, `build_msi_figures.py`, `make_msi_plots.py`,
`make_validation_figs.py`, `grid_cue_weight.py`, `heatmap_cue_weight.py`; convenience wrappers
`_run_figs.sh`, `_run_latefusion.sh`.

**Directories**
- `analysis/` — all committed results: `msi/` (the 11-test battery CSVs + figures),
  `deepdive/` (mechanism CSVs, the d′ investigation, self-test logs, figures),
  `phonetic_clustering_av/`.
- `examples/` — six short tutorial scripts (run in order) to get a new reader up to speed.
- `models/` — trained checkpoints (**gitignored**; ~240 MB). Canonical ones:
  `audio_only_filtered.pt`, `video_only_fair.pt`, `av_fused.pt`, `av_fused_rawnoise.pt`, and the
  architecture variants `av_fused_{additive,early,late,recurrent}.pt`; the reliability-gate model is
  `av_fused_latefusion.pt` (epoch/candidate snapshots are the d′-investigation trail).
- `data/`, `processed/` — raw clips and preprocessed caches (**gitignored**; large).
- `logs/` — training run logs (**gitignored**).

---

## 8 · Requirements

- **Python 3.10+**
- **PyTorch ≥ 2.0**, `numpy`, `scipy`, **`scikit-learn==1.8.0`** (pinned — the linear probes are
  version-sensitive at the sub-0.5% level), `matplotlib`, `Pillow`
- **`ffmpeg`** on the system `PATH` (video decoding in preprocessing)
- No Lightning, no Hydra, no Hugging Face. A single modern GPU trains any one network in ~1–3 h;
  the analysis battery is inference-only.

---

## 9 · Reproducibility notes

- **One pinned val set** for everything: content hash `03c5a87a`, N = 5244, 180 classes, seed 0.
- **Precision.** The evaluation/analysis harness runs **eager fp32** and reproduces the anchor
  numbers bit-exact. The slightly higher figures sometimes seen (95.80% AV, 86.56% V) are **bf16
  training-log best-vals** of the *same* checkpoints (~+0.13pp). When in doubt, fp32 is canonical:
  AV clean = **95.67%**.
- Determinism via fixed seeds and `cudnn.deterministic`. Linear probes are 5-fold stratified logistic
  regression with `StandardScaler`; CKA is the linear (Kornblith 2019) form; RSA uses cosine
  class-mean representational dissimilarity matrices.

---

## 10 · Caveats and limitations

- **The biology is a set of predictions.** Nothing here was measured in a brain; the neuroscience
  framing (inverse effectiveness, phase-reset-like inhibition, the ±200 ms window) states what the
  *model* does and where it lines up with known results.
- **The multiplicative gate is a computational abstraction**, not a claim about a specific cortical
  circuit — real association cortex uses long-range projections, not an explicit `(1 + α·σ(·))` term.
- **The frame-drop curve is an out-of-distribution artifact** (the 3D-conv's temporal kernel plus
  frozen BatchNorm smear a single zeroed frame across neighbours); treat it separately from the
  Gaussian noise sweeps.
- **Open follow-ups** noted across the study: BatchNorm-normalised early fusion, recovery dynamics of
  the v = 0 cliff, phoneme-distinct (rather than word-distinct) McGurk pairs, natural-speech
  generalisation, and identity-conditioned reliability priors for the confusable-pair regime.

---

## 11 · Project status and handoff

The audiovisual integration work above is **complete and validated**: the mechanism deep-dive, the
11-test MSI battery, the 19-question study (18/19 independently reproduced), the architecture
comparison, and the late-fusion d′/biology arc all have committed artifacts and figures.

A **separate, paused effort** — an *audio-only biological-noise* project (cochlear-synaptopathy and
synaptic-release-failure mechanisms baked into the audio network) — was mid-design when work paused.
Its detailed tick-by-tick handoff previously lived in a 140 KB `SESSION_STATE.md` at the repo root.
That file has been removed from the working tree as part of this cleanup **but is preserved in full**
in the pre-cleanup backup (see the note the assistant reports alongside this commit); restore it from
there if you resume that thread.

---

## 12 · Glossary

- **top-1 accuracy** — fraction of clips whose single highest-scoring prediction is the correct word
  (chance = 1/180 ≈ 0.56%).
- **d′ (d-prime)** — a signal-detection sensitivity measure: how many standard deviations separate
  two classes' response distributions; higher = more discriminable.
- **viseme** — the visual counterpart of a phoneme: a distinguishable lip/mouth shape.
- **inverse effectiveness** — multisensory benefit is largest when each single sense is weak.
- **McGurk effect** — conflicting heard and seen speech fusing into a third perceived word.
- **inverse-effectiveness / super-additivity** — a combined response exceeding the sum of the two
  unisensory responses.
- **CKA / RSA** — representational-similarity measures comparing what two networks (or layers) encode.
- **σ_a, σ_v** — the standard deviation of Gaussian noise added to the audio and video streams.
- **α (alpha)** — the single learned scalar setting the strength of the multiplicative gate (≈ 5.20).

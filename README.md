# zak_daedalus

Spoken word recognition with audiovisual integration. 180 classes, 12 speakers,
audio and video for every clip. Three networks: audio-only (92.7%), video-only
(86.5%), and audiovisual with a mid-block gate (95.8%). Most of the work in here
is mechanism analysis — basically, how does the AV gate combine the two streams.

The gate sits between the two blocks of the audio ResNet and modulates each
audio channel multiplicatively:

    a_out = a_mid * (1 + alpha * sigmoid(W_a · a_mid + W_v · v_mid))

`v_mid` is a 64-channel feature map from a small 3D-conv lip encoder. `W_a`
and `W_v` are 1x1 convs. `alpha` is a learned scalar that ends up around 5.20.
Basically the visual stream produces a per-channel gain on the audio features.

## Quickstart

```
python preprocess.py            # mel cache
python train.py                 # A-only baseline
python train_av.py              # AV
python run_deepdive_tier0.py    # analysis battery
```

WAVs go in `~/Downloads/audio/<speaker>/`, lip videos go in
`data/visual/video_data/<speaker>/`. Filenames look like
`speaker-02_grp1-item7_TEXT-C.wav`. Training takes ~1–3 h per network on an A6000.

## Files to read first

- `model_av.py` — the AV network and the gate.
- `train_av.py` — training loop.
- `paired_dataset.py` — audio/video pairing (clips aren't pre-aligned;
  the pairing is sequential within each speaker/group).
- `analysis/AV_5_QUESTIONS_ANSWERED.md` — short writeup of what came out.

The longer version: `analysis/AV_INTEGRATION_DEEP_DIVE_SYNTHESIS.md`.

## examples/

Six short scripts to get a new reader up to speed. Run them in order.

01. Load each checkpoint, print params + val acc.
02. Pull one sample, plot the mel + lip frames.
03. Run AV and A-only on a few items where AV rescues an A-only mistake.
04. Extract penultimate features for a small batch.
05. Linear probe on those features. Shows the calibration thing — when you
    run the AV network with v=0, softmax accuracy drops to 0%, but a fresh
    linear classifier on the same features still gets ~74%. So the features
    still encode word identity; the trained classifier was just fit with
    v != 0 inputs and breaks when v goes to 0.
06. Visualise the gate on one sample. Annotates channels 12 and 22 (the
    two that carry most of the integration).

## Headline findings

The AV advantage on clean speech is small — about +3 pp over A-only. At
moderate audio noise though, AV beats A-only by roughly 50 pp. That's the
inverse effectiveness pattern.

The mid-block gate is sparsely coded: 2 of its 64 channels carry most of the
cross-modal information. Zeroing either one drops AV by about 40 pp. The
trained `alpha` sits on a sharp peak; both weaker and stronger gains hurt.

At the pre-gate layer the AV audio features get 28% standalone word
accuracy vs 43% for the A-only model. At the post-block-2 layer the
order flips: AV at 94%, A-only at 90%. So joint training shapes the AV
audio path for fusion with the visual stream — it gives up a bit of
standalone discriminability up front and gets it back downstream.

The cross-architecture ablation covers mid-multiplicative, mid-additive,
late fusion, and early fusion. All four hit similar clean accuracy
(within ~1.5 pp). Only the multiplicative gate shows inhibitory regulation
and context-dependent gain control. Late fusion uses a fully distributed
readout. Early fusion underperforms because the audio mel and video pixel
values differ by ~200× in magnitude at the input layer.

The synthesis doc has the full breakdown.

## Requirements

Python 3.10+, `torch >= 2.0`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`,
`Pillow`. `ffmpeg` on the system path. No Lightning, no Hydra, no Hugging Face.

## What's gitignored

- `models/` — trained checkpoints
- `processed/` — mel cache and video memmap
- `data/visual/cache/` and `data/visual/video_data/` — raw clips and cache
- `examples/output/` — generated PNGs and NPZs from the examples
- `*.log` — training output

Everything in `analysis/` is tracked.

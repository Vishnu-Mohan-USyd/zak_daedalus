# Examples

Six small scripts that walk you through the trained audiovisual word models — from "does the
checkpoint load" to "what is the fusion gate actually doing". Run them in order the first time;
each one is standalone except `05`, which eats the file `04` writes.

Everything here reads the repo's real checkpoints and real validation split, so you need
`models/` and `processed/` populated first (see the top-level README quickstart). Both are
gitignored, so a fresh clone has to build them. Figures and arrays land in `examples/output/`,
which is also gitignored — delete it any time.

```bash
python examples/01_load_model.py     # from the repo root, not from inside examples/
```

### `01_load_model.py`
Loads all six checkpoints — audio-only, video-only, and the four fusion variants (mid-block
multiplicative, additive, late, early) — and prints a table of parameter count and best
validation accuracy for each. It also prints the SHA of the validation index, a fingerprint of
which clips were held out; when that hash matches across checkpoints, their accuracies are
genuinely comparable. Ends with a module-by-module breakdown of the fused model, which is the
fastest way to remind yourself what the blocks are called.

### `02_inspect_data.py`
Pulls a single paired clip out of the dataset and shows you what the network sees: the log-mel
spectrogram (audio as a time × frequency image) and four evenly-spaced frames of the lip ROI.
Prints the word, speaker, tensor shapes, and value range; saves `output/02_sample.png`. Start
here if the data layout is unfamiliar.

### `03_run_inference.py`
The point of the whole project in one screen. Runs 200 validation clips through both the fused
model and the audio-only model, and labels each as RESCUE (fusion gets it right, audio alone
doesn't), AGREE (both right), or MISS (both wrong). Prints five representative cases with each
model's top-3 guesses, then the overall tally. The RESCUE rows are where the lips are doing
real work.

### `04_extract_features.py`
Extracts the 128-dimensional penultimate features — the representation just before the final
classifier — for 1000 validation clips, in four conditions: the fused model, the fused model
with the visual pathway zeroed, audio-only, and the capacity-matched video-only model. Reports
per-feature variance and flags dead dimensions, then saves everything to
`output/04_features.npz` for the next script.

### `05_linear_probe.py`
Answers a question `04` sets up. When you zero the visual input, the fused model's accuracy
falls off a cliff — but is the word information gone, or just unreadable? This trains a fresh
logistic-regression classifier (5-fold cross-validated) on those same features and recovers most
of the accuracy, which means the information was still there. The trained classifier had simply
been fit on inputs where the visual stream was present and mis-calibrates without it. Prints the
probe accuracies next to the model's own accuracy for comparison. Needs `04` to have run.

### `06_gate_visualization.py`
Opens up the cross-modal gate, the mechanism that lets vision modulate audio. Prints the learned
gain scalar α (initialised at 0.2 — how far it drifted tells you how hard the model leaned on
vision) and the per-channel gain, then saves `output/06_gate.png`: the mel input, a lip frame, a
bar chart of gain across all 64 audio channels, and a time × frequency heatmap of the gate. The
two channels highlighted in red were the biggest accuracy losses under lesioning, i.e. the ones
carrying most of the visual contribution.

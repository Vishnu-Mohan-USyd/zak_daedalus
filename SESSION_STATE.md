# Session handoff — A-only bio-noise project

Last checkpoint: 2026-06-09. Machine going down for repair. Resume from here.

## Where we are in the flow

Researcher delivered the round-2 decisive spec. **User has NOT yet approved.** Decision point on resume: approve the 2-mechanism spec below → dispatch coder, OR ask for revisions.

No code touched, no training launched. Decision + spec only.

## The approved-pending spec — minimal developmental noise to land A-only at 70–79%

Target: `models/audio_only_filtered.pt` (`WordResNet`, train.py:94-111, 92.70% clean val, val sha `03c5a87a…`, N=5244, 180 classes) → predicted ~75% (range 72–79%) post-retrain under the set below.

### M1 — Cochlear synaptopathy (load-bearing, structural, type-a)

- **Site**: output of `self.block1` (the `(N,64,40,50)` activation)
- **Op**: fixed boolean buffer `mask1` shape `[64]`, ~26 zeros (40%) chosen once at init from a fixed seed; `x = x * mask1[None,:,None,None]`
- **Dose**: 40% (26/64). From ~40–50% afferent-synapse / ~40% SGN loss in aged CBA/CaJ mouse (Sergeyenko 2013) and ~30% in aged human temporal bones with full hair-cell complements (Liberman & Liberman 2015, JARO 16:205).
- **Active when**: fixed permanent mask — identical at train and inference, never resampled, never disabled
- **Citations**: Kujawa & Liberman 2009, J Neurosci 29:14077; Sergeyenko et al. 2013, J Neurosci 33:13686; Liberman & Kujawa 2017, Hear Res 349:138

### M2 — Cortical synaptic release failure (recovery-margin consumer, stochastic, type-b)

- **Site**: output of `self.block2` (the `(N,128,20,25)` activation)
- **Op**: per-(sample,channel) Bernoulli `keep ~ Bern(0.7)` shape `[N,128]`; `x = x * keep[:,:,None,None]` — channel-wise so it survives GAP, NO 1/p rescale (a release failure is signal lost, not gain-corrected)
- **Dose**: p_keep = 0.7 (30% failure). From cortical release probability p≈0.1–0.5 with across-bouton CV>0.5 (Branco & Staras 2009; Faisal 2008); 0.7 sits in the reliable end of that range
- **Active when**: training AND inference — resampled every forward including eval
- **Citations**: Branco & Staras 2009, Nat Rev Neurosci 10:373; Faisal, Selen & Wolpert 2008, Nat Rev Neurosci 9:292

### Training protocol

Unchanged `train_filtered.py` (200 epochs, AdamW lr 1e-3, wd 1e-2, CosineAnnealing→1e-6, batch 64, SpecAugment + Dropout(0.3) retained), with M1+M2 inserted in the forward pass. Validation runs with BOTH active. `val_idx` sha-pinned (03c5a87a…) so the number is comparable to 92.70%.

## Why this set is minimal

Researcher's read-only inference probes on the actual checkpoint (proof before retrain):

- Multiplicative gain noise (Ribrault σ=0.3) → 91.4% un-adapted. Would recover to ~92% post-retrain. Compensable. **Rejected.**
- Channel-wise Bernoulli at block2 p_keep=0.7 → 84.6% un-adapted; p_keep=0.5 → 72.5%. But Bernoulli IS dropout — post-retrain net learns redundancy, climbs back. Compensable on its own. M2 alone not enough.
- Permanent 40% channel removal: un-adapted input-mel 11.8% / block1 2.6% / block2 82.4%. Late-layer capacity is hugely redundant — only EARLY permanent removal is non-compensable. M1 site must be block1.
- M1 alone post-retrain → ~83–89% (surviving 38 channels + block2 redundancy re-optimize and recover). Too high.
- M1+M2 together: M1 caps early capacity, M2 strips block2's recovery margin (which it would have used to climb back). ~75%.

## Controls (one per mechanism)

**M1 control** — matched 40% lesion at block2 instead of block1. Same channel count, late site. Tests whether the synaptopathy claim is specifically about early/peripheral capacity (the biological essence) vs generic capacity loss. Un-adapted probe gap (2.6% vs 82.4%) predicts the late lesion will recover ~baseline post-retrain.

**M2 control** — matched per-element variance additive Gaussian at block2, always-on (replacing the multiplicative Bernoulli). Tests whether the all-or-none quantal-failure signature does specific work, or any same-variance noise reproduces the drop. (Rescale-vs-no-rescale is NOT the right control — probed identical.)

## Caveats flagged by researcher

1. M1 permanence is mouse/human (Kujawa/Liberman lineage); guinea pigs regenerate (Hickman et al. 2021, Front Cell Neurosci 15:684706). Don't cite cross-species — keep claims to mouse/human.
2. Post-retrain recovery is the one number that can't be bounded from read-only probes. 72–79% range is the honest estimate.
3. Developmental regulation of release-p over maturation is genuinely open. Treat p=0.7 as adult steady-state.
4. Two clean tuning knobs if it lands off-range: HIGH → lesion 45–50% or p_keep 0.6; LOW → lesion 35% or p_keep 0.75. Both moves stay inside cited dose ranges.

## Methods paragraph (researcher's draft, for the paper)

> To model a listener whose auditory periphery and cortex develop and operate under irreducible biological noise, we degraded the audio pathway with two mechanisms active throughout training and inference. First, we modelled cochlear synaptopathy — the early-onset, progressive, and (in mouse and human) permanent loss of inner-hair-cell→auditory-nerve afferent synapses that accumulates across the lifespan and leaves audiometric thresholds intact while permanently reducing suprathreshold neural output (Kujawa & Liberman 2009, J Neurosci 29:14077; Sergeyenko et al. 2013, J Neurosci 33:13686; Liberman & Kujawa 2017, Hear Res 349:138) — as a fixed, randomly chosen removal of 40% of the first convolutional stage's 64 feature channels, applied identically at training and test, consistent with the ~40–50% afferent-synapse and ~40% spiral-ganglion-neuron loss reported in aged CBA/CaJ mice and ~30% in aged human temporal bones with full hair-cell complements (Liberman & Liberman 2015, JARO 16:205). Second, we modelled the lifelong stochasticity of central synaptic transmission — cortical synapses release neurotransmitter probabilistically with low, heterogeneous release probability (p≈0.1–0.5, across-bouton CV>0.5; Branco & Staras 2009, Nat Rev Neurosci 10:373; Faisal, Selen & Wolpert 2008, Nat Rev Neurosci 9:292) — as an always-on, channel-wise Bernoulli release-failure mask (release probability 0.7, no amplitude rescaling) on the second convolutional stage, again active at both training and test. Both mechanisms were present from initialization and never disabled, so the network developed under and was evaluated with them in place; neither is a train-time-only augmentation.

## Key user constraints learned this session

- Scientific paper context — every dose must cite a primary source
- Developmental noise only (training-time present). Inference-only changes are invalid.
- "Most minimal" mechanism set — 1, 2, or 3, whichever genuinely required (no filler additions)
- Pure stochastic noise during training → model recovers (rawnoise model 92.70%→93.25% proves this). Mechanism must be type-(a) permanent structural OR type-(b) always-on at train+inference.
- No mashup of weakly-justified knobs. Each member of the set must carry independent biological + quantitative load.
- No retrain until adversarially proven on cheap substrate (PROVE-BEFORE-RETRAIN).

## Team state

Team `daedalus-v6` (config at `~/.claude/teams/daedalus-v6/config.json`). Four members all idle awaiting dispatch:
- `team-lead@daedalus-v6` (me on resume)
- `coder@daedalus-v6` (opus, comms verified)
- `researcher@daedalus-v6` (opus, comms verified, two reports delivered, currently idle)
- `debugger@daedalus-v6` (opus, comms verified)
- `validator@daedalus-v6` (opus, comms verified)

Roster names match — address by name in SendMessage. No spawning new agents (CLAUDE.md hard rule).

## Git state

- Branch: `av-integration`
- Latest commit: `c5143d65 AV integration: gate variants, analysis, tutorials`
- Working tree clean (will be dirty after this file is written + committed)
- Pushed to `publication` on `Vishnu-Mohan-USyd/zak_daedalus`

## On resume — what to do first

1. Inbox sweep (Rule 5.1) on `~/.claude/teams/daedalus-v6/inboxes/team-lead.json`
2. Re-read this file
3. Surface to user: "spec is on the table for M1+M2. Approve to dispatch coder, or revise?"
4. If approve → coder dispatch: implement M1 (block1 channel mask) + M2 (block2 channel Bernoulli) in `WordResNet` forward pass, sha-pinned val. Adversarial pre-verification first (debugger or validator) on cheap proxy — confirm un-adapted probe numbers match researcher's 2.6%/82.4% before launching the full retrain.
5. Queue M1 control (block2 lesion) and M2 control (additive-Gaussian-matched-variance) retrains behind the main run.

## Next-step prompt drafts (if user redirects)

If user wants the coder dispatched directly: prompt structure should reference `train.py:94-111` for the layer names, the exact ops in the M1/M2 sections above, the training protocol unchanged, val-sha pinning, and the pre-verification gate. No fresh research needed — the spec above is the implementation contract.

If user wants a third mechanism added: re-open with researcher; the current rejection of multiplicative quantal-gain (Ribrault σ=0.3) was probe-evidence-based (91.4% un-adapted, ~92% post-retrain). Any addition needs to clear that bar.

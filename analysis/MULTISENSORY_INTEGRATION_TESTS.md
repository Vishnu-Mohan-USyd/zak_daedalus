# MSI test battery — project notes

This is the pre-deep-dive plan for mapping the usual multisensory-integration measures (Stein–Meredith inverse effectiveness, Calvert pSTS supra-additivity, McGurk fusion, Miller race bound, temporal-window, Bayesian cue combination, Lakatos cross-modal modulation) onto the trained networks. The battery ran on the AV-clean checkpoint (`models/av_fused.pt`, 95.80%), A-only (`audio_only_filtered.pt`, 92.70%) and V-only (`video_only_fair.pt`, 86.59%). Results live in `MSI_RESULTS.md`; per-experiment CSVs and plots are in `analysis/msi/E*_*.{csv,png}`.

Here's the mapping (the full table is kinda superseded by the deep-dive results now):

| Bio metric | Network analog | Tightness |
|---|---|---|
| Stein–Meredith inverse effectiveness | (AV − A) accuracy gap vs σ_a | TIGHT |
| Calvert pSTS supra-additivity | per-channel MEI / SA at `audio_block2` | TIGHT |
| McGurk fusion | top-1 drift to a third word under viseme-distinct conflict | TIGHT |
| Stein–Meredith temporal window | AV accuracy vs Δt video shift | TIGHT |
| Lakatos cross-modal modulation | gate magnitude when v real vs zeroed | TIGHT |
| Miller race bound | per-item: 𝟙[AV] ≤ 𝟙[A] + 𝟙[V] | MEDIUM |
| Ernst–Banks optimal cue | σ_AV² vs σ_A² σ_V² / (σ_A² + σ_V²) | LOOSE |

Stuff you basically can't run on this architecture: true Miller RMI (no RT distributions to work with), psychophysical thresholds (categorical 180-class output), ERP analysis (single forward pass, no oscillations), TMS (no focal cortical lesion possible), and inverse effectiveness on V-degradation (no video-noise pipeline at the time).

The deep-dive (Tier-0 / Tier-1) extended this battery with per-channel mechanism analyses (D-blocks D1–D5) and a cross-variant comparison across four AV architectures. See `AV_INTEGRATION_DEEP_DIVE_SYNTHESIS.md`.

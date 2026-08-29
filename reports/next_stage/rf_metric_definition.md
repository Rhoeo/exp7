# P4 Digital Waveform-Validity Metric Definition

These metrics assess digital plausibility under a fixed 35 MHz sampled complex-baseband representation. They do not establish DAC/PA/channel/receiver or over-the-air feasibility.

The common order is:

`token edit -> frozen decode -> reference-power match -> 22 MHz perturbation band-limit -> perturbation-power projection -> peak-delta projection -> metric audit -> optional classifier audit`

Success or identity claims use only the projected waveform. Pre-projection values are retained for diagnostics.

Primary metrics:

- SNR and relative perturbation power, relative to the codec reconstruction used as the edit reference.
- Final-signal PAPR; perturbation PAPR is secondary.
- Normalized peak delta and clipping fraction at four reference RMS units.
- EVM-like error relative to the clean/codec waveform. This is not standards-compliant demodulation EVM.
- Normalized PSD L1 distance.
- Observable out-of-band energy ratio outside versus inside the centered 22 MHz occupied band. Standard ACLR is disabled because 35 MHz sampling does not cover complete adjacent Wi-Fi channels.
- Waveform correlation and normalized maximum first derivative for both final waveform and perturbation.
- Preamble correlation is disabled unless the evaluated window is independently verified to be preamble-aligned.

The initial projection bounds are a 22 dB minimum perturbation SNR and a normalized peak-delta cap of 1.0. Distribution-based Gate thresholds are calibrated against clean-to-codec reconstruction, standard RF augmentation where available, and the existing band-limited PGD diagnostic.

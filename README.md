---
title: StereoAwareGNN BBB Predictor
emoji: 🧠
colorFrom: green
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

# StereoGNN-BBB: Blood-Brain Barrier Permeability Predictor

A graph neural network for blood-brain-barrier permeability prediction.

> **Correction (2026).** An earlier version of this README advertised "0.9612 AUC,
> state-of-the-art on external validation." That figure came from training on BBBP
> and testing on B3DB, which overlaps BBBP by ~22%, so it was not true external
> validation. Under leakage-controlled evaluation (scaffold splitting) the model
> scores roughly **0.83-0.84 AUC**, is competitive with but does **not** clearly
> beat a simple ECFP + random-forest baseline, and its stereo features do not
> affect BBB predictions (it returns identical outputs for enantiomers). A full,
> reproducible audit is here:
> https://github.com/abinittio/bbb-honest-eval

## Author
Nabil Yasini-Ardekani

## Features
- Graph neural network for BBB permeability prediction
- Real-time BBB permeability prediction
- Molecular visualization
- Export results as JSON/CSV

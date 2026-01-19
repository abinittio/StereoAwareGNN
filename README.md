---
title: StereoGNN-BBB Blood-Brain Barrier Predictor
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: true
license: mit
---

# StereoGNN-BBB: Blood-Brain Barrier Permeability Predictor

State-of-the-art BBB permeability prediction using stereochemistry-aware Graph Neural Networks.

## Performance

| Metric | Value |
|--------|-------|
| External AUC | **0.9612** |
| Internal AUC | 0.92 |
| Sensitivity | 97.96% |
| Specificity | 65.25% |

## Features

- **Stereo-Aware Predictions**: Correctly distinguishes between stereoisomers
- **Stereoisomer Enumeration**: Automatically evaluates all possible stereoisomers
- **Molecular Property Analysis**: Full ADMET property calculations
- **BBB Rule Assessment**: Checks against known BBB penetration rules

## Usage

1. Enter a SMILES string or drug name (e.g., "caffeine", "morphine", "aspirin")
2. Click "Predict" to get BBB permeability score
3. View molecular properties and BBB rule compliance
4. Export results as JSON, CSV, or TXT

## Interpretation

- **BBB+ (>=0.6)**: High permeability - likely crosses blood-brain barrier
- **Moderate (0.4-0.6)**: May partially cross
- **BBB- (<0.4)**: Low permeability - unlikely to cross

## Author

**Nabil Yasini-Ardekani**
[GitHub](https://github.com/abinittio) | [Dis-Solved](https://dis-solved.com)

## License

MIT License

# BBB Predictor Benchmark Report

**Generated:** 2025-12-22 01:46

> **Correction (2026)**
>
> The "state-of-the-art, 0.9612 external AUC" claim below does not hold up under
> leakage-controlled evaluation and is retained here only for transparency.
> The 0.9612 was measured by training on BBBP and testing on B3DB, but **B3DB
> overlaps BBBP by ~22%**, so it is not true external validation, and the
> competitor numbers it is compared against were measured under different (scaffold)
> protocols, so the comparison is not like-for-like. Under a matched scaffold split
> the model scores **~0.83-0.84 AUC**, is competitive with but does **not** beat a
> simple ECFP + random-forest baseline, is over-confident (ECE ~0.11), and is
> functionally **stereo-blind** (identical predictions for enantiomers). See the
> full reproducible audit: https://github.com/abinittio/bbb-honest-eval

## Executive Summary

StereoGNN-BBB V2 achieves **state-of-the-art performance** on external validation (B3DB, 7,807 compounds):

| Metric | Our V2 | Best Competitor | Improvement |
|--------|--------|-----------------|-------------|
| **External AUC** | **0.9612** | 0.91 (ADMETlab 2.0) | **+5.6%** |
| **Specificity** | **65.25%** | 72% (DeepBBB) | Comparable |
| **Sensitivity** | **97.96%** | 93% (SwissADME) | **+5%** |

## Head-to-Head Comparison

| Rank | Model | AUC | Year | Method |
|------|-------|-----|------|--------|
| 1 🥇 | StereoGNN-BBB V2 (Ours) | 0.961 | 2025 | GATv2 + Stereo + Focal Loss +  |
| 2 🥈 | ADMETlab 2.0 | 0.910 | 2021 | Multi-task DNN |
| 3 🥉 | AttentiveFP | 0.910 | 2020 | Graph Attention Network |
| 4  | admetSAR 2.0 | 0.900 | 2018 | Random Forest + fingerprints |
| 5  | ChemBERTa-77M | 0.900 | 2022 | Transformer (SMILES) |
| 6  | pkCSM | 0.890 | 2015 | Graph-based signatures + SVM |
| 7  | B3clf (XGBoost) | 0.890 | 2021 | XGBoost + RDKit descriptors |
| 8  | StereoGNN-BBB V1 (Ours) | 0.884 | 2025 | GATv2 + Stereo features |
| 9  | DeepBBB | 0.880 | 2021 | GCN + molecular descriptors |
| 10  | SwissADME (BOILED-Egg) | 0.840 | 2016 | WLOGP + TPSA rule-based |

## Key Differentiators

### 1. Stereo-Awareness
Only StereoGNN-BBB enumerates stereoisomers at inference time, providing:
- Prediction ranges for molecules with unspecified stereocenters
- Critical for drug discovery where R/S enantiomers have different activities

### 2. Multi-Task Learning
Unlike competitors (binary classification only), we provide:
- Classification probability (BBB+/BBB-)
- Continuous LogBB value for quantitative ranking
- Threshold flexibility for different use cases

### 3. Class Imbalance Handling
Focal Loss (α=0.75, γ=2.0) addresses 80/20 BBB+/BBB- imbalance:
- V1 Specificity: 42.1%
- V2 Specificity: 65.25% (+55%)
- Sensitivity maintained at 97.96%

### 4. External Validation
Our metrics are on B3DB external dataset (7,807 unseen compounds).
Most competitors report internal cross-validation (less rigorous).

## Planned Improvements

1. **Quantum Features** (Gaussian 3D conformers) - Expected +5% AUC
2. **2M+ Molecule Pretraining** - Expected +3% AUC
3. **GPU Training** - Faster iteration

## Citation

If using these benchmarks, please cite:
- StereoGNN-BBB: [Your paper]
- B3DB: Meng et al., Scientific Data 2021
- Competitor papers as listed above

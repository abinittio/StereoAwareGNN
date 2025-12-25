# PROJECT LOCKED

## BBB Permeability Predictor - Stereo-Aware GNN v1.0

**Status:** COMPLETED AND LOCKED
**Lock Date:** December 20, 2025

---

## Final Performance

| Metric | Value |
|--------|-------|
| **Mean AUC** | **0.8968 ± 0.0156** |
| Mean Accuracy | 85.04% |
| Baseline Improvement | +6.52% |

---

## Project Summary

- **Model:** StereoAwareEncoder (GATv2 + Transformer)
- **Features:** 21 dimensions (15 atomic + 6 stereo)
- **Pretraining:** 322,594 ZINC stereoisomer graphs
- **Fine-tuning:** BBBP dataset (2,050 molecules)
- **Web App:** Streamlit UI with name/formula/SMILES input

---

## Key Files (DO NOT MODIFY)

```
models/
  pretrained_stereo_full.pth      # Pretrained encoder
  bbb_stereo_fold1_best.pth       # Fine-tuned models
  bbb_stereo_fold2_best.pth
  bbb_stereo_fold3_best.pth
  bbb_stereo_fold4_best.pth       # Best fold (AUC 0.9111)
  bbb_stereo_fold5_best.pth

data/
  zinc_stereo_graphs.pkl          # 322k preprocessed graphs (1.3 GB)
  bbbp_dataset.csv                # Training data

Core Scripts:
  zinc_stereo_pretraining.py      # StereoAwareEncoder architecture
  pretrain_full_stereo.py         # Pretraining script
  finetune_bbb_stereo.py          # Fine-tuning script
  bbb_webapp.py                   # Web application
  TECHNICAL_SUMMARY.md            # Documentation
```

---

## Version Tag

**StereoGNN-BBB-v1.0-FINAL**

This project is complete. Do not modify core model files.
For improvements, create a new project directory.

---

## Citation

If using this model, reference:
- Architecture: Stereo-Aware GATv2 + TransformerConv
- Features: 21-dim (atomic + R/S chirality + E/Z geometry)
- Pretraining: Self-supervised on ZINC stereoisomers

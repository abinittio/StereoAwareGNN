---
title: StereoAwareGNN BBB Predictor
emoji: 🧠
colorFrom: green
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

# StereoGNN-BBB — blood–brain barrier permeability, evaluated honestly

A stereochemistry-aware graph neural network for blood–brain barrier (BBB)
permeability prediction, and — more usefully — a **protocol-controlled benchmark of
what BBB models are actually worth** when the evaluation is tightened.

The headline finding is not about this architecture. It is that for this endpoint,
**the evaluation protocol moves the score more than the architecture does**, and the
model that looks best under the standard protocol is not the model that generalises.

## The benchmark, tagged by protocol

Eight architectures, identical data and folds. Every number below is measured in this
study (5-fold, ± std where folds apply); "external" means train on BBBP and test on
B3DB after removing every InChIKey that also appears in BBBP.

| Model | Random split | Scaffold split | External (de-duplicated) |
|---|---|---|---|
| ChemBERTa | **0.958** | 0.924 | 0.746 |
| SMILES-LSTM | 0.954 | **0.926** | 0.766 |
| SMILES-CNN | 0.947 | 0.917 | 0.741 |
| ECFP + random forest | 0.926 | 0.863 | **0.921** |
| ECFP + logistic regression | 0.909 | 0.821 | 0.876 |
| MLP (descriptors) | 0.908 | 0.814 | 0.878 |
| StereoGNN (from scratch) | 0.897 | 0.834 | 0.846 |
| StereoGNN (ZINC-pretrained) | 0.896 | 0.856 | 0.856 |

Read the first column against the last. ChemBERTa wins the random split by a clear
margin and loses 0.21 AUC going external. ECFP + random forest — fingerprints and a
forest, the cheapest thing in the table — gives up 0.005 and finishes first where it
counts. Ranking by the standard protocol would have picked the wrong model.

## The 0.96, and why it is still here

This project previously advertised **0.9612 AUC "state-of-the-art on external
validation."** That number is not deleted from the record, because it is informative:

| Protocol | Result | What it measures |
|---|---|---|
| StereoGNN V2 ensemble, scored on B3DB it was trained on | **0.96** | memorisation — the folds had seen the test set |
| Published BBB headlines (GMC-MPNN, MegaMolBART, FP-XGBoost) | 0.88–0.97 | random or scaffold split, in-dataset, no de-duplication, no novel-chemistry holdout |
| This work, leakage-controlled external | 0.846–0.856 | generalisation to unseen compounds |
| This work, novel chemistry only (NN Tanimoto < 0.4) | ≈0.6 | generalisation to unfamiliar scaffolds |

The 0.96 was produced **by the same class of protocol that produces the published
headline numbers** — which is the point. Tighten the protocol on the same model and
the score falls to 0.85; restrict to genuinely novel chemistry and every architecture
in the table, ChemBERTa included, lands at AUC ≈ 0.58–0.64, close to chance.

*Honest limitation, stated plainly: no published BBB model reports a leakage-controlled,
low-similarity external AUC, so the claim "published SOTA would also collapse" is an
extrapolation from the protocol comparison above, not a measured head-to-head. The
independent evidence for it is a documented random-vs-scaffold gap of ~0.16 AUC on the
same data (PMC8708321) and ~0.68 AUROC on near-neighbour stereoisomer discrimination
(CANDID-CNS).*

## What the stereo features do

Nothing measurable, for this endpoint. The model returns identical predictions for
enantiomers, so the stereochemical channel it is named for does not influence BBB
output. That is reported rather than quietly dropped, and it is why the interesting
result here is the benchmark rather than the architecture.

## Related repositories

- **[bbb-honest-eval](https://github.com/abinittio/bbb-honest-eval)** — the reproducible
  leakage-controlled audit behind the correction above.
- **[bbb-permeability](https://abinittio.github.io/bbb-permeability/)** — findings page
  for the wider architecture comparison.
- **[filtranex](https://github.com/abinittio/filtranex)** — where this line of work is
  applied: a screening cascade that ships an applicability-domain gate precisely because
  of the collapse documented here.

## Use

```bash
pip install -r requirements.txt
python predict_bbb.py --smiles "C[C@@H](N)Cc1ccccc1"
```

A Streamlit/Docker app (`app.py`) provides the same prediction through a browser. In
environments without `torch_geometric` the app falls back to the descriptor model and
says so on screen.

## Citation

If this benchmark or the correction informs your work, please cite it — see
[`CITATION.cff`](CITATION.cff) or use GitHub's "Cite this repository" button.

## Licence

MIT — see [`LICENSE`](LICENSE).

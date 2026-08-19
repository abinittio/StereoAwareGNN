# StereoAwareGNN — ECS7037P Audit Deck

Generated from a multi-agent deep-read + optimisation audit of the live repo (28 Jun 2026).
Each finding is a card. We work through them one at a time. Slide anchors refer to the
ECS7037P Week 3 decks (Lecture 01 = L1, Lecture 02 = L2).

Card shape: Severity | validity (threat-to-validity) | leakage-type (Kapoor & Narayanan 2023 where
relevant) | what the code does (file:line) | why it matters (+ slide) | fix | acceptance | report line.

---

## Index

### Track A — Leakage / evaluation rigour (the scientific contribution)
| ID | Sev | Title |
|----|-----|-------|
| LEAK-01 | critical | B3DB "external validation" set is 100% inside training; headline 0.9612 is in-sample |
| LEAK-02 | critical | No InChIKey dedup between BBBP-train and B3DB (B3DB subsumes BBBP) |
| LEAK-03 | high | Reported CV AUC selects best epoch on the same val fold it reports (no held-out test) |
| LEAK-04 | high | All splits random over graph lists; no scaffold/cluster split, no dedup |
| LEAK-05 | high | "Beats SOTA 8/8" is a cross-regime artefact |
| LEAK-06 | medium | Fabricated logBB targets inflate regression R2/RMSE |

### Track B — Optimisation / training (Lecture 3)
| ID | Sev | Title |
|----|-----|-------|
| OPT-01 | critical | (= LEAK-03) model-selection-on-validation reported as generalisation |
| OPT-02 | critical | (= LEAK-06) primary regression loss fit to a fabricated 2-value label function |
| OPT-03 | high | No RNG seeding; mean±std conflates init/dropout/shuffle with fold variance |
| OPT-04 | high | Class imbalance unhandled in stereo path; raw accuracy led on 77%-positive set |
| OPT-05 | medium | Gradient clipping inconsistent across compared models (train_advanced omits it) |
| OPT-06 | medium | Selection metric + LR scheduler differ across the "comparison" |
| OPT-07 | medium | Fixed 30-epoch budget, two decoupled cosine cycles, no early stopping |
| OPT-08 | low | Adam+L2 (coupled) in finetune vs AdamW elsewhere |
| OPT-09 | low | Pretrained encoder selected by training loss, no pretrain val set |
| OPT-10 | low | No explicit weight init for from-scratch heads on a deep residual encoder |

### Track C — Brief-gap (what to build for the coursework)
| ID | Sev | Title |
|----|-----|-------|
| GAP-00 | critical gate | Data gitignored/absent, B3DB has no downloader, shipped checkpoints are leakage-trained |
| GAP-1 | foundational | Unified metrics module (logic duplicated across 6+ scripts, inconsistent) |
| GAP-2 | build | Exp2 InChIKey dedup (primitive exists, no dedup) |
| GAP-3 | build | Exp1 scaffold + cluster splitters + >=5-seed loop (only random exists) |
| GAP-4 | build | ECFP LogReg + RandomForest baselines (only a toy RF regressor exists) |
| GAP-5 | build | Exp3 ablation grid (toggles exist, no driver; edge path is dead) |
| GAP-6 | build | Exp4 calibration: Platt + temperature, reliability, ECE (absent) |
| GAP-7 | build | Single Jupyter notebook deliverable (none exists) |

---

## Track A — Leakage / evaluation rigour

### LEAK-01 — B3DB "external validation" set is 100% inside the training data
- Severity: critical | validity: external | leakage-type: K&N L1 (no clean train/test separation: test set used in training)
- What the code does: `bbb_stereo_v2.py:405-455` `load_training_data()` reads the entire `data/B3DB_classification.tsv` (all ~7807 rows) plus all BBBP, trains 5-fold CV over the combined graphs (`:519-532`), saves `models/bbb_stereo_v2_fold{1..5}_best.pth` (`:610`). `external_validation.py:174` reloads those exact checkpoints and evaluates on the full B3DB (`:159-195`, ensemble AUC `:195`). Banner calls it "completely unseen data" (`:4-5,154`), which is false: every B3DB compound was in 4 of 5 training folds.
- Why it matters (L1 p27 "performance on unseen data"; L1 p30 "Generalization errors: error on testing data"): a test set that the model trained on measures memorisation, not generalisation. The 0.9612 is an upper-bound in-sample score.
- Fix: hold B3DB out entirely (train on BBBP only) or carve a disjoint B3DB test split before training; re-run external eval only on compounds absent from every fold's training set; delete the "unseen data" wording.
- Acceptance: assert `test_idx ∩ (train_idx ∪ val_idx) == ∅`; report AUC only on the held-out remainder.
- Report line: "The reported external ROC-AUC of 0.9612 is an in-sample figure: the full B3DB set was included in 5-fold training, so it measures memorisation rather than generalisation."

### LEAK-02 — No InChIKey/canonical dedup between BBBP-train and B3DB
- Severity: critical | validity: external | leakage-type: K&N L1.3 (duplicates across train/test)
- What the code does: `load_training_data()` (`bbb_stereo_v2.py:412-455`) concatenates B3DB then BBBP with no overlap check, no `MolToInchiKey`, no `drop_duplicates`. The only InChIKey/canonical code (`pubchemqc_integration.py:76-98`) is for PubChemQC DFT lookup, never dedup. B3DB is a documented aggregation that includes BBBP-derived compounds.
- Why it matters (L1 p30 train/test separation): even ignoring LEAK-01, a BBBP-trained-then-B3DB-tested number is inflated because much of B3DB *is* the BBBP training compounds (and their stereoisomer/SMILES variants).
- Fix: canonicalise both sets via `Chem.MolToInchiKey`; drop every test InChIKey present in train; report overlap count; recompute AUC on the genuinely unseen remainder; also `drop_duplicates` within the combined training set.
- Acceptance: overlap count printed (expected non-zero); no filtered-test InChIKey appears in train.
- Report line: "Because B3DB aggregates BBBP, an InChIKey overlap of N compounds existed between the training and external sets; after removal the external AUC fell from X to Y."
- Sub-task (added 28 Jun): also compute the **true ZINC-pretraining-to-test overlap** (InChIKey intersection of the ~250K ZINC pretraining molecules with the test set), so the soft pretraining-leakage figure is measured, not assumed. A "22%" figure was floating around with no provenance (not in code, not computed by anyone) and must be replaced by a computed number. Reuse `dedup.inchikey()`; report `n_zinc, n_test, n_overlap, pct` alongside the BBBP-to-B3DB overlap.

### LEAK-03 — Reported CV AUC selects best epoch on the same val fold it reports
- Severity: high | validity: internal | leakage-type: K&N L1 (selection on the evaluation set)
- What the code does: `bbb_stereo_v2.py:606-610` saves the checkpoint whenever val AUC improves; `:616` records `best_val_auc` (running max over 40 epochs) as the fold result; `:620-636` reload that checkpoint and re-evaluate on the same `val_loader`. `finetune_bbb_stereo.py:227-276` identical. No test split exists in either path. The cited 0.8968 CV figure comes from this.
- Why it matters (L1 p30: validation = model selection; test = the final *unbiased* evaluation): using one set for both selection and reporting is double-dipping; max-over-epochs of a noisy metric is upward-biased.
- Fix: nested splits, outer held-out test never touched in selection; pick best epoch on val, report once on the untouched test. At minimum report val AUC at a fixed epoch (not max-over-epochs) plus a true test partition.
- Acceptance: selection indices and reporting indices are provably disjoint.
- Report line: "Reported CV AUC was the maximum validation AUC over epochs on the same fold used for checkpoint selection; under a held-out test split it dropped to Z."

### LEAK-04 — Random splits over post-featurisation graph lists; no scaffold/cluster, no dedup
- Severity: high | validity: external | leakage-type: K&N L3.1 (non-independence: near-duplicate analogues across train/test)
- What the code does: `run_full_comparison_v3.py:307-308` `train_test_split(0.2,42)` then `(0.5,42)`; `bbb_stereo_v2.py:520-532` `StratifiedKFold(shuffle=True, random_state=42)` on graphs, stratified by label only. Split runs on PyG graphs after conversion, with no Bemis-Murcko grouping and no dedup. No scaffold/cluster split exists anywhere.
- Why it matters (L1 p30 unseen-data; L2 variance p4): on a small congeneric set, random splitting puts analogues and the same molecule (different SMILES/stereo) on both sides, inflating AUC.
- Fix: dedup by InChIKey, then scaffold split (RDKit MurckoScaffold) or Butina cluster split so whole scaffold groups stay on one side; report random vs scaffold gap.
- Acceptance: zero scaffold-group overlap across train/test under scaffold split.
- Report line: "Replacing the random split with a scaffold split reduced ROC-AUC by Δ, quantifying the inflation attributable to scaffold leakage."

### LEAK-05 — "Beats SOTA 8/8" is a cross-regime artefact
- Severity: high | validity: conclusion / construct | leakage-type: n/a (benchmarking validity)
- What the code does: `benchmark_competitors.py:143-152` hard-codes our V2 AUC 0.9612 as "SOTA"; competitor entries (`:120-128`) are values copied from papers on different datasets/splits; `:271-281` claim "outperforms 8/8" while the caveats admit competitors were not re-run and datasets differ. No competitor re-run, no matched split.
- Why it matters (L1 p27 evaluation; benchmark validity): pitting a leaked in-sample number against others' held-out published numbers is not a head-to-head; the +5.6% margin is meaningless.
- Fix: remove the ranking until 0.9612 is replaced by a leakage-controlled, scaffold-split, dedup'd number; only compare against competitors re-run on the identical split, or present competitor numbers as context, not a leaderboard.
- Acceptance: no "SOTA"/"beats N/N" claim survives that is not on a matched, leakage-free split.
- Report line: "The earlier 'state-of-the-art' comparison was a cross-regime artefact; under matched, leakage-free evaluation the comparison is not supported."

### LEAK-06 — Fabricated logBB targets inflate regression R2/RMSE
- Severity: medium | validity: construct | leakage-type: n/a (target leakage from the label)
- What the code does: `bbb_stereo_v2.py:447-448` sets logBB = 0.3 (BBB+) / -1.5 (BBB-) for all BBBP rows from the binary label; `:428` sets 0.5/-1.5 for B3DB rows lacking logBB. These become the MSE target (`:569`); R2/RMSE computed against them (`:602-603`).
- Why it matters (L1 p27 "types of errors"): logBB is the documented primary task (loss weight 1.0) yet for most rows it is a deterministic function of the classification label, so reported R2/RMSE measure recovery of injected constants.
- Fix: train/report logBB only on rows with measured logBB (`pd.notna`); for binary-only data train pure BCE; never derive the regression target from the jointly-trained label.
- Acceptance: regression metrics computed only on measured-logBB rows, N reported.
- Report line: "Regression metrics were recomputed on the measured-logBB subset only (N=…); the fabricated label-derived targets were excluded from loss and reporting."

---

## Track B — Optimisation / training (Lecture 3)

### OPT-03 — No RNG seeding anywhere
- Severity: high | validity: conclusion (reproducibility)
- What the code does: no `torch.manual_seed`/`np.random.seed`/`random.seed`/`cudnn.deterministic` anywhere (grep-negative). Only data splits are seeded (`random_state=42`). Init, DataLoader shuffle order, dropout masks are unseeded.
- Why it matters (L2 p2-6 bias-variance; variance p4 = sensitivity to the training sample): the reported mean±std across folds absorbs random-init/shuffle/dropout variance, so it overstates reliability and is not repeatable.
- Fix: `seed_everything(seed)` at each entry point (torch/numpy/random/cuda + cudnn.deterministic + DataLoader generator/worker_init_fn); repeat each fold over k seeds to separate fold from init variance.
- Acceptance: two runs with the same seed produce identical metrics.
- Report line: "All experiments fixed torch/NumPy/Python/CUDA seeds; variance is reported over ≥5 seeds so fold variance is separated from initialisation noise."

### OPT-04 — Class imbalance unhandled in the stereo path; raw accuracy led
- Severity: high | validity: construct / conclusion
- What the code does: BBBP is ~76.8% positive (`train_advanced.py:164-171`). Stereo finetune uses plain `BCEWithLogitsLoss()` no pos_weight (`finetune_bbb_stereo.py:207`; `bbb_stereo_v2.py:549`) while comparators use `pos_weight=3.24` (`train_advanced.py:173`, `run_full_comparison_v3.py:170`, `train_pretrained_finetune.py:65`). Finetune reports raw accuracy at threshold 0.5 (`:266-273`).
- Why it matters (L1 p29: accuracy misleading on imbalance, F1 preferred; PR-AUC sensitive to positive rate): an all-positive classifier scores 0.77 accuracy; leading with it overstates competence, and the stereo model is compared unfairly against imbalance-corrected baselines.
- Fix: pick one protocol (pos_weight everywhere or nowhere); lead with ROC-AUC / PR-AUC / balanced accuracy / MCC; tune the threshold on validation, not hardcoded 0.5.
- Acceptance: identical loss-weighting protocol across all compared models; accuracy not the headline.
- Report line: "Given the 77% positive rate we lead with ROC-AUC, PR-AUC, balanced accuracy and F1; the loss weighting was held identical across all compared models."

### OPT-05 — Gradient clipping inconsistent across compared models
- Severity: medium | validity: conclusion (confounded comparison)
- What the code does: `clip_grad_norm_(...,1.0)` in `finetune_bbb_stereo.py:116`, `bbb_stereo_v2.py:576`, both pretrainers, `train_gnn.py`; absent in `train_advanced.py:219-220` (the 1.38M-param comparator).
- Why it matters (L1 p25-26 gradient clipping by norm): differing gradient-stability recipes confound architecture vs optimisation; the unclipped deeper model is more exposed to exploding-gradient spikes.
- Fix: apply the same `clip_grad_norm_(model.parameters(), 1.0)` in `train_advanced.py`'s loop.
- Acceptance: every compared model shares one clipping setting.
- Report line: "Gradient clipping (max-norm 1.0) was applied uniformly so optimisation recipe does not confound the architecture comparison."

### OPT-06 — Selection metric + LR scheduler differ across the comparison
- Severity: medium | validity: conclusion
- What the code does: stereo finetune + comparisons select on val AUC with `ReduceLROnPlateau(mode='max')`, but `train_advanced.py:177-184` uses `mode='min'`+EarlyStopping on val_loss; `train_gnn.py:171` selects on val MSE; `finetune_bbb_stereo.py` uses CosineAnnealingLR with no plateau detection.
- Why it matters (L1 p30 model selection; L2 p13 early stopping): models stopped on different signals are read at different points on their AUC curves, mixing protocol effects with architecture effects.
- Fix: fix one selection criterion (val AUC) and one scheduler across all compared runs; align patience and epoch budget.
- Acceptance: one documented selection+scheduler protocol used everywhere.
- Report line: "A single selection criterion (validation ROC-AUC) and scheduler were used across all models."

### OPT-07 — Fixed 30-epoch budget, two decoupled cosine cycles, no early stopping
- Severity: medium | validity: internal
- What the code does: `finetune_bbb_stereo.py` runs 10 frozen + 20 finetune epochs with no patience stopping; each phase gets its own `CosineAnnealingLR` (T_max=10 at `:219`, T_max=20 at `:242`), so LR anneals to ~0 then jumps back up at phase 2.
- Why it matters (L2 p13 early stopping caps effective complexity by stopping before val error climbs): a fixed budget with a cosine restart is decoupled from convergence; only max-over-epochs checkpointing prevents over/under-training, which is itself the optimistic-bias source (LEAK-03).
- Fix: patience-based early stopping on a val metric with best-weight restore, or a single continuous schedule tied to a validation plateau.
- Acceptance: training stops on a validation signal, not a fixed count.
- Report line: "Early stopping on validation loss with best-weight restore replaced the fixed two-cycle budget."

### OPT-08 — Adam+L2 (coupled) in finetune vs AdamW elsewhere
- Severity: low | validity: conclusion
- What the code does: `finetune_bbb_stereo.py:214-241` `optim.Adam(..., weight_decay=...)`; `bbb_stereo_v2.py:551` and `pretrain_full_stereo.py:136-139` use AdamW. Betas never set (defaults 0.9, 0.999).
- Why it matters (L2 p10-11 L2 regularisation): coupling L2 into Adam scales the penalty by the adaptive LR (the exact problem AdamW fixes), so the encoder is regularised differently when finetuned vs pretrained.
- Fix: use AdamW consistently; keep weight_decay explicit and identical where a fair comparison is intended.
- Acceptance: one optimiser family across pretrain/finetune/v2.
- Report line: "AdamW (decoupled weight decay) was used throughout for consistent L2 regularisation."

### OPT-09 — Pretrained encoder selected by training loss, no pretrain val set
- Severity: low | validity: internal
- What the code does: `pretrain_full_stereo.py:169-204` selects the best checkpoint by epoch *training* loss and saves to `pretrained_stereo_full.pth`, which every finetune fold loads.
- Why it matters (L2 p13 / L1 p30 validation role): selecting on training loss risks shipping an overfit encoder into every fold, biasing all results in the same direction (not averaged out by CV).
- Fix: hold out a ZINC validation subset; select by val loss, or use the final-epoch encoder; document which checkpoint produced the numbers.
- Acceptance: pretrain checkpoint chosen on held-out loss.
- Report line: "The pretrained encoder was selected on a held-out ZINC validation loss."

### OPT-10 — No explicit weight init for from-scratch heads on a deep residual encoder
- Severity: low | validity: conclusion (minor)
- What the code does: no `kaiming_`/`xavier_`/`reset_parameters` anywhere; encoder stacks 4 residual GATv2 blocks + residual TransformerConv (`x = x + x_new`) with ReLU; heads use GELU; all rely on PyTorch default Linear init.
- Why it matters (L1 p20-22 Xavier for tanh/sigmoid, He for ReLU): default init plus repeated residual adds lets activation variance grow before BatchNorm rescales it; mostly absorbed by BN, so minor, but init is unseeded and the fan is ReLU-oriented for GELU heads.
- Fix: explicit Kaiming/Xavier matched to the nonlinearity; optionally zero-init the residual branch's last layer; makes init reproducible alongside seeding.
- Acceptance: init is explicit and seeded.
- Report line: "Layers were explicitly initialised (He for ReLU, Xavier for the GELU heads) for reproducibility."

---

## Track C — Brief-gap (build plan)

GAP-00 (critical gate): `data/` is gitignored/absent; `download_bbbp.py` exists but there is no B3DB downloader; shipped `bbb_stereo_v2_fold*_best.pth` were trained with B3DB inside training, so reusing them reproduces the leakage. Resolve before anything else: regenerate `data/bbbp_dataset.csv` (prefer direct S3 CSV, not the deepchem fallback which reorders rows), script/document B3DB acquisition, retrain leakage-free.

Reusable anchors confirmed live:
- metrics template: `external_validation.py:109-148`
- InChIKey/canonical primitive: `pubchemqc_integration.py:88-96`
- 21-dim stereo graph: `mol_to_graph_enhanced.py:286-389`
- train/eval loop: `finetune_bbb_stereo.py:101-148`
- ablation toggles: `bbb_stereo_v2.py:277-282,465-470` and `zinc_stereo_pretraining.py:152-187`

Build order (dependency-driven):
T0 data + leakage-free trainer [GATE] → T1 metrics.py → CPU first-slice validation →
T2 dedup/Exp2 → T3 splits+seeds/Exp1 → T4 ECFP baselines → T5 ablation/Exp3 →
T6 calibration/Exp4 → T7 notebook + IEEE tables/plots.

Proposed package: `cw/{metrics,dedup,splits,seeds,data,featurize,train,baselines,ablation,calibration,report}.py`
plus `notebooks/ECS7037P_BBB.ipynb` (orchestration only; heavy logic in `cw/`).

Must-land fixes (these ARE the contribution): exclude B3DB from training when it is the eval set;
held-out test distinct from selection set; InChIKey dedup; scaffold split alongside random;
RNG seeding + ≥5 seeds; drop/mask fabricated logBB; lead with AUC/PR-AUC/balanced-acc/MCC;
delete the SOTA leaderboard until leakage-free.

Honesty constraint for Exp3: the `edge_attr` path is dead (`model(x, edge_index, batch)` called with
no edge_attr at `bbb_stereo_v2.py:566,292`), so bond-level stereo never reaches the model. Stereo the
model actually consumes = 6 molecule-level scalars tiled to every node (`mol_to_graph_enhanced.py:338-346`),
and only `[:6]` of an 8-vec are used (r/s counts dropped, docstring mislabels them, `:223-234`).
Scope "remove stereo" to exactly those 6 scalars; do not claim bond-level chirality is learned.

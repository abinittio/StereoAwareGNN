"""
V2 Model Validation Script
Run after training to validate on B3DB external dataset
"""

import sys
import os
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix

from bbb_predictor_v2 import BBBPredictorV2, PHARMA_COMPOUNDS

def main():
    print("=" * 70)
    print("BBB PREDICTOR V2 - FULL VALIDATION")
    print("=" * 70)

    # Load predictor
    predictor = BBBPredictorV2()
    predictor.load_ensemble('models/')

    if not predictor.models:
        print("ERROR: No models found!")
        return

    # =========================================================================
    # 1. PHARMA COMPOUND VALIDATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: PHARMA-RELEVANT COMPOUNDS")
    print("=" * 70)

    for category in ['cannabinoids', 'opioids', 'benzodiazepines', 'bbb_negative']:
        compounds = PHARMA_COMPOUNDS.get(category, [])
        if not compounds:
            continue

        print(f"\n{category.upper()}:")
        print("-" * 50)

        correct = 0
        total = 0

        for smiles, name, exp_label, exp_logBB in compounds[:6]:
            try:
                r = predictor.predict(smiles, name=name)
                exp_class = 'BBB+' if exp_label == 1.0 else 'BBB-'

                # Check if prediction matches
                if r.classification == exp_class or r.classification == 'BBB+/-':
                    ok = 'OK'
                    correct += 1
                else:
                    ok = 'MISS'
                total += 1

                print(f"  [{ok:4s}] {name:15s}: LogBB={r.logBB_mean:+.2f} (exp {exp_logBB:+.2f}), {r.classification}")
            except Exception as e:
                print(f"  [ERR ] {name:15s}: {e}")

        if total > 0:
            print(f"  Accuracy: {correct}/{total} ({100*correct/total:.0f}%)")

    # =========================================================================
    # 2. B3DB EXTERNAL VALIDATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: B3DB EXTERNAL VALIDATION (7,807 compounds)")
    print("=" * 70)

    b3db_path = 'data/B3DB_classification.tsv'
    if not os.path.exists(b3db_path):
        print(f"B3DB not found at {b3db_path}")
        return

    df = pd.read_csv(b3db_path, sep='\t')
    print(f"Loaded {len(df)} compounds from B3DB")

    # Predict
    y_true = []
    y_pred_prob = []
    y_pred_logBB = []
    y_true_logBB = []

    print("\nPredicting (this may take a few minutes)...")

    for i, row in df.iterrows():
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(df)}")

        smiles = row['SMILES']
        label = 1.0 if row['BBB+/BBB-'] == 'BBB+' else 0.0
        logBB_true = row.get('logBB', None)

        try:
            result = predictor.predict(smiles, enumerate_stereo=False)  # Skip stereo enum for speed

            y_true.append(label)
            y_pred_prob.append(result.probability_mean)
            y_pred_logBB.append(result.logBB_mean)

            if pd.notna(logBB_true):
                y_true_logBB.append(float(logBB_true))
            else:
                y_true_logBB.append(None)

        except Exception as e:
            continue

    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)
    y_pred_logBB = np.array(y_pred_logBB)

    # Metrics
    auc = roc_auc_score(y_true, y_pred_prob)
    y_pred_class = (y_pred_prob > 0.5).astype(float)
    bal_acc = balanced_accuracy_score(y_true, y_pred_class)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print("\n" + "-" * 50)
    print("EXTERNAL VALIDATION RESULTS:")
    print("-" * 50)
    print(f"  Compounds evaluated: {len(y_true)}")
    print(f"  AUC:                 {auc:.4f}")
    print(f"  Balanced Accuracy:   {bal_acc:.4f}")
    print(f"  Sensitivity (BBB+):  {sensitivity:.4f} ({tp}/{tp+fn})")
    print(f"  Specificity (BBB-):  {specificity:.4f} ({tn}/{tn+fp})")

    # LogBB regression metrics (where ground truth exists)
    valid_logBB = [(pred, true) for pred, true in zip(y_pred_logBB, y_true_logBB) if true is not None]
    if valid_logBB:
        pred_vals = np.array([x[0] for x in valid_logBB])
        true_vals = np.array([x[1] for x in valid_logBB])

        from sklearn.metrics import r2_score, mean_absolute_error
        r2 = r2_score(true_vals, pred_vals)
        mae = mean_absolute_error(true_vals, pred_vals)

        print(f"\n  LogBB Regression ({len(valid_logBB)} compounds with ground truth):")
        print(f"    R²:  {r2:.4f}")
        print(f"    MAE: {mae:.4f}")

    # =========================================================================
    # 3. COMPARISON WITH V1
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 3: V1 vs V2 COMPARISON")
    print("=" * 70)

    print(f"""
    Metric              V1 (old)      V2 (new)      Change
    ─────────────────────────────────────────────────────
    CV AUC              0.8968        0.9371        +4.0%
    CV Balanced Acc     ~0.70         0.7988        +10%
    CV R² (LogBB)       N/A           0.5810        NEW

    External AUC        0.8840        {auc:.4f}        {'+' if auc > 0.884 else ''}{100*(auc-0.884)/0.884:.1f}%
    External Sens       0.9860        {sensitivity:.4f}        {'+' if sensitivity > 0.986 else ''}{100*(sensitivity-0.986)/0.986:.1f}%
    External Spec       0.4210        {specificity:.4f}        {'+' if specificity > 0.421 else ''}{100*(specificity-0.421)/0.421:.1f}%
    """)

    print("=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

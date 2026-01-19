"""
External Validation of Stereo-Aware BBB Model on B3DB Dataset

Tests our model (trained on BBBP ~2000 compounds) on B3DB (7807 compounds)
This is TRUE external validation - completely unseen data from different sources.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from torch_geometric.loader import DataLoader
import sys
from pathlib import Path

# Add path
sys.path.insert(0, str(Path(__file__).parent))

from zinc_stereo_pretraining import StereoAwareEncoder
from mol_to_graph_enhanced import mol_to_graph_enhanced


class BBBStereoClassifier(nn.Module):
    """Same architecture as training."""
    def __init__(self, encoder, hidden_dim=128):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, edge_index, batch):
        graph_embed = self.encoder(x, edge_index, batch)
        return self.classifier(graph_embed)


def load_b3db():
    """Load B3DB external test set."""
    print("Loading B3DB external dataset...")
    df = pd.read_csv('data/B3DB_classification.tsv', sep='\t')

    print(f"  Total compounds: {len(df)}")
    print(f"  BBB+: {(df['BBB+/BBB-'] == 'BBB+').sum()}")
    print(f"  BBB-: {(df['BBB+/BBB-'] == 'BBB-').sum()}")

    return df


def convert_to_graphs(df):
    """Convert B3DB to stereo-aware graphs."""
    print("\nConverting to stereo-aware graphs (21 features)...")

    graphs = []
    labels = []
    failed = 0

    for idx, row in df.iterrows():
        smiles = row['SMILES']
        label = 1.0 if row['BBB+/BBB-'] == 'BBB+' else 0.0

        graph = mol_to_graph_enhanced(
            smiles,
            y=label,
            include_quantum=False,
            include_stereo=True,
            use_dft=False
        )

        if graph is not None and graph.x.shape[1] == 21:
            graphs.append(graph)
            labels.append(label)
        else:
            failed += 1

        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx+1}/{len(df)} ({len(graphs)} valid, {failed} failed)")
            sys.stdout.flush()

    print(f"\nConversion complete: {len(graphs)}/{len(df)} valid ({failed} failed)")
    return graphs, np.array(labels)


def load_model(model_path):
    """Load trained stereo model (v2 with multi-task architecture)."""
    from bbb_stereo_v2 import BBBStereoV2Model

    encoder = StereoAwareEncoder(node_features=21, hidden_dim=128, num_layers=4)
    model = BBBStereoV2Model(encoder, hidden_dim=128)

    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    return model


def evaluate(model, graphs, labels):
    """Evaluate model on external data."""
    print("\nRunning inference...")

    loader = DataLoader(graphs, batch_size=64)
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            # BBBStereoV2Model returns (logBB, classification_prob)
            logBB, prob = model(batch.x, batch.edge_index, batch.batch)
            probs = torch.sigmoid(prob).cpu().numpy().flatten()
            all_preds.extend(probs)

    preds = np.array(all_preds)
    preds_binary = (preds > 0.5).astype(int)

    # Metrics
    auc = roc_auc_score(labels, preds)
    ap = average_precision_score(labels, preds)
    acc = accuracy_score(labels, preds_binary)
    precision = precision_score(labels, preds_binary)
    recall = recall_score(labels, preds_binary)
    f1 = f1_score(labels, preds_binary)

    cm = confusion_matrix(labels, preds_binary)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)

    return {
        'auc': auc,
        'average_precision': ap,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': preds
    }


def main():
    print("=" * 70)
    print("EXTERNAL VALIDATION: Stereo-GNN on B3DB")
    print("Model trained on BBBP (~2000) | Testing on B3DB (7807)")
    print("=" * 70)
    print()

    # Load B3DB
    df = load_b3db()

    # Convert to graphs
    graphs, labels = convert_to_graphs(df)

    # Test each fold model
    print("\n" + "=" * 60)
    print("TESTING ALL 5 FOLD MODELS")
    print("=" * 60)

    all_aucs = []
    all_accs = []
    ensemble_preds = []

    for fold in range(1, 6):
        model_path = f'models/bbb_stereo_v2_fold{fold}_best.pth'  # Use v2 models

        try:
            model = load_model(model_path)
            results = evaluate(model, graphs, labels)

            all_aucs.append(results['auc'])
            all_accs.append(results['accuracy'])
            ensemble_preds.append(results['predictions'])

            print(f"\nFold {fold}: AUC={results['auc']:.4f} | Acc={results['accuracy']:.4f} | "
                  f"Prec={results['precision']:.4f} | Rec={results['recall']:.4f}")

        except FileNotFoundError:
            print(f"\nFold {fold}: Model not found")
        except Exception as e:
            print(f"\nFold {fold}: Error - {e}")

    # Ensemble (average predictions)
    if len(ensemble_preds) > 0:
        ensemble_avg = np.mean(ensemble_preds, axis=0)
        ensemble_auc = roc_auc_score(labels, ensemble_avg)
        ensemble_binary = (ensemble_avg > 0.5).astype(int)
        ensemble_acc = accuracy_score(labels, ensemble_binary)
        ensemble_f1 = f1_score(labels, ensemble_binary)

        print("\n" + "=" * 60)
        print("FINAL RESULTS ON B3DB (EXTERNAL VALIDATION)")
        print("=" * 60)
        print(f"\nPer-fold AUCs: {[f'{a:.4f}' for a in all_aucs]}")
        print(f"Mean AUC:      {np.mean(all_aucs):.4f} +/- {np.std(all_aucs):.4f}")
        print(f"Mean Accuracy: {np.mean(all_accs):.4f} +/- {np.std(all_accs):.4f}")
        print()
        print(f"ENSEMBLE (5-model average):")
        print(f"  AUC:      {ensemble_auc:.4f}")
        print(f"  Accuracy: {ensemble_acc:.4f}")
        print(f"  F1:       {ensemble_f1:.4f}")

        # Confusion matrix for ensemble
        cm = confusion_matrix(labels, ensemble_binary)
        tn, fp, fn, tp = cm.ravel()
        print(f"\nConfusion Matrix:")
        print(f"  TP={tp}, FP={fp}")
        print(f"  FN={fn}, TN={tn}")
        print(f"  Sensitivity: {tp/(tp+fn):.4f}")
        print(f"  Specificity: {tn/(tn+fp):.4f}")

        # Compare to training performance
        print("\n" + "-" * 40)
        print("COMPARISON")
        print("-" * 40)
        print(f"Training (BBBP, 5-fold CV):  AUC = 0.8968")
        print(f"External (B3DB, 7807 mols):  AUC = {ensemble_auc:.4f}")

        diff = ensemble_auc - 0.8968
        if diff >= 0:
            print(f"\nGeneralization: +{diff*100:.2f}% (EXCELLENT)")
        elif diff > -0.05:
            print(f"\nGeneralization: {diff*100:.2f}% (GOOD - minimal drop)")
        else:
            print(f"\nGeneralization: {diff*100:.2f}% (model may be overfit)")


if __name__ == "__main__":
    main()

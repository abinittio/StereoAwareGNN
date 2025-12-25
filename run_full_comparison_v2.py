"""
Run Full Model Comparison V2

Compares FOUR model variants:
1. Baseline: Standard model trained from scratch on BBBP (15 features)
2. Pretrained: Model pretrained on ZINC 250k, then fine-tuned on BBBP (15 features)
3. Quantum: Model with quantum descriptors trained on BBBP (28 features)
4. Pretrained + Quantum: Pretrained on ZINC 250k + quantum descriptors (28 features)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import os
import sys

from advanced_bbb_model import AdvancedHybridBBBNet
from advanced_bbb_model_quantum import AdvancedHybridBBBNetQuantum
from mol_to_graph import batch_smiles_to_graphs
from mol_to_graph_quantum import batch_smiles_to_graphs_quantum


def load_bbbp_data(include_quantum=False):
    """Load BBBP dataset and convert to graphs"""
    print(f"Loading BBBP dataset (quantum={include_quantum})...")
    sys.stdout.flush()

    data_path = "data/bbbp_dataset.csv"
    df = pd.read_csv(data_path)

    smiles_col = 'smiles' if 'smiles' in df.columns else 'SMILES'
    target_col = 'p_np' if 'p_np' in df.columns else 'BBB_permeability'

    smiles_list = df[smiles_col].tolist()
    y_list = df[target_col].tolist()

    print(f"Total samples: {len(smiles_list)}")
    sys.stdout.flush()

    if include_quantum:
        graphs = batch_smiles_to_graphs_quantum(smiles_list, y_list, include_quantum=True)
    else:
        graphs = batch_smiles_to_graphs(smiles_list, y_list)

    print(f"Valid graphs: {len(graphs)}")
    print(f"Features per node: {graphs[0].x.shape[1]}")
    sys.stdout.flush()

    return graphs


def train_model(model, train_loader, val_loader, epochs=100, lr=0.0001,
                patience=30, device='cpu', class_weight=3.24, model_name='model'):
    """Train the model with early stopping"""
    model = model.to(device)

    pos_weight = torch.tensor([class_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                      patience=10, min_lr=1e-6)

    best_auc = 0
    best_epoch = 0
    no_improve = 0

    print(f"\nTraining {model_name}...")
    print("=" * 60)
    sys.stdout.flush()

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_preds = []
        train_labels = []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch.x, batch.edge_index, batch.batch)
            out_flat = out.view(-1)
            y_flat = batch.y.view(-1)

            loss = criterion(out_flat, y_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_preds.extend(torch.sigmoid(out_flat).detach().cpu().numpy())
            train_labels.extend(y_flat.cpu().numpy())

        train_auc = roc_auc_score(train_labels, train_preds)

        # Validation
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                out_flat = out.view(-1)
                y_flat = batch.y.view(-1)
                val_preds.extend(torch.sigmoid(out_flat).cpu().numpy())
                val_labels.extend(y_flat.cpu().numpy())

        val_auc = roc_auc_score(val_labels, val_preds)
        val_acc = accuracy_score(val_labels, [1 if p > 0.5 else 0 for p in val_preds])

        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]['lr']

        # Check for improvement
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            no_improve = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_auc': val_auc,
                'val_accuracy': val_acc,
            }, f'models/best_{model_name}.pth')

            print(f"Epoch {epoch}/{epochs} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | "
                  f"Val Acc: {val_acc*100:.1f}% | LR: {current_lr:.6f} | *BEST*")
            sys.stdout.flush()
        else:
            no_improve += 1
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | "
                      f"Val Acc: {val_acc*100:.1f}% | LR: {current_lr:.6f}")
                sys.stdout.flush()

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}. Best AUC: {best_auc:.4f} at epoch {best_epoch}")
            sys.stdout.flush()
            break

    # Load best model
    checkpoint = torch.load(f'models/best_{model_name}.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, best_auc


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model on test set and return predictions"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            preds = torch.sigmoid(out).view(-1).cpu().numpy()
            labels = batch.y.view(-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    binary_preds = [1 if p > 0.5 else 0 for p in all_preds]

    metrics = {
        'auc': roc_auc_score(all_labels, all_preds),
        'accuracy': accuracy_score(all_labels, binary_preds),
        'precision': precision_score(all_labels, binary_preds),
        'recall': recall_score(all_labels, binary_preds),
        'f1': f1_score(all_labels, binary_preds),
        'predictions': all_preds,
        'labels': all_labels,
        'binary_preds': binary_preds,
    }

    return metrics


def transfer_pretrained_weights(model, pretrained_path, model_type='standard'):
    """Transfer pretrained weights to model"""
    print(f"Loading pretrained weights from {pretrained_path}...")
    sys.stdout.flush()

    checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    pretrained_dict = checkpoint['model_state_dict']
    model_dict = model.state_dict()

    transferred = 0
    skipped = 0

    for name, param in pretrained_dict.items():
        if name in model_dict:
            if model_dict[name].shape == param.shape:
                model_dict[name] = param
                transferred += 1
            else:
                # For quantum model, only the first layer differs in size
                skipped += 1
        else:
            skipped += 1

    model.load_state_dict(model_dict)
    print(f"Transferred {transferred} layers, skipped {skipped} layers")
    sys.stdout.flush()

    return model


def main():
    """Run full 4-model comparison"""

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.makedirs('models', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    sys.stdout.flush()

    # Training parameters
    epochs = 150
    patience = 40
    batch_size = 32
    class_weight = 3.24

    results = {}

    # ========================================
    # Load Standard BBBP Data (15 features)
    # ========================================
    print("\n" + "=" * 70)
    print("LOADING BBBP DATA (STANDARD - 15 features)")
    print("=" * 70)
    sys.stdout.flush()

    graphs_standard = load_bbbp_data(include_quantum=False)

    train_graphs, temp_graphs = train_test_split(graphs_standard, test_size=0.2, random_state=42)
    val_graphs, test_graphs = train_test_split(temp_graphs, test_size=0.5, random_state=42)

    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")
    sys.stdout.flush()

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size)
    test_loader = DataLoader(test_graphs, batch_size=batch_size)

    # ========================================
    # Model 1: Baseline (no pretraining)
    # ========================================
    print("\n" + "=" * 70)
    print("MODEL 1: BASELINE (no pretraining, 15 features)")
    print("=" * 70)
    sys.stdout.flush()

    model_baseline = AdvancedHybridBBBNet(
        num_node_features=15,
        hidden_channels=128,
        num_heads=8,
        dropout=0.3
    )

    model_baseline, best_auc_baseline = train_model(
        model_baseline, train_loader, val_loader,
        epochs=epochs, lr=0.0001, patience=patience,
        device=device, class_weight=class_weight,
        model_name='baseline_v2'
    )

    metrics_baseline = evaluate_model(model_baseline, test_loader, device)
    results['baseline'] = {
        'best_val_auc': best_auc_baseline,
        'test_metrics': metrics_baseline
    }

    print(f"\nBaseline Test Results: AUC={metrics_baseline['auc']:.4f}, Acc={metrics_baseline['accuracy']*100:.1f}%")
    sys.stdout.flush()

    # Save intermediate results after Model 1
    np.save('models/full_comparison_results_v2.npy', results)
    print("Checkpoint saved: Model 1 complete")
    sys.stdout.flush()

    # ========================================
    # Model 2: Pretrained + Fine-tuned
    # ========================================
    print("\n" + "=" * 70)
    print("MODEL 2: PRETRAINED (ZINC 250k, 15 features)")
    print("=" * 70)
    sys.stdout.flush()

    pretrained_path = 'models/pretrained_encoder.pth'

    model_pretrained = AdvancedHybridBBBNet(
        num_node_features=15,
        hidden_channels=128,
        num_heads=8,
        dropout=0.3
    )

    model_pretrained = transfer_pretrained_weights(model_pretrained, pretrained_path)

    model_pretrained, best_auc_pretrained = train_model(
        model_pretrained, train_loader, val_loader,
        epochs=epochs, lr=0.00005, patience=patience,
        device=device, class_weight=class_weight,
        model_name='pretrained_v2'
    )

    metrics_pretrained = evaluate_model(model_pretrained, test_loader, device)
    results['pretrained'] = {
        'best_val_auc': best_auc_pretrained,
        'test_metrics': metrics_pretrained
    }

    print(f"\nPretrained Test Results: AUC={metrics_pretrained['auc']:.4f}, Acc={metrics_pretrained['accuracy']*100:.1f}%")
    sys.stdout.flush()

    # Save intermediate results after Model 2
    np.save('models/full_comparison_results_v2.npy', results)
    print("Checkpoint saved: Model 2 complete")
    sys.stdout.flush()

    # ========================================
    # Load Quantum BBBP Data (28 features)
    # ========================================
    print("\n" + "=" * 70)
    print("LOADING BBBP DATA (QUANTUM - 28 features)")
    print("=" * 70)
    sys.stdout.flush()

    graphs_quantum = load_bbbp_data(include_quantum=True)

    train_graphs_q, temp_graphs_q = train_test_split(graphs_quantum, test_size=0.2, random_state=42)
    val_graphs_q, test_graphs_q = train_test_split(temp_graphs_q, test_size=0.5, random_state=42)

    train_loader_q = DataLoader(train_graphs_q, batch_size=batch_size, shuffle=True)
    val_loader_q = DataLoader(val_graphs_q, batch_size=batch_size)
    test_loader_q = DataLoader(test_graphs_q, batch_size=batch_size)

    # ========================================
    # Model 3: Quantum-Enhanced (no pretraining)
    # ========================================
    print("\n" + "=" * 70)
    print("MODEL 3: QUANTUM-ONLY (28 features, no pretraining)")
    print("=" * 70)
    sys.stdout.flush()

    model_quantum = AdvancedHybridBBBNetQuantum(
        num_node_features=28,
        hidden_channels=128,
        num_heads=8,
        dropout=0.3
    )

    model_quantum, best_auc_quantum = train_model(
        model_quantum, train_loader_q, val_loader_q,
        epochs=epochs, lr=0.0001, patience=patience,
        device=device, class_weight=class_weight,
        model_name='quantum_v2'
    )

    metrics_quantum = evaluate_model(model_quantum, test_loader_q, device)
    results['quantum'] = {
        'best_val_auc': best_auc_quantum,
        'test_metrics': metrics_quantum
    }

    print(f"\nQuantum Test Results: AUC={metrics_quantum['auc']:.4f}, Acc={metrics_quantum['accuracy']*100:.1f}%")
    sys.stdout.flush()

    # Save intermediate results after Model 3
    np.save('models/full_comparison_results_v2.npy', results)
    print("Checkpoint saved: Model 3 complete")
    sys.stdout.flush()

    # ========================================
    # Model 4: Pretrained + Quantum (COMBINED)
    # ========================================
    print("\n" + "=" * 70)
    print("MODEL 4: PRETRAINED + QUANTUM (ZINC 250k pretraining + 28 features)")
    print("=" * 70)
    sys.stdout.flush()

    model_combined = AdvancedHybridBBBNetQuantum(
        num_node_features=28,
        hidden_channels=128,
        num_heads=8,
        dropout=0.3
    )

    # Transfer pretrained weights (will skip first layer due to size mismatch)
    model_combined = transfer_pretrained_weights(model_combined, pretrained_path, model_type='quantum')

    model_combined, best_auc_combined = train_model(
        model_combined, train_loader_q, val_loader_q,
        epochs=epochs, lr=0.00005, patience=patience,
        device=device, class_weight=class_weight,
        model_name='combined_v2'
    )

    metrics_combined = evaluate_model(model_combined, test_loader_q, device)
    results['pretrained_quantum'] = {
        'best_val_auc': best_auc_combined,
        'test_metrics': metrics_combined
    }

    print(f"\nCombined Test Results: AUC={metrics_combined['auc']:.4f}, Acc={metrics_combined['accuracy']*100:.1f}%")
    sys.stdout.flush()

    # ========================================
    # Final Comparison
    # ========================================
    print("\n" + "=" * 70)
    print("FINAL 4-MODEL COMPARISON SUMMARY")
    print("=" * 70)
    sys.stdout.flush()

    print("\n{:<30} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "Model", "Test AUC", "Accuracy", "Precision", "Recall", "F1"))
    print("-" * 80)

    baseline_auc = results['baseline']['test_metrics']['auc']

    for name, data in results.items():
        m = data['test_metrics']
        improvement = ((m['auc'] - baseline_auc) / baseline_auc) * 100

        print("{:<30} {:<10.4f} {:<10.1%} {:<10.4f} {:<10.4f} {:<10.4f} ({:+.1f}%)".format(
            name.replace('_', ' ').title(),
            m['auc'], m['accuracy'], m['precision'], m['recall'], m['f1'],
            improvement
        ))

    print("-" * 80)

    # Determine winners in different categories
    print("\n" + "=" * 70)
    print("CATEGORY WINNERS")
    print("=" * 70)

    # Best AUC
    best_auc_model = max(results.items(), key=lambda x: x[1]['test_metrics']['auc'])
    print(f"\nBest Overall (AUC): {best_auc_model[0].upper()}")
    print(f"  AUC: {best_auc_model[1]['test_metrics']['auc']:.4f}")

    # Best Recall
    best_recall_model = max(results.items(), key=lambda x: x[1]['test_metrics']['recall'])
    print(f"\nBest Recall (finds most BBB+ compounds): {best_recall_model[0].upper()}")
    print(f"  Recall: {best_recall_model[1]['test_metrics']['recall']:.4f}")

    # Best Precision
    best_precision_model = max(results.items(), key=lambda x: x[1]['test_metrics']['precision'])
    print(f"\nBest Precision (fewest false positives): {best_precision_model[0].upper()}")
    print(f"  Precision: {best_precision_model[1]['test_metrics']['precision']:.4f}")

    # Save results
    np.save('models/full_comparison_results_v2.npy', results)
    print("\n\nResults saved to models/full_comparison_results_v2.npy")
    sys.stdout.flush()

    return results


if __name__ == "__main__":
    main()

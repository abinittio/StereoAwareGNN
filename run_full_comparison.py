"""
Run Full Model Comparison

This script compares three model variants:
1. Baseline: Standard model trained from scratch on BBBP
2. Pretrained: Model pretrained on ZINC 250k, then fine-tuned on BBBP
3. Quantum: Model with quantum descriptors trained on BBBP

Run this after pretraining on ZINC 250k is complete.
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
import time

from advanced_bbb_model import AdvancedHybridBBBNet
from advanced_bbb_model_quantum import AdvancedHybridBBBNetQuantum
from mol_to_graph import mol_to_graph, batch_smiles_to_graphs
from mol_to_graph_quantum import batch_smiles_to_graphs_quantum


def load_bbbp_data(include_quantum=False):
    """Load BBBP dataset and convert to graphs"""
    print(f"Loading BBBP dataset (quantum={include_quantum})...")

    data_path = "data/bbbp_dataset.csv"
    df = pd.read_csv(data_path)

    smiles_col = 'smiles' if 'smiles' in df.columns else 'SMILES'
    target_col = 'p_np' if 'p_np' in df.columns else 'BBB_permeability'

    smiles_list = df[smiles_col].tolist()
    y_list = df[target_col].tolist()

    print(f"Total samples: {len(smiles_list)}")

    if include_quantum:
        graphs = batch_smiles_to_graphs_quantum(smiles_list, y_list, include_quantum=True)
    else:
        graphs = batch_smiles_to_graphs(smiles_list, y_list)

    print(f"Valid graphs: {len(graphs)}")
    print(f"Features per node: {graphs[0].x.shape[1]}")

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

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0
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

            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(out_flat).detach().cpu().numpy())
            train_labels.extend(y_flat.cpu().numpy())

        train_loss /= len(train_loader)
        train_auc = roc_auc_score(train_labels, train_preds)

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                out_flat = out.view(-1)
                y_flat = batch.y.view(-1)

                loss = criterion(out_flat, y_flat)
                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(out_flat).cpu().numpy())
                val_labels.extend(y_flat.cpu().numpy())

        val_loss /= len(val_loader)
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
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'val_accuracy': val_acc,
            }, f'models/best_{model_name}.pth')

            print(f"Epoch {epoch}/{epochs} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | "
                  f"Val Acc: {val_acc*100:.1f}% | LR: {current_lr:.6f} | *BEST*")
        else:
            no_improve += 1
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | "
                      f"Val Acc: {val_acc*100:.1f}% | LR: {current_lr:.6f}")

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}. Best AUC: {best_auc:.4f} at epoch {best_epoch}")
            break

    return model, best_auc


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model on test set"""
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
    }

    return metrics


def main():
    """Run full comparison"""

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.makedirs('models', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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

    graphs_standard = load_bbbp_data(include_quantum=False)

    # Split data
    train_graphs, temp_graphs = train_test_split(graphs_standard, test_size=0.2, random_state=42)
    val_graphs, test_graphs = train_test_split(temp_graphs, test_size=0.5, random_state=42)

    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size)
    test_loader = DataLoader(test_graphs, batch_size=batch_size)

    # ========================================
    # Model 1: Baseline (no pretraining)
    # ========================================
    print("\n" + "=" * 70)
    print("MODEL 1: BASELINE (no pretraining)")
    print("=" * 70)

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
        model_name='baseline_comparison'
    )

    metrics_baseline = evaluate_model(model_baseline, test_loader, device)
    results['baseline'] = {
        'best_val_auc': best_auc_baseline,
        'test_metrics': metrics_baseline
    }

    print(f"\nBaseline Test Results:")
    print(f"  AUC: {metrics_baseline['auc']:.4f}")
    print(f"  Accuracy: {metrics_baseline['accuracy']*100:.2f}%")
    print(f"  F1: {metrics_baseline['f1']:.4f}")

    # ========================================
    # Model 2: Pretrained + Fine-tuned
    # ========================================
    print("\n" + "=" * 70)
    print("MODEL 2: PRETRAINED (ZINC 250k) + FINE-TUNED")
    print("=" * 70)

    pretrained_path = 'models/pretrained_encoder.pth'
    if not os.path.exists(pretrained_path):
        print(f"WARNING: Pretrained weights not found at {pretrained_path}")
        print("Skipping pretrained model. Run pretrain_zinc.py first!")
        results['pretrained'] = None
    else:
        model_pretrained = AdvancedHybridBBBNet(
            num_node_features=15,
            hidden_channels=128,
            num_heads=8,
            dropout=0.3
        )

        # Transfer pretrained weights
        print("Transferring pretrained weights...")
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = model_pretrained.state_dict()

        transferred = 0
        for name, param in pretrained_dict.items():
            if name in model_dict and model_dict[name].shape == param.shape:
                model_dict[name] = param
                transferred += 1

        model_pretrained.load_state_dict(model_dict)
        print(f"Transferred {transferred} layers from pretrained model")

        model_pretrained, best_auc_pretrained = train_model(
            model_pretrained, train_loader, val_loader,
            epochs=epochs, lr=0.00005, patience=patience,  # Lower LR for fine-tuning
            device=device, class_weight=class_weight,
            model_name='pretrained_comparison'
        )

        metrics_pretrained = evaluate_model(model_pretrained, test_loader, device)
        results['pretrained'] = {
            'best_val_auc': best_auc_pretrained,
            'test_metrics': metrics_pretrained
        }

        print(f"\nPretrained Test Results:")
        print(f"  AUC: {metrics_pretrained['auc']:.4f}")
        print(f"  Accuracy: {metrics_pretrained['accuracy']*100:.2f}%")
        print(f"  F1: {metrics_pretrained['f1']:.4f}")

    # ========================================
    # Load Quantum BBBP Data (28 features)
    # ========================================
    print("\n" + "=" * 70)
    print("LOADING BBBP DATA (QUANTUM - 28 features)")
    print("=" * 70)

    graphs_quantum = load_bbbp_data(include_quantum=True)

    train_graphs_q, temp_graphs_q = train_test_split(graphs_quantum, test_size=0.2, random_state=42)
    val_graphs_q, test_graphs_q = train_test_split(temp_graphs_q, test_size=0.5, random_state=42)

    train_loader_q = DataLoader(train_graphs_q, batch_size=batch_size, shuffle=True)
    val_loader_q = DataLoader(val_graphs_q, batch_size=batch_size)
    test_loader_q = DataLoader(test_graphs_q, batch_size=batch_size)

    # ========================================
    # Model 3: Quantum-Enhanced
    # ========================================
    print("\n" + "=" * 70)
    print("MODEL 3: QUANTUM-ENHANCED (28 features)")
    print("=" * 70)

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
        model_name='quantum_comparison'
    )

    metrics_quantum = evaluate_model(model_quantum, test_loader_q, device)
    results['quantum'] = {
        'best_val_auc': best_auc_quantum,
        'test_metrics': metrics_quantum
    }

    print(f"\nQuantum Test Results:")
    print(f"  AUC: {metrics_quantum['auc']:.4f}")
    print(f"  Accuracy: {metrics_quantum['accuracy']*100:.2f}%")
    print(f"  F1: {metrics_quantum['f1']:.4f}")

    # ========================================
    # Final Comparison
    # ========================================
    print("\n" + "=" * 70)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 70)

    print("\n{:<25} {:<12} {:<12} {:<12} {:<12}".format(
        "Model", "Val AUC", "Test AUC", "Test Acc", "Test F1"))
    print("-" * 70)

    # Baseline
    print("{:<25} {:<12.4f} {:<12.4f} {:<12.2%} {:<12.4f}".format(
        "Baseline (15 features)",
        results['baseline']['best_val_auc'],
        results['baseline']['test_metrics']['auc'],
        results['baseline']['test_metrics']['accuracy'],
        results['baseline']['test_metrics']['f1']
    ))

    # Pretrained
    if results['pretrained'] is not None:
        print("{:<25} {:<12.4f} {:<12.4f} {:<12.2%} {:<12.4f}".format(
            "Pretrained+Fine-tuned",
            results['pretrained']['best_val_auc'],
            results['pretrained']['test_metrics']['auc'],
            results['pretrained']['test_metrics']['accuracy'],
            results['pretrained']['test_metrics']['f1']
        ))

        # Calculate improvement from pretraining
        baseline_auc = results['baseline']['test_metrics']['auc']
        pretrained_auc = results['pretrained']['test_metrics']['auc']
        improvement_pretrain = (pretrained_auc - baseline_auc) / baseline_auc * 100
        print(f"  -> Improvement from pretraining: {improvement_pretrain:+.2f}%")
    else:
        print("{:<25} {:<12} {:<12} {:<12} {:<12}".format(
            "Pretrained+Fine-tuned", "N/A", "N/A", "N/A", "N/A"
        ))

    # Quantum
    print("{:<25} {:<12.4f} {:<12.4f} {:<12.2%} {:<12.4f}".format(
        "Quantum (28 features)",
        results['quantum']['best_val_auc'],
        results['quantum']['test_metrics']['auc'],
        results['quantum']['test_metrics']['accuracy'],
        results['quantum']['test_metrics']['f1']
    ))

    # Calculate improvement from quantum
    baseline_auc = results['baseline']['test_metrics']['auc']
    quantum_auc = results['quantum']['test_metrics']['auc']
    improvement_quantum = (quantum_auc - baseline_auc) / baseline_auc * 100
    print(f"  -> Improvement from quantum descriptors: {improvement_quantum:+.2f}%")

    print("\n" + "=" * 70)

    # Determine winner
    best_model = max(results.items(),
                     key=lambda x: x[1]['test_metrics']['auc'] if x[1] is not None else 0)
    print(f"\nBEST MODEL: {best_model[0].upper()}")
    print(f"  Test AUC: {best_model[1]['test_metrics']['auc']:.4f}")
    print(f"  Test Accuracy: {best_model[1]['test_metrics']['accuracy']*100:.2f}%")

    # Save results
    np.save('models/full_comparison_results.npy', results)
    print("\nResults saved to models/full_comparison_results.npy")

    return results


if __name__ == "__main__":
    main()

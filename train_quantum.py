"""
Train the Quantum-Enhanced BBB GNN Model

This script trains the model with quantum descriptors and compares
performance against the baseline model.

Three model variants:
1. Baseline: 15 node features (no quantum)
2. Pretrained: 15 node features with ZINC pretraining
3. Quantum: 28 node features (15 atomic + 13 quantum)
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
from tqdm import tqdm

from advanced_bbb_model_quantum import AdvancedHybridBBBNetQuantum
from mol_to_graph_quantum import mol_to_graph_quantum, batch_smiles_to_graphs_quantum
from mol_to_graph import mol_to_graph


def load_bbbp_data(include_quantum=True):
    """Load BBBP dataset and convert to graphs"""

    print(f"Loading BBBP dataset (quantum={include_quantum})...")

    # Load data
    data_path = "data/BBBP.csv"
    df = pd.read_csv(data_path)

    # Get columns
    smiles_col = 'smiles' if 'smiles' in df.columns else 'SMILES'
    target_col = 'p_np' if 'p_np' in df.columns else 'BBB_permeability'

    smiles_list = df[smiles_col].tolist()
    y_list = df[target_col].tolist()

    print(f"Total samples: {len(smiles_list)}")

    # Convert to graphs
    if include_quantum:
        graphs = batch_smiles_to_graphs_quantum(smiles_list, y_list, include_quantum=True)
    else:
        from mol_to_graph import batch_smiles_to_graphs
        graphs = batch_smiles_to_graphs(smiles_list, y_list)

    print(f"Valid graphs: {len(graphs)}")
    print(f"Features per node: {graphs[0].x.shape[1]}")

    return graphs


def train_model(model, train_loader, val_loader, epochs=100, lr=0.0001,
                patience=30, device='cpu', class_weight=3.24, model_name='model'):
    """
    Train the model

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs
        lr: Learning rate
        patience: Early stopping patience
        device: Device to use
        class_weight: Weight for positive class
        model_name: Name for saving
    """
    model = model.to(device)

    # Loss with class weights
    pos_weight = torch.tensor([class_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                      patience=10, min_lr=1e-6)

    best_auc = 0
    best_epoch = 0
    no_improve = 0

    history = {
        'train_loss': [], 'val_loss': [],
        'train_auc': [], 'val_auc': [],
        'val_accuracy': []
    }

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

        # Update scheduler
        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]['lr']

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        history['val_accuracy'].append(val_acc)

        # Check for improvement
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            no_improve = 0

            # Save best model
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

        # Early stopping
        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}. Best AUC: {best_auc:.4f} at epoch {best_epoch}")
            break

    return model, history, best_auc


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

    # Calculate metrics
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
    """Main training function"""

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.makedirs('models', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Training parameters
    epochs = 150
    lr = 0.0001
    patience = 40
    batch_size = 32
    class_weight = 3.24

    results = {}

    # ========================================
    # Model 1: Quantum-Enhanced (28 features)
    # ========================================
    print("\n" + "=" * 60)
    print("TRAINING QUANTUM-ENHANCED MODEL (28 features)")
    print("=" * 60)

    # Load data with quantum features
    graphs_quantum = load_bbbp_data(include_quantum=True)

    # Split data
    train_graphs, temp_graphs = train_test_split(graphs_quantum, test_size=0.2, random_state=42)
    val_graphs, test_graphs = train_test_split(temp_graphs, test_size=0.5, random_state=42)

    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

    # Create loaders
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size)
    test_loader = DataLoader(test_graphs, batch_size=batch_size)

    # Create quantum model
    model_quantum = AdvancedHybridBBBNetQuantum(
        num_node_features=28,
        hidden_channels=128,
        num_heads=8,
        dropout=0.3
    )

    # Train
    model_quantum, history_quantum, best_auc_quantum = train_model(
        model_quantum, train_loader, val_loader,
        epochs=epochs, lr=lr, patience=patience,
        device=device, class_weight=class_weight,
        model_name='quantum_model'
    )

    # Evaluate
    metrics_quantum = evaluate_model(model_quantum, test_loader, device)
    results['quantum'] = {
        'best_val_auc': best_auc_quantum,
        'test_metrics': metrics_quantum
    }

    print(f"\nQuantum Model Test Results:")
    print(f"  AUC: {metrics_quantum['auc']:.4f}")
    print(f"  Accuracy: {metrics_quantum['accuracy']*100:.2f}%")
    print(f"  F1 Score: {metrics_quantum['f1']:.4f}")

    # Save history
    np.save('models/quantum_training_history.npy', history_quantum)

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 60)

    print(f"\nQuantum Model (28 features):")
    print(f"  Best Val AUC: {results['quantum']['best_val_auc']:.4f}")
    print(f"  Test AUC: {results['quantum']['test_metrics']['auc']:.4f}")
    print(f"  Test Accuracy: {results['quantum']['test_metrics']['accuracy']*100:.2f}%")

    # Load baseline for comparison
    baseline_path = 'models/best_advanced_model.pth'
    if os.path.exists(baseline_path):
        print(f"\nBaseline Model (15 features):")
        checkpoint = torch.load(baseline_path, map_location='cpu', weights_only=False)
        baseline_auc = checkpoint.get('val_auc', 'N/A')
        if isinstance(baseline_auc, float):
            print(f"  Best Val AUC: {baseline_auc:.4f}")

            # Calculate improvement
            improvement = (results['quantum']['best_val_auc'] - baseline_auc) / baseline_auc * 100
            print(f"\nImprovement from adding quantum descriptors: {improvement:+.2f}%")

    return results


if __name__ == "__main__":
    main()

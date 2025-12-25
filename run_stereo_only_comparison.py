"""
Run Stereochemistry-Only Comparison

Tests whether stereochemistry features ALONE can beat the current best AUC of 0.8316.

Compares THREE model variants:
1. Baseline: Standard model (15 features) - for reference
2. Stereo-Only: 15 atomic + 6 stereo = 21 features (NO quantum)
3. Pretrained + Stereo: ZINC 250k pretraining + 21 features

This isolates the contribution of stereochemistry from quantum features.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, TransformerConv, global_mean_pool, global_max_pool
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import os
import sys

from advanced_bbb_model import AdvancedHybridBBBNet
from mol_to_graph import batch_smiles_to_graphs
from mol_to_graph_enhanced import batch_smiles_to_graphs_enhanced


class StereoOnlyBBBNet(nn.Module):
    """
    BBB model with 21 input features (15 atomic + 6 stereo, NO quantum)
    """
    def __init__(self, node_features=21, hidden_dim=128, num_layers=4, dropout=0.2):
        super().__init__()

        self.node_features = node_features

        # Initial embedding
        self.input_embed = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()

        for _ in range(num_layers):
            self.gat_layers.append(
                GATv2Conv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout, concat=True)
            )
            self.gat_norms.append(nn.BatchNorm1d(hidden_dim))

        # Transformer layer
        self.transformer = TransformerConv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout)
        self.transformer_norm = nn.BatchNorm1d(hidden_dim)

        # Output layers
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, edge_index, batch):
        # Initial embedding
        x = self.input_embed(x)

        # GAT layers with residuals
        for gat, norm in zip(self.gat_layers, self.gat_norms):
            x_new = gat(x, edge_index)
            x_new = norm(x_new)
            x_new = nn.functional.relu(x_new)
            x = x + x_new

        # Transformer
        x_trans = self.transformer(x, edge_index)
        x_trans = self.transformer_norm(x_trans)
        x = x + x_trans

        # Pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        return self.output(x)


def load_bbbp_data_stereo_only():
    """Load BBBP dataset with ONLY stereo features (no quantum)"""
    print("Loading BBBP dataset with STEREO ONLY (21 features: 15 atomic + 6 stereo)...")
    sys.stdout.flush()

    data_path = "data/bbbp_dataset.csv"
    df = pd.read_csv(data_path)

    smiles_col = 'smiles' if 'smiles' in df.columns else 'SMILES'
    target_col = 'p_np' if 'p_np' in df.columns else 'BBB_permeability'

    smiles_list = df[smiles_col].tolist()
    y_list = df[target_col].tolist()

    print(f"Total samples: {len(smiles_list)}")
    sys.stdout.flush()

    # include_quantum=FALSE, include_stereo=TRUE
    graphs = batch_smiles_to_graphs_enhanced(
        smiles_list, y_list,
        include_quantum=False,  # NO quantum features
        include_stereo=True,    # YES stereo features
        use_dft=False,          # Doesn't matter since quantum=False
        verbose=True
    )

    print(f"Valid graphs: {len(graphs)}")
    print(f"Features per node: {graphs[0].x.shape[1]}")
    sys.stdout.flush()

    return graphs


def load_bbbp_data_basic():
    """Load BBBP dataset with basic features (15)"""
    print("Loading BBBP dataset (basic - 15 features)...")
    sys.stdout.flush()

    data_path = "data/bbbp_dataset.csv"
    df = pd.read_csv(data_path)

    smiles_col = 'smiles' if 'smiles' in df.columns else 'SMILES'
    target_col = 'p_np' if 'p_np' in df.columns else 'BBB_permeability'

    smiles_list = df[smiles_col].tolist()
    y_list = df[target_col].tolist()

    print(f"Total samples: {len(smiles_list)}")
    sys.stdout.flush()

    graphs = batch_smiles_to_graphs(smiles_list, y_list)

    print(f"Valid graphs: {len(graphs)}")
    print(f"Features per node: {graphs[0].x.shape[1]}")
    sys.stdout.flush()

    return graphs


def train_model(model, train_loader, val_loader, epochs=150, lr=0.0001,
                patience=40, device='cpu', class_weight=3.24, model_name='model'):
    """Train the model with early stopping"""
    model = model.to(device)

    pos_weight = torch.tensor([class_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                      patience=10, min_lr=1e-6)

    best_auc = 0
    best_epoch = 0
    best_state = None
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
                val_preds.extend(torch.sigmoid(out.view(-1)).cpu().numpy())
                val_labels.extend(batch.y.view(-1).cpu().numpy())

        val_auc = roc_auc_score(val_labels, val_preds)
        val_acc = accuracy_score(val_labels, [1 if p > 0.5 else 0 for p in val_preds])

        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]['lr']

        # Check for improvement
        improved = ""
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            improved = "*BEST*"
        else:
            no_improve += 1

        # Print progress
        if improved or epoch % 10 == 0 or epoch <= 20:
            print(f"Epoch {epoch}/{epochs} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc*100:.1f}% | LR: {current_lr:.6f} {improved}")
            sys.stdout.flush()

        # Early stopping
        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}. Best AUC: {best_auc:.4f} at epoch {best_epoch}")
            break

    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    model = model.to(device)

    return model, best_auc, best_epoch


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            all_preds.extend(torch.sigmoid(out.view(-1)).cpu().numpy())
            all_labels.extend(batch.y.view(-1).cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    pred_binary = (all_preds > 0.5).astype(int)

    metrics = {
        'auc': roc_auc_score(all_labels, all_preds),
        'accuracy': accuracy_score(all_labels, pred_binary),
        'precision': precision_score(all_labels, pred_binary, zero_division=0),
        'recall': recall_score(all_labels, pred_binary, zero_division=0),
        'f1': f1_score(all_labels, pred_binary, zero_division=0)
    }

    return metrics


def main():
    print("=" * 70)
    print("BBB PERMEABILITY - STEREOCHEMISTRY-ONLY COMPARISON")
    print("Testing if stereo features alone can beat AUC 0.8316")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs('models', exist_ok=True)

    results = {}

    # Reference: Current best is 0.8316 (Pretrained + Quantum RDKit approx)
    CURRENT_BEST = 0.8316

    # ========================================================================
    # MODEL 1: BASELINE (15 features, no pretraining) - Reference
    # ========================================================================
    print("\n" + "=" * 70)
    print("MODEL 1: BASELINE (15 features) - Reference")
    print("=" * 70)

    graphs_basic = load_bbbp_data_basic()

    # Split data
    train_basic, temp_basic = train_test_split(graphs_basic, test_size=0.2, random_state=42)
    val_basic, test_basic = train_test_split(temp_basic, test_size=0.5, random_state=42)

    train_loader_basic = DataLoader(train_basic, batch_size=32, shuffle=True)
    val_loader_basic = DataLoader(val_basic, batch_size=32)
    test_loader_basic = DataLoader(test_basic, batch_size=32)

    print(f"Train: {len(train_basic)}, Val: {len(val_basic)}, Test: {len(test_basic)}")

    model_baseline = AdvancedHybridBBBNet(num_node_features=15)
    model_baseline, _, _ = train_model(
        model_baseline, train_loader_basic, val_loader_basic,
        epochs=150, lr=0.0001, patience=40, device=device, model_name='baseline'
    )

    baseline_metrics = evaluate_model(model_baseline, test_loader_basic, device)
    print(f"\nBaseline Test Results: AUC={baseline_metrics['auc']:.4f}, Acc={baseline_metrics['accuracy']*100:.1f}%")

    torch.save(model_baseline.state_dict(), 'models/best_baseline_stereo_test.pth')
    results['baseline'] = {'test_metrics': baseline_metrics}

    # ========================================================================
    # MODEL 2: STEREO-ONLY (21 features = 15 atomic + 6 stereo)
    # ========================================================================
    print("\n" + "=" * 70)
    print("MODEL 2: STEREO-ONLY (21 features = 15 atomic + 6 stereo)")
    print("NO quantum features - testing stereo contribution alone")
    print("=" * 70)

    graphs_stereo = load_bbbp_data_stereo_only()

    # Same split for fair comparison
    train_stereo, temp_stereo = train_test_split(graphs_stereo, test_size=0.2, random_state=42)
    val_stereo, test_stereo = train_test_split(temp_stereo, test_size=0.5, random_state=42)

    train_loader_stereo = DataLoader(train_stereo, batch_size=32, shuffle=True)
    val_loader_stereo = DataLoader(val_stereo, batch_size=32)
    test_loader_stereo = DataLoader(test_stereo, batch_size=32)

    model_stereo = StereoOnlyBBBNet(node_features=21)
    model_stereo, _, _ = train_model(
        model_stereo, train_loader_stereo, val_loader_stereo,
        epochs=150, lr=0.0001, patience=40, device=device, model_name='stereo_only'
    )

    stereo_metrics = evaluate_model(model_stereo, test_loader_stereo, device)
    print(f"\nStereo-Only Test Results: AUC={stereo_metrics['auc']:.4f}, Acc={stereo_metrics['accuracy']*100:.1f}%")

    torch.save(model_stereo.state_dict(), 'models/best_stereo_only.pth')
    results['stereo_only'] = {'test_metrics': stereo_metrics}

    # ========================================================================
    # MODEL 3: PRETRAINED + STEREO (ZINC 250k + 21 features)
    # ========================================================================
    print("\n" + "=" * 70)
    print("MODEL 3: PRETRAINED + STEREO (ZINC 250k pretraining + 21 features)")
    print("=" * 70)

    model_pretrained_stereo = StereoOnlyBBBNet(node_features=21)

    # Try to load pretrained weights (partial transfer since dimensions differ)
    pretrained_path = "models/pretrained_encoder.pth"
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path}...")
        pretrained_state = torch.load(pretrained_path, map_location=device)
        model_state = model_pretrained_stereo.state_dict()

        transferred = 0
        skipped = 0
        for name, param in pretrained_state.items():
            if name in model_state and param.shape == model_state[name].shape:
                model_state[name] = param
                transferred += 1
            else:
                skipped += 1
        model_pretrained_stereo.load_state_dict(model_state)
        print(f"Transferred {transferred} layers, skipped {skipped} layers")
    else:
        print("WARNING: No pretrained weights found. Training from scratch.")

    model_pretrained_stereo, _, _ = train_model(
        model_pretrained_stereo, train_loader_stereo, val_loader_stereo,
        epochs=150, lr=0.00005, patience=40, device=device, model_name='pretrained_stereo'
    )

    pretrained_stereo_metrics = evaluate_model(model_pretrained_stereo, test_loader_stereo, device)
    print(f"\nPretrained+Stereo Test Results: AUC={pretrained_stereo_metrics['auc']:.4f}, Acc={pretrained_stereo_metrics['accuracy']*100:.1f}%")

    torch.save(model_pretrained_stereo.state_dict(), 'models/best_pretrained_stereo.pth')
    results['pretrained_stereo'] = {'test_metrics': pretrained_stereo_metrics}

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEREOCHEMISTRY-ONLY COMPARISON SUMMARY")
    print("=" * 70)

    baseline_auc = results['baseline']['test_metrics']['auc']

    print(f"\n{'Model':<25} {'Test AUC':<10} {'vs Baseline':<12} {'vs Best (0.8316)':<15}")
    print("-" * 65)

    for name, data in results.items():
        m = data['test_metrics']
        vs_baseline = ((m['auc'] - baseline_auc) / baseline_auc) * 100
        vs_best = ((m['auc'] - CURRENT_BEST) / CURRENT_BEST) * 100
        beat_best = "YES!" if m['auc'] > CURRENT_BEST else "no"
        print(f"{name:<25} {m['auc']:.4f}     {vs_baseline:+.1f}%        {vs_best:+.1f}% ({beat_best})")

    print("-" * 65)

    # Key question answer
    print("\n" + "=" * 70)
    print("KEY FINDING")
    print("=" * 70)

    best_stereo_auc = max(results['stereo_only']['test_metrics']['auc'],
                          results['pretrained_stereo']['test_metrics']['auc'])

    if best_stereo_auc > CURRENT_BEST:
        print(f"\n>>> STEREO ALONE BEATS CURRENT BEST! <<<")
        print(f"Best stereo model: {best_stereo_auc:.4f} > {CURRENT_BEST:.4f}")
        print(f"Improvement: {((best_stereo_auc - CURRENT_BEST) / CURRENT_BEST) * 100:+.2f}%")
        print("\nConclusion: Stereochemistry features provide value!")
    else:
        print(f"\n>>> STEREO ALONE DOES NOT BEAT CURRENT BEST <<<")
        print(f"Best stereo model: {best_stereo_auc:.4f} < {CURRENT_BEST:.4f}")
        print(f"Gap: {((best_stereo_auc - CURRENT_BEST) / CURRENT_BEST) * 100:.2f}%")
        print("\nConclusion: Need REAL Gaussian/DFT quantum features to improve further!")
        print("The RDKit 'quantum' approximations are the key driver of current performance.")

    # Save results
    np.save('models/stereo_only_comparison_results.npy', results)
    print(f"\nResults saved to models/stereo_only_comparison_results.npy")


if __name__ == "__main__":
    main()

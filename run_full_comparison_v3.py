"""
Run Full Model Comparison V3 - Enhanced with Real DFT + Stereochemistry

Compares FOUR model variants:
1. Baseline: Standard model trained from scratch on BBBP (15 features)
2. Pretrained: Model pretrained on ZINC 250k, then fine-tuned on BBBP (15 features)
3. Enhanced: Model with real DFT quantum + E-Z isomers (34 features)
4. Pretrained + Enhanced: Pretrained + real DFT + E-Z (34 features)

Enhancement over v2:
- Uses PubChemQC B3LYP/6-31G* database (86M molecules) for REAL DFT quantum descriptors
- Falls back to RDKit approximations for molecules not in PubChemQC
- Includes E-Z isomer (stereochemistry) encoding (6 additional features)
- Total: 15 atomic + 13 quantum + 6 stereo = 34 features
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
from mol_to_graph import batch_smiles_to_graphs
from mol_to_graph_enhanced import batch_smiles_to_graphs_enhanced


class AdvancedHybridBBBNetEnhanced(nn.Module):
    """
    Enhanced BBB model with 34 input features (15 atomic + 13 quantum + 6 stereo)
    """
    def __init__(self, node_features=34, hidden_dim=128, num_layers=4, dropout=0.2):
        super().__init__()
        from torch_geometric.nn import GATv2Conv, TransformerConv, global_mean_pool, global_max_pool

        self.node_features = node_features

        # Initial embedding to higher dimension
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

        # Transformer layer for global context
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
        from torch_geometric.nn import global_mean_pool, global_max_pool

        # Initial embedding
        x = self.input_embed(x)

        # GAT layers with residuals
        for gat, norm in zip(self.gat_layers, self.gat_norms):
            x_new = gat(x, edge_index)
            x_new = norm(x_new)
            x_new = nn.functional.relu(x_new)
            x = x + x_new  # Residual

        # Transformer
        x_trans = self.transformer(x, edge_index)
        x_trans = self.transformer_norm(x_trans)
        x = x + x_trans

        # Pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        return self.output(x)


def load_bbbp_data_enhanced(include_quantum=True, include_stereo=True, use_dft=True):
    """Load BBBP dataset with enhanced features"""
    print(f"Loading BBBP dataset (quantum={include_quantum}, stereo={include_stereo}, dft={use_dft})...")
    sys.stdout.flush()

    data_path = "data/bbbp_dataset.csv"
    df = pd.read_csv(data_path)

    smiles_col = 'smiles' if 'smiles' in df.columns else 'SMILES'
    target_col = 'p_np' if 'p_np' in df.columns else 'BBB_permeability'

    smiles_list = df[smiles_col].tolist()
    y_list = df[target_col].tolist()

    print(f"Total samples: {len(smiles_list)}")
    sys.stdout.flush()

    if include_quantum or include_stereo:
        graphs = batch_smiles_to_graphs_enhanced(
            smiles_list, y_list,
            include_quantum=include_quantum,
            include_stereo=include_stereo,
            use_dft=use_dft,
            verbose=True
        )
    else:
        graphs = batch_smiles_to_graphs(smiles_list, y_list)

    print(f"Valid graphs: {len(graphs)}")
    print(f"Features per node: {graphs[0].x.shape[1]}")
    sys.stdout.flush()

    return graphs


def load_bbbp_data_basic():
    """Load BBBP dataset with basic features (15)"""
    print(f"Loading BBBP dataset (basic - 15 features)...")
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
    print("BBB PERMEABILITY PREDICTION - V3 FULL COMPARISON")
    print("Enhanced with Real DFT (PubChemQC) + E-Z Isomer Encoding")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs('models', exist_ok=True)

    results = {}

    # ========================================================================
    # MODEL 1: BASELINE (15 features, no pretraining)
    # ========================================================================
    print("\n" + "=" * 70)
    print("LOADING BBBP DATA (BASIC - 15 features)")
    print("=" * 70)

    graphs_basic = load_bbbp_data_basic()

    # Split data
    train_graphs, temp_graphs = train_test_split(graphs_basic, test_size=0.2, random_state=42)
    val_graphs, test_graphs = train_test_split(temp_graphs, test_size=0.5, random_state=42)

    train_loader_basic = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader_basic = DataLoader(val_graphs, batch_size=32)
    test_loader_basic = DataLoader(test_graphs, batch_size=32)

    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

    print("\n" + "=" * 70)
    print("MODEL 1: BASELINE (no pretraining, 15 features)")
    print("=" * 70)

    model_baseline = AdvancedHybridBBBNet(num_node_features=15)
    model_baseline, _, _ = train_model(
        model_baseline, train_loader_basic, val_loader_basic,
        epochs=150, lr=0.0001, patience=40, device=device, model_name='baseline_v3'
    )

    baseline_metrics = evaluate_model(model_baseline, test_loader_basic, device)
    print(f"\nBaseline Test Results: AUC={baseline_metrics['auc']:.4f}, Acc={baseline_metrics['accuracy']*100:.1f}%")

    torch.save(model_baseline.state_dict(), 'models/best_baseline_v3.pth')
    results['baseline'] = {'test_auc': baseline_metrics['auc'], 'test_metrics': baseline_metrics}

    # ========================================================================
    # MODEL 2: PRETRAINED (ZINC 250k, 15 features)
    # ========================================================================
    print("\n" + "=" * 70)
    print("MODEL 2: PRETRAINED (ZINC 250k, 15 features)")
    print("=" * 70)

    model_pretrained = AdvancedHybridBBBNet(num_node_features=15)

    pretrained_path = "models/pretrained_encoder.pth"
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path}...")
        pretrained_state = torch.load(pretrained_path, map_location=device)
        model_state = model_pretrained.state_dict()

        transferred = 0
        skipped = 0
        for name, param in pretrained_state.items():
            if name in model_state and param.shape == model_state[name].shape:
                model_state[name] = param
                transferred += 1
            else:
                skipped += 1
        model_pretrained.load_state_dict(model_state)
        print(f"Transferred {transferred} layers, skipped {skipped} layers")
    else:
        print("WARNING: No pretrained weights found. Training from scratch.")

    model_pretrained, _, _ = train_model(
        model_pretrained, train_loader_basic, val_loader_basic,
        epochs=150, lr=0.00005, patience=40, device=device, model_name='pretrained_v3'
    )

    pretrained_metrics = evaluate_model(model_pretrained, test_loader_basic, device)
    print(f"\nPretrained Test Results: AUC={pretrained_metrics['auc']:.4f}, Acc={pretrained_metrics['accuracy']*100:.1f}%")

    torch.save(model_pretrained.state_dict(), 'models/best_pretrained_v3.pth')
    results['pretrained'] = {'test_auc': pretrained_metrics['auc'], 'test_metrics': pretrained_metrics}

    # ========================================================================
    # MODEL 3: ENHANCED (Real DFT + E-Z, 34 features)
    # ========================================================================
    print("\n" + "=" * 70)
    print("LOADING BBBP DATA (ENHANCED - 34 features)")
    print("Real DFT from PubChemQC + E-Z Isomer Encoding")
    print("=" * 70)

    graphs_enhanced = load_bbbp_data_enhanced(
        include_quantum=True,
        include_stereo=True,
        use_dft=True
    )

    # Same split indices for fair comparison
    train_enhanced, temp_enhanced = train_test_split(graphs_enhanced, test_size=0.2, random_state=42)
    val_enhanced, test_enhanced = train_test_split(temp_enhanced, test_size=0.5, random_state=42)

    train_loader_enhanced = DataLoader(train_enhanced, batch_size=32, shuffle=True)
    val_loader_enhanced = DataLoader(val_enhanced, batch_size=32)
    test_loader_enhanced = DataLoader(test_enhanced, batch_size=32)

    print("\n" + "=" * 70)
    print("MODEL 3: ENHANCED (Real DFT + E-Z, 34 features)")
    print("=" * 70)

    model_enhanced = AdvancedHybridBBBNetEnhanced(node_features=34)
    model_enhanced, _, _ = train_model(
        model_enhanced, train_loader_enhanced, val_loader_enhanced,
        epochs=150, lr=0.0001, patience=40, device=device, model_name='enhanced_v3'
    )

    enhanced_metrics = evaluate_model(model_enhanced, test_loader_enhanced, device)
    print(f"\nEnhanced Test Results: AUC={enhanced_metrics['auc']:.4f}, Acc={enhanced_metrics['accuracy']*100:.1f}%")

    torch.save(model_enhanced.state_dict(), 'models/best_enhanced_v3.pth')
    results['enhanced'] = {'test_auc': enhanced_metrics['auc'], 'test_metrics': enhanced_metrics}

    # ========================================================================
    # MODEL 4: PRETRAINED + ENHANCED
    # ========================================================================
    print("\n" + "=" * 70)
    print("MODEL 4: PRETRAINED + ENHANCED (ZINC 250k + Real DFT + E-Z)")
    print("=" * 70)

    model_combined = AdvancedHybridBBBNetEnhanced(node_features=34)

    # Load pretrained weights (partial transfer)
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path}...")
        pretrained_state = torch.load(pretrained_path, map_location=device)
        model_state = model_combined.state_dict()

        transferred = 0
        skipped = 0
        for name, param in pretrained_state.items():
            if name in model_state and param.shape == model_state[name].shape:
                model_state[name] = param
                transferred += 1
            else:
                skipped += 1
        model_combined.load_state_dict(model_state)
        print(f"Transferred {transferred} layers, skipped {skipped} layers")

    model_combined, _, _ = train_model(
        model_combined, train_loader_enhanced, val_loader_enhanced,
        epochs=150, lr=0.00005, patience=40, device=device, model_name='combined_v3'
    )

    combined_metrics = evaluate_model(model_combined, test_loader_enhanced, device)
    print(f"\nCombined Test Results: AUC={combined_metrics['auc']:.4f}, Acc={combined_metrics['accuracy']*100:.1f}%")

    torch.save(model_combined.state_dict(), 'models/best_combined_v3.pth')
    results['pretrained_enhanced'] = {'test_auc': combined_metrics['auc'], 'test_metrics': combined_metrics}

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("FINAL 4-MODEL COMPARISON SUMMARY (V3 - Enhanced)")
    print("=" * 70)

    baseline_auc = results['baseline']['test_auc']

    print(f"\n{'Model':<30} {'Test AUC':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 80)

    for name, data in results.items():
        m = data['test_metrics']
        improvement = ((data['test_auc'] - baseline_auc) / baseline_auc) * 100
        print(f"{name.replace('_', ' ').title():<30} {m['auc']:.4f}     {m['accuracy']*100:.1f}%      {m['precision']:.4f}     {m['recall']:.4f}     {m['f1']:.4f}     ({improvement:+.1f}%)")

    print("-" * 80)

    # Find winners
    print("\n" + "=" * 70)
    print("CATEGORY WINNERS")
    print("=" * 70)

    best_auc_model = max(results.items(), key=lambda x: x[1]['test_auc'])
    best_recall_model = max(results.items(), key=lambda x: x[1]['test_metrics']['recall'])
    best_precision_model = max(results.items(), key=lambda x: x[1]['test_metrics']['precision'])

    print(f"\nBest Overall (AUC): {best_auc_model[0].upper()}")
    print(f"  AUC: {best_auc_model[1]['test_auc']:.4f}")

    print(f"\nBest Recall (finds most BBB+ compounds): {best_recall_model[0].upper()}")
    print(f"  Recall: {best_recall_model[1]['test_metrics']['recall']:.4f}")

    print(f"\nBest Precision (fewest false positives): {best_precision_model[0].upper()}")
    print(f"  Precision: {best_precision_model[1]['test_metrics']['precision']:.4f}")

    # Save results
    np.save('models/full_comparison_results_v3.npy', results)
    print(f"\nResults saved to models/full_comparison_results_v3.npy")


if __name__ == "__main__":
    main()

"""
Resume Stereo-Aware Pretraining from Saved Graphs
Uses the saved 56,916 graphs from data/zinc_stereo_graphs.pkl
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
import pickle

from zinc_stereo_pretraining import (
    StereoAwareBBBNet,
    StereoPretrainingModel,
)
from mol_to_graph_enhanced import batch_smiles_to_graphs_enhanced
from mol_to_graph import batch_smiles_to_graphs
from advanced_bbb_model import AdvancedHybridBBBNet


def load_saved_graphs(path='data/zinc_stereo_graphs.pkl', max_graphs=10000):
    """Load pre-processed graphs from disk (subset for speed)."""
    print(f"Loading saved graphs from {path}...")
    with open(path, 'rb') as f:
        data = pickle.load(f)

    graphs = data['graphs'][:max_graphs]  # Use subset for faster training
    print(f"Loaded {len(graphs)} graphs (subset)")
    return graphs


def pretrain_from_saved(graphs, epochs=30, device='cpu'):
    """Pretrain encoder using saved graphs with checkpoint saving."""

    save_path = 'models/pretrained_stereo_encoder.pth'
    checkpoint_path = 'models/pretrain_checkpoint.pth'

    print("=" * 70)
    print("PRETRAINING ON SAVED ZINC STEREO GRAPHS")
    print(f"Graphs: {len(graphs)} | Epochs: {epochs}")
    print("=" * 70)
    sys.stdout.flush()

    # Create dataloader - larger batch for faster training
    loader = DataLoader(graphs, batch_size=128, shuffle=True)

    model = StereoPretrainingModel(node_features=21).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 1

    # Check for existing checkpoint
    if os.path.exists(checkpoint_path):
        print(f"\nFound checkpoint at {checkpoint_path}, resuming...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    print(f"\nTraining for {epochs} epochs (starting at {start_epoch})...")
    print("=" * 60)
    sys.stdout.flush()

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            pred_mw, pred_ac, pred_stereo = model(batch.x, batch.edge_index, batch.batch)

            loss_mw = mse_loss(pred_mw.view(-1), batch.mol_weight.view(-1))
            loss_ac = mse_loss(pred_ac.view(-1), batch.atom_count.view(-1))
            loss_stereo = bce_loss(pred_stereo.view(-1), batch.has_stereo.view(-1))

            loss = loss_mw + loss_ac + 0.5 * loss_stereo

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / num_batches

        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            os.makedirs('models', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'loss': avg_loss
            }, checkpoint_path)
            print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} [CHECKPOINT SAVED]")
        elif epoch == 1:
            print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        sys.stdout.flush()

    # Save final model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nPretrained model saved to {save_path}")

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Checkpoint cleaned up")

    return save_path


def train_model(model, train_loader, val_loader, epochs=150, lr=0.0001,
                patience=40, device='cpu', class_weight=3.24, model_name='model'):
    """Train BBB model with early stopping."""
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
        model.train()
        train_preds, train_labels = [], []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out.view(-1), batch.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_preds.extend(torch.sigmoid(out.view(-1)).detach().cpu().numpy())
            train_labels.extend(batch.y.view(-1).cpu().numpy())

        train_auc = roc_auc_score(train_labels, train_preds)

        # Validation
        model.eval()
        val_preds, val_labels = [], []

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

        improved = ""
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            improved = "*BEST*"
        else:
            no_improve += 1

        if improved or epoch % 10 == 0 or epoch <= 5:
            print(f"Epoch {epoch}/{epochs} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc*100:.1f}% | LR: {current_lr:.6f} {improved}")
            sys.stdout.flush()

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}. Best AUC: {best_auc:.4f} at epoch {best_epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)
    model = model.to(device)

    return model, best_auc, best_epoch


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model on test set."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            all_preds.extend(torch.sigmoid(out.view(-1)).cpu().numpy())
            all_labels.extend(batch.y.view(-1).cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    pred_binary = (all_preds > 0.5).astype(int)

    return {
        'auc': roc_auc_score(all_labels, all_preds),
        'accuracy': accuracy_score(all_labels, pred_binary),
        'precision': precision_score(all_labels, pred_binary, zero_division=0),
        'recall': recall_score(all_labels, pred_binary, zero_division=0),
        'f1': f1_score(all_labels, pred_binary, zero_division=0)
    }


def main():
    print("=" * 70)
    print("BBB PREDICTION - STEREO-AWARE PRETRAINING FROM SAVED GRAPHS")
    print("Goal: Beat current best of 0.8316")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    CURRENT_BEST = 0.8316
    results = {}

    # ========================================================================
    # PHASE 1: PRETRAIN FROM SAVED GRAPHS
    # ========================================================================
    pretrained_path = 'models/pretrained_stereo_encoder.pth'

    if os.path.exists(pretrained_path):
        print(f"\nFound existing pretrained model: {pretrained_path}")
        print("Skipping pretraining...")
    else:
        # Load saved graphs (10k subset for speed on CPU)
        graphs = load_saved_graphs('data/zinc_stereo_graphs.pkl', max_graphs=10000)
        pretrained_path = pretrain_from_saved(graphs, epochs=10, device=device)

    if not os.path.exists(pretrained_path):
        print("ERROR: Pretraining failed")
        return

    # ========================================================================
    # PHASE 2: LOAD BBBP DATA
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: LOADING BBBP DATA")
    print("=" * 70)

    data_path = "data/bbbp_dataset.csv"
    df = pd.read_csv(data_path)

    smiles_col = 'smiles' if 'smiles' in df.columns else 'SMILES'
    target_col = 'p_np' if 'p_np' in df.columns else 'BBB_permeability'

    smiles_list = df[smiles_col].tolist()
    y_list = df[target_col].tolist()

    # Basic graphs (15 features)
    print("\nConverting BBBP to basic graphs (15 features)...")
    graphs_basic = batch_smiles_to_graphs(smiles_list, y_list)

    # Stereo graphs (21 features)
    print("\nConverting BBBP to stereo graphs (21 features)...")
    graphs_stereo = batch_smiles_to_graphs_enhanced(
        smiles_list, y_list,
        include_quantum=False,
        include_stereo=True,
        use_dft=False,
        verbose=True
    )

    # Split
    train_basic, temp_basic = train_test_split(graphs_basic, test_size=0.2, random_state=42)
    val_basic, test_basic = train_test_split(temp_basic, test_size=0.5, random_state=42)

    train_stereo, temp_stereo = train_test_split(graphs_stereo, test_size=0.2, random_state=42)
    val_stereo, test_stereo = train_test_split(temp_stereo, test_size=0.5, random_state=42)

    train_loader_basic = DataLoader(train_basic, batch_size=32, shuffle=True)
    val_loader_basic = DataLoader(val_basic, batch_size=32)
    test_loader_basic = DataLoader(test_basic, batch_size=32)

    train_loader_stereo = DataLoader(train_stereo, batch_size=32, shuffle=True)
    val_loader_stereo = DataLoader(val_stereo, batch_size=32)
    test_loader_stereo = DataLoader(test_stereo, batch_size=32)

    # ========================================================================
    # MODEL 1: BASELINE (15 features, no pretraining)
    # ========================================================================
    print("\n" + "=" * 70)
    print("MODEL 1: BASELINE (15 features, no pretraining)")
    print("=" * 70)

    model_baseline = AdvancedHybridBBBNet(num_node_features=15)
    model_baseline, _, _ = train_model(
        model_baseline, train_loader_basic, val_loader_basic,
        epochs=150, lr=0.0001, patience=40, device=device, model_name='baseline'
    )

    baseline_metrics = evaluate_model(model_baseline, test_loader_basic, device)
    print(f"\nBaseline Test: AUC={baseline_metrics['auc']:.4f}")
    results['baseline'] = baseline_metrics

    # ========================================================================
    # MODEL 2: STEREO-ONLY (21 features, no pretraining)
    # ========================================================================
    print("\n" + "=" * 70)
    print("MODEL 2: STEREO-ONLY (21 features, no pretraining)")
    print("=" * 70)

    model_stereo = StereoAwareBBBNet(node_features=21)
    model_stereo, _, _ = train_model(
        model_stereo, train_loader_stereo, val_loader_stereo,
        epochs=150, lr=0.0001, patience=40, device=device, model_name='stereo_only'
    )

    stereo_metrics = evaluate_model(model_stereo, test_loader_stereo, device)
    print(f"\nStereo-Only Test: AUC={stereo_metrics['auc']:.4f}")
    results['stereo_only'] = stereo_metrics

    # ========================================================================
    # MODEL 3: PRETRAINED + STEREO (21 features, ZINC pretraining)
    # ========================================================================
    print("\n" + "=" * 70)
    print("MODEL 3: PRETRAINED + STEREO (21 features, ZINC stereo pretraining)")
    print("=" * 70)

    model_pretrained_stereo = StereoAwareBBBNet(node_features=21)

    # Load pretrained encoder
    print(f"Loading pretrained encoder from {pretrained_path}...")
    checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)

    # Extract and load encoder weights
    encoder_state = {}
    for key, value in checkpoint.items():
        if key.startswith('encoder.'):
            new_key = key.replace('encoder.', '')
            encoder_state[new_key] = value

    model_pretrained_stereo.encoder.load_state_dict(encoder_state)
    print(f"Loaded {len(encoder_state)} encoder layers")

    model_pretrained_stereo, _, _ = train_model(
        model_pretrained_stereo, train_loader_stereo, val_loader_stereo,
        epochs=150, lr=0.00005, patience=40, device=device, model_name='pretrained_stereo'
    )

    pretrained_stereo_metrics = evaluate_model(model_pretrained_stereo, test_loader_stereo, device)
    print(f"\nPretrained+Stereo Test: AUC={pretrained_stereo_metrics['auc']:.4f}")
    results['pretrained_stereo'] = pretrained_stereo_metrics

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 70)

    baseline_auc = results['baseline']['auc']

    print(f"\n{'Model':<30} {'Test AUC':<10} {'vs Baseline':<12} {'vs Best (0.8316)':<15}")
    print("-" * 70)

    for name, metrics in results.items():
        vs_baseline = ((metrics['auc'] - baseline_auc) / baseline_auc) * 100
        vs_best = ((metrics['auc'] - CURRENT_BEST) / CURRENT_BEST) * 100
        beat_best = "YES!" if metrics['auc'] > CURRENT_BEST else "no"
        print(f"{name:<30} {metrics['auc']:.4f}     {vs_baseline:+.1f}%        {vs_best:+.1f}% ({beat_best})")

    print("-" * 70)

    # Best model
    best_model = max(results.items(), key=lambda x: x[1]['auc'])
    print(f"\nBest Model: {best_model[0]} with AUC = {best_model[1]['auc']:.4f}")

    if best_model[1]['auc'] > CURRENT_BEST:
        print(f"\n>>> NEW RECORD! Beat {CURRENT_BEST} by {((best_model[1]['auc'] - CURRENT_BEST) / CURRENT_BEST) * 100:.2f}% <<<")
    else:
        print(f"\nDid not beat current best ({CURRENT_BEST}). Need real Gaussian features.")

    # Save results
    np.save('models/stereo_pretrain_comparison_results.npy', results)
    print(f"\nResults saved to models/stereo_pretrain_comparison_results.npy")


if __name__ == "__main__":
    main()

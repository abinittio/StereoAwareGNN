"""
BBB Fine-tuning with Pretrained Stereo Encoder
Uses pretrained_stereo_full.pth from ZINC pretraining.
Target: Beat 0.8316 AUC

Run: python finetune_bbb_stereo.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

from zinc_stereo_pretraining import StereoAwareEncoder
from mol_to_graph_enhanced import mol_to_graph_enhanced


class BBBClassifier(nn.Module):
    """BBB classifier with pretrained stereo encoder."""

    def __init__(self, encoder, hidden_dim=128, freeze_encoder=False):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Classification head
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
        with torch.set_grad_enabled(not self.freeze_encoder):
            graph_embed = self.encoder(x, edge_index, batch)
        return self.classifier(graph_embed)

    def unfreeze_encoder(self):
        """Unfreeze encoder for fine-tuning."""
        self.freeze_encoder = False
        for param in self.encoder.parameters():
            param.requires_grad = True


def load_bbb_data(csv_path='data/bbbp_dataset.csv'):
    """Load BBB dataset and convert to graphs."""
    print("Loading BBB dataset...")
    df = pd.read_csv(csv_path)
    print(f"  Total molecules: {len(df)}")
    print(f"  BBB+ (permeable): {df['BBB_permeability'].sum()}")
    print(f"  BBB- (non-permeable): {len(df) - df['BBB_permeability'].sum()}")

    graphs = []
    labels = []
    valid_count = 0

    print("Converting to stereo-aware graphs...")
    for idx, row in df.iterrows():
        smiles = row['SMILES']
        label = float(row['BBB_permeability'])

        # Convert to graph with stereo features (21 features)
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
            valid_count += 1

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx+1}/{len(df)} ({valid_count} valid)")
            sys.stdout.flush()

    print(f"Valid graphs: {len(graphs)}/{len(df)}")
    return graphs, np.array(labels)


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out.view(-1), batch.y.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(torch.sigmoid(out).detach().cpu().numpy().flatten())
        all_labels.extend(batch.y.cpu().numpy().flatten())

    auc = roc_auc_score(all_labels, all_preds)
    return total_loss / len(loader), auc


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out.view(-1), batch.y.view(-1))

            total_loss += loss.item()
            all_preds.extend(torch.sigmoid(out).cpu().numpy().flatten())
            all_labels.extend(batch.y.cpu().numpy().flatten())

    auc = roc_auc_score(all_labels, all_preds)
    preds_binary = (np.array(all_preds) > 0.5).astype(int)
    acc = accuracy_score(all_labels, preds_binary)

    return total_loss / len(loader), auc, acc, all_preds, all_labels


def main():
    print("=" * 70)
    print("BBB FINE-TUNING WITH PRETRAINED STEREO ENCODER")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Config
    PRETRAINED_PATH = 'models/pretrained_stereo_full.pth'
    BATCH_SIZE = 32
    EPOCHS_FROZEN = 10      # Train with frozen encoder first
    EPOCHS_FINETUNE = 20    # Then fine-tune everything
    LR_FROZEN = 0.001
    LR_FINETUNE = 0.0001
    N_FOLDS = 5
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Device: {DEVICE}")
    print(f"Pretrained model: {PRETRAINED_PATH}")
    print(f"Training: {EPOCHS_FROZEN} epochs frozen + {EPOCHS_FINETUNE} epochs fine-tuning")
    print()

    # Load data
    graphs, labels = load_bbb_data()
    print()

    # 5-fold cross-validation
    kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    all_fold_aucs = []
    all_fold_accs = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(graphs, labels)):
        print("=" * 60)
        print(f"FOLD {fold + 1}/{N_FOLDS}")
        print("=" * 60)

        # Split data
        train_graphs = [graphs[i] for i in train_idx]
        val_graphs = [graphs[i] for i in val_idx]

        train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE)

        print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}")

        # Create model with pretrained encoder
        encoder = StereoAwareEncoder(node_features=21, hidden_dim=128, num_layers=4)

        # Load pretrained weights
        pretrained_weights = torch.load(PRETRAINED_PATH, map_location=DEVICE)
        encoder.load_state_dict(pretrained_weights)
        print(f"Loaded pretrained encoder from {PRETRAINED_PATH}")

        model = BBBClassifier(encoder, hidden_dim=128, freeze_encoder=True).to(DEVICE)

        criterion = nn.BCEWithLogitsLoss()

        best_val_auc = 0
        best_epoch = 0

        # Phase 1: Train with frozen encoder
        print(f"\nPhase 1: Training classifier (encoder frozen)...")
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR_FROZEN,
            weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_FROZEN)

        for epoch in range(1, EPOCHS_FROZEN + 1):
            train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_loss, val_auc, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
            scheduler.step()

            marker = ""
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                marker = " *BEST*"
                # Save best model for this fold
                torch.save(model.state_dict(), f'models/bbb_stereo_fold{fold+1}_best.pth')

            print(f"  Epoch {epoch:2d} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}{marker}")
            sys.stdout.flush()

        # Phase 2: Fine-tune entire model
        print(f"\nPhase 2: Fine-tuning entire model...")
        model.unfreeze_encoder()

        optimizer = optim.Adam(model.parameters(), lr=LR_FINETUNE, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_FINETUNE)

        for epoch in range(1, EPOCHS_FINETUNE + 1):
            train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_loss, val_auc, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
            scheduler.step()

            marker = ""
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = EPOCHS_FROZEN + epoch
                marker = " *BEST*"
                torch.save(model.state_dict(), f'models/bbb_stereo_fold{fold+1}_best.pth')

            print(f"  Epoch {epoch:2d} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}{marker}")
            sys.stdout.flush()

        # Load best model and get final metrics
        model.load_state_dict(torch.load(f'models/bbb_stereo_fold{fold+1}_best.pth', map_location=DEVICE))
        _, final_auc, final_acc, preds, true_labels = evaluate(model, val_loader, criterion, DEVICE)

        all_fold_aucs.append(final_auc)
        all_fold_accs.append(final_acc)

        preds_binary = (np.array(preds) > 0.5).astype(int)
        precision = precision_score(true_labels, preds_binary)
        recall = recall_score(true_labels, preds_binary)
        f1 = f1_score(true_labels, preds_binary)

        print(f"\nFold {fold+1} Results (Best @ Epoch {best_epoch}):")
        print(f"  AUC:       {final_auc:.4f}")
        print(f"  Accuracy:  {final_acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1:        {f1:.4f}")
        print()

    # Final summary
    print("=" * 70)
    print("FINAL RESULTS (5-FOLD CROSS-VALIDATION)")
    print("=" * 70)
    print(f"Mean AUC:      {np.mean(all_fold_aucs):.4f} +/- {np.std(all_fold_aucs):.4f}")
    print(f"Mean Accuracy: {np.mean(all_fold_accs):.4f} +/- {np.std(all_fold_accs):.4f}")
    print()
    print(f"Per-fold AUCs: {[f'{auc:.4f}' for auc in all_fold_aucs]}")
    print()

    # Compare to baseline
    BASELINE_AUC = 0.8316
    mean_auc = np.mean(all_fold_aucs)
    if mean_auc > BASELINE_AUC:
        print(f"SUCCESS! Beat baseline AUC of {BASELINE_AUC:.4f} by {(mean_auc - BASELINE_AUC)*100:.2f}%")
    else:
        print(f"Did not beat baseline AUC of {BASELINE_AUC:.4f} (diff: {(mean_auc - BASELINE_AUC)*100:.2f}%)")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Best models saved in models/bbb_stereo_fold*_best.pth")


if __name__ == "__main__":
    main()

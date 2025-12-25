"""
Full-scale pretraining on 322k+ ZINC graphs with stereoisomers.
Saves checkpoints every epoch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import pickle
import os
import sys
from datetime import datetime

# Import the stereo-aware encoder
from zinc_stereo_pretraining import StereoAwareEncoder


def load_graphs(path='data/zinc_stereo_graphs.pkl'):
    """Load preprocessed graphs."""
    print(f"Loading graphs from {path}...")
    sys.stdout.flush()

    with open(path, 'rb') as f:
        data = pickle.load(f)

    graphs = data['graphs']
    print(f"Loaded {len(graphs)} graphs")
    print(f"  Original molecules: {data.get('num_original', 'unknown')}")
    print(f"  Features per node: {graphs[0].x.shape[1]}")
    sys.stdout.flush()

    return graphs


class PretrainHead(nn.Module):
    """Prediction head for self-supervised pretraining."""
    def __init__(self, input_dim=256):
        super().__init__()
        # Predict 3 targets: mol_weight, atom_count, has_stereo
        self.head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.head(x)


def pretrain_epoch(model, head, loader, optimizer, device):
    """Run one pretraining epoch."""
    model.train()
    head.train()

    total_loss = 0
    num_batches = 0

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward
        embeddings = model(batch.x, batch.edge_index, batch.batch)
        predictions = head(embeddings)

        # Targets
        targets_mw = batch.mol_weight.view(-1, 1)
        targets_ac = batch.atom_count.view(-1, 1)
        targets_stereo = batch.has_stereo.view(-1, 1)

        # Loss
        loss_mw = mse(predictions[:, 0:1], targets_mw)
        loss_ac = mse(predictions[:, 1:2], targets_ac)
        loss_stereo = bce(predictions[:, 2:3], targets_stereo)

        loss = loss_mw + loss_ac + loss_stereo

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def main():
    print("=" * 70)
    print("FULL-SCALE STEREO PRETRAINING")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    sys.stdout.flush()

    # Config
    EPOCHS = 20
    BATCH_SIZE = 256
    LR = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CHECKPOINT_DIR = 'models/checkpoints'

    print(f"\nConfig:")
    print(f"  Device: {DEVICE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LR}")
    sys.stdout.flush()

    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Load data
    graphs = load_graphs()

    # Create dataloader
    loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=0, pin_memory=True)
    print(f"\nDataLoader: {len(loader)} batches")
    sys.stdout.flush()

    # Create models
    model = StereoAwareEncoder(node_features=21, hidden_dim=128, num_layers=4).to(DEVICE)
    head = PretrainHead(input_dim=256).to(DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Encoder parameters: {total_params:,}")
    sys.stdout.flush()

    # Optimizer
    optimizer = optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=LR,
        weight_decay=0.01
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Check for existing checkpoint
    latest_checkpoint = None
    start_epoch = 0

    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('stereo_epoch_')]
    if checkpoint_files:
        latest = max(checkpoint_files, key=lambda x: int(x.split('_')[2].split('.')[0]))
        latest_checkpoint = os.path.join(CHECKPOINT_DIR, latest)
        print(f"\nFound checkpoint: {latest_checkpoint}")

        checkpoint = torch.load(latest_checkpoint, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        head.load_state_dict(checkpoint['head'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
        sys.stdout.flush()

    # Training loop
    print(f"\n{'='*70}")
    print("STARTING PRETRAINING")
    print(f"{'='*70}")
    sys.stdout.flush()

    best_loss = float('inf')

    for epoch in range(start_epoch, EPOCHS):
        epoch_start = datetime.now()

        loss = pretrain_epoch(model, head, loader, optimizer, DEVICE)
        scheduler.step()

        epoch_time = (datetime.now() - epoch_start).total_seconds()
        lr = scheduler.get_last_lr()[0]

        is_best = loss < best_loss
        if is_best:
            best_loss = loss

        print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {loss:.6f} | LR: {lr:.6f} | Time: {epoch_time:.1f}s {'*BEST*' if is_best else ''}")
        sys.stdout.flush()

        # Save checkpoint every epoch
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'head': head.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': loss,
            'best_loss': best_loss
        }
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'stereo_epoch_{epoch+1:02d}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")
        sys.stdout.flush()

        # Also save best model
        if is_best:
            best_path = 'models/pretrained_stereo_full.pth'
            torch.save(model.state_dict(), best_path)
            print(f"  Saved best model: {best_path}")
            sys.stdout.flush()

    print(f"\n{'='*70}")
    print(f"PRETRAINING COMPLETE!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Final model: models/pretrained_stereo_full.pth")
    print(f"Checkpoints: {CHECKPOINT_DIR}/")
    print(f"{'='*70}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

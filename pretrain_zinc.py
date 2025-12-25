"""
Self-Supervised Pretraining on ZINC 250k Dataset

Uses contrastive learning and masked atom prediction to learn
molecular representations before fine-tuning on BBB prediction.

Pretraining tasks:
1. Masked Atom Prediction (MAM) - predict masked atom types
2. Context Prediction - predict if subgraphs are from same molecule
3. Graph-level contrastive learning - distinguish different molecules

This learns general molecular representations that transfer to BBB prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_mean_pool
import pandas as pd
import numpy as np
from rdkit import Chem
import os
import time
import random
from tqdm import tqdm

from mol_to_graph import mol_to_graph
from advanced_bbb_model import AdvancedHybridBBBNet


class MolecularEncoder(nn.Module):
    """
    Encoder network for self-supervised pretraining.
    Same architecture as AdvancedHybridBBBNet but with projection heads.
    """

    def __init__(self, num_node_features=15, hidden_channels=128, num_heads=8, dropout=0.3):
        super().__init__()

        self.num_node_features = num_node_features

        # === Same architecture as AdvancedHybridBBBNet ===
        # Layer 1: GAT
        self.gat1 = GATConv(num_node_features, hidden_channels, heads=num_heads,
                           dropout=dropout, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_channels * num_heads)

        # Layer 2: GCN
        self.gcn1 = GCNConv(hidden_channels * num_heads, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        # Layer 3: GraphSAGE
        self.sage1 = SAGEConv(hidden_channels, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)

        # Layer 4: GAT again
        self.gat2 = GATConv(hidden_channels, hidden_channels, heads=4, dropout=dropout, concat=False)
        self.bn4 = nn.BatchNorm1d(hidden_channels)

        self.dropout = nn.Dropout(dropout)

        # === Projection head for contrastive learning ===
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 64)
        )

        # === Atom prediction head (for masked atom prediction) ===
        self.atom_predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 118)  # Predict atom type (periodic table)
        )

    def encode(self, x, edge_index, batch):
        """Encode molecule to representation"""
        # Layer 1: GAT
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)

        # Layer 2: GCN
        x = self.gcn1(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)

        # Layer 3: SAGE
        x = self.sage1(x, edge_index)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.dropout(x)

        # Layer 4: GAT
        x = self.gat2(x, edge_index)
        x = self.bn4(x)
        x = F.elu(x)

        return x

    def forward(self, x, edge_index, batch):
        """Forward pass returning node and graph representations"""
        node_repr = self.encode(x, edge_index, batch)

        # Graph-level representation
        graph_repr = global_mean_pool(node_repr, batch)

        # Projected representation for contrastive learning
        proj_repr = self.projection_head(graph_repr)

        return node_repr, graph_repr, proj_repr

    def predict_atoms(self, node_repr):
        """Predict atom types from node representations"""
        return self.atom_predictor(node_repr)


def mask_atoms(data, mask_ratio=0.15):
    """
    Mask random atoms for masked atom prediction task.

    Returns:
        masked_data: Data object with masked atom features
        mask_indices: Indices of masked atoms
        original_atoms: Original atom types of masked atoms
    """
    num_nodes = data.x.size(0)
    num_mask = max(1, int(num_nodes * mask_ratio))

    # Random indices to mask
    mask_indices = torch.randperm(num_nodes)[:num_mask]

    # Store original atom types (first feature is atomic number)
    original_atoms = data.x[mask_indices, 0].clone().long()

    # Create masked version
    masked_x = data.x.clone()
    # Replace masked atoms with special mask token (zeros)
    masked_x[mask_indices] = 0

    masked_data = Data(
        x=masked_x,
        edge_index=data.edge_index,
        y=data.y if hasattr(data, 'y') else None
    )

    return masked_data, mask_indices, original_atoms


def augment_molecule(data):
    """
    Create augmented view of molecule for contrastive learning.
    Uses random edge dropout and feature noise.
    """
    # Edge dropout (remove some bonds)
    num_edges = data.edge_index.size(1)
    keep_ratio = random.uniform(0.8, 1.0)
    keep_mask = torch.rand(num_edges) < keep_ratio

    # Ensure we keep at least some edges
    if keep_mask.sum() < 2:
        keep_mask[:2] = True

    aug_edge_index = data.edge_index[:, keep_mask]

    # Feature noise
    noise = torch.randn_like(data.x) * 0.1
    aug_x = data.x + noise

    aug_data = Data(
        x=aug_x,
        edge_index=aug_edge_index,
        y=data.y if hasattr(data, 'y') else None
    )

    return aug_data


def contrastive_loss(z1, z2, temperature=0.5):
    """
    NT-Xent contrastive loss (SimCLR style)
    """
    batch_size = z1.size(0)

    # Normalize
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Similarity matrix
    sim_matrix = torch.mm(z1, z2.t()) / temperature

    # Labels: diagonal elements are positive pairs
    labels = torch.arange(batch_size).to(z1.device)

    # Cross entropy loss
    loss = F.cross_entropy(sim_matrix, labels)

    return loss


def pretrain_epoch(model, dataloader, optimizer, device, epoch):
    """Run one pretraining epoch"""
    model.train()

    total_mam_loss = 0  # Masked Atom Modeling
    total_cl_loss = 0   # Contrastive Learning
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()

        # === Task 1: Masked Atom Modeling ===
        masked_batch, mask_indices, original_atoms = mask_atoms(batch, mask_ratio=0.15)
        masked_batch = masked_batch.to(device)

        node_repr, _, _ = model(masked_batch.x, masked_batch.edge_index, batch.batch)
        atom_pred = model.predict_atoms(node_repr[mask_indices])
        mam_loss = F.cross_entropy(atom_pred, original_atoms.to(device))

        # === Task 2: Contrastive Learning ===
        # Create two augmented views
        # For simplicity, use original and masked as two views
        _, _, proj1 = model(batch.x, batch.edge_index, batch.batch)
        aug_batch = augment_molecule(batch)
        aug_batch = aug_batch.to(device)
        _, _, proj2 = model(aug_batch.x, aug_batch.edge_index, batch.batch)

        cl_loss = contrastive_loss(proj1, proj2)

        # Combined loss
        loss = mam_loss + 0.5 * cl_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_mam_loss += mam_loss.item()
        total_cl_loss += cl_loss.item()
        num_batches += 1

        pbar.set_postfix({
            'MAM': f'{mam_loss.item():.4f}',
            'CL': f'{cl_loss.item():.4f}'
        })

    return total_mam_loss / num_batches, total_cl_loss / num_batches


def load_zinc_data(max_molecules=None, batch_size=64):
    """Load ZINC 250k and convert to graph data"""

    print("Loading ZINC 250k dataset...")
    zinc_path = "data/zinc250k.csv"

    if not os.path.exists(zinc_path):
        raise FileNotFoundError(f"ZINC dataset not found at {zinc_path}. Run download_zinc250k.py first.")

    df = pd.read_csv(zinc_path)

    # Get SMILES column
    smiles_col = 'smiles' if 'smiles' in df.columns else 'SMILES'
    smiles_list = df[smiles_col].tolist()

    if max_molecules:
        smiles_list = smiles_list[:max_molecules]

    print(f"Converting {len(smiles_list)} molecules to graphs...")

    graphs = []
    failed = 0

    for i, smiles in enumerate(tqdm(smiles_list, desc="Processing molecules")):
        try:
            graph = mol_to_graph(smiles)
            if graph is not None:
                graphs.append(graph)
            else:
                failed += 1
        except Exception as e:
            failed += 1

        # Progress update
        if (i + 1) % 50000 == 0:
            print(f"Processed {i+1}/{len(smiles_list)}, Success: {len(graphs)}, Failed: {failed}")

    print(f"\nTotal graphs: {len(graphs)}, Failed: {failed}")

    # Create dataloader
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True, num_workers=0)

    return loader, len(graphs)


def pretrain(epochs=10, batch_size=64, lr=0.001, max_molecules=None, device=None):
    """
    Main pretraining function

    Args:
        epochs: Number of pretraining epochs
        batch_size: Batch size
        lr: Learning rate
        max_molecules: Limit number of molecules (None for all)
        device: torch device
    """

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    # Load data
    dataloader, num_molecules = load_zinc_data(max_molecules=max_molecules, batch_size=batch_size)

    # Initialize encoder
    model = MolecularEncoder(
        num_node_features=15,  # Same as our BBB model
        hidden_channels=128,
        num_heads=8,
        dropout=0.3
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    print(f"\nStarting pretraining on {num_molecules} molecules for {epochs} epochs...")
    print("=" * 60)

    history = {
        'mam_loss': [],
        'cl_loss': [],
        'lr': []
    }

    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        mam_loss, cl_loss = pretrain_epoch(model, dataloader, optimizer, device, epoch)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - start_time

        history['mam_loss'].append(mam_loss)
        history['cl_loss'].append(cl_loss)
        history['lr'].append(current_lr)

        total_loss = mam_loss + 0.5 * cl_loss

        print(f"Epoch {epoch}/{epochs} | MAM Loss: {mam_loss:.4f} | CL Loss: {cl_loss:.4f} | "
              f"LR: {current_lr:.6f} | Time: {elapsed:.1f}s")

        # Save best model
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mam_loss': mam_loss,
                'cl_loss': cl_loss,
            }, 'models/pretrained_encoder.pth')
            print(f"  -> Saved best model (loss: {total_loss:.4f})")

    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'history': history
    }, 'models/pretrained_encoder_final.pth')

    print("\n" + "=" * 60)
    print("Pretraining complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Models saved to models/pretrained_encoder.pth")

    return model, history


def transfer_weights_to_bbb_model(pretrained_path='models/pretrained_encoder.pth'):
    """
    Transfer pretrained weights to BBB prediction model
    """
    print("Transferring pretrained weights to BBB model...")

    # Load pretrained encoder
    checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)

    # Create BBB model
    bbb_model = AdvancedHybridBBBNet(
        num_node_features=15,
        hidden_channels=128,
        num_heads=8,
        dropout=0.3
    )

    # Get state dicts
    pretrained_dict = checkpoint['model_state_dict']
    bbb_dict = bbb_model.state_dict()

    # Filter pretrained weights (only layers with matching names/shapes)
    transfer_dict = {}
    for name, param in pretrained_dict.items():
        if name in bbb_dict and bbb_dict[name].shape == param.shape:
            transfer_dict[name] = param
            print(f"  Transferred: {name}")

    # Update BBB model weights
    bbb_dict.update(transfer_dict)
    bbb_model.load_state_dict(bbb_dict)

    print(f"\nTransferred {len(transfer_dict)}/{len(pretrained_dict)} layers")

    # Save as initialization for BBB training
    torch.save({
        'model_state_dict': bbb_model.state_dict(),
        'pretrained': True,
        'source': pretrained_path
    }, 'models/bbb_model_pretrained_init.pth')

    print("Saved initialized BBB model to models/bbb_model_pretrained_init.pth")

    return bbb_model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--max_molecules', type=int, default=None, help='Max molecules to use')
    args = parser.parse_args()

    # Set environment variable for OpenMP
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)

    # Run pretraining
    model, history = pretrain(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_molecules=args.max_molecules
    )

    # Transfer weights
    transfer_weights_to_bbb_model()

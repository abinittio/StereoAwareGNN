"""
Advanced Hybrid BBB GNN Model with Quantum Descriptors

This model extends the AdvancedHybridBBBNet to incorporate quantum
descriptors as additional node features.

Architecture:
- Input: 28 features (15 atomic + 13 quantum)
- Hybrid GNN: GAT -> GCN -> GraphSAGE -> GAT
- Output: BBB permeability prediction

The quantum descriptors are broadcast to all atoms in the molecule,
providing global molecular context to each node's local features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_mean_pool, global_max_pool


class AdvancedHybridBBBNetQuantum(nn.Module):
    """
    Advanced Hybrid GNN for BBB prediction with quantum descriptors.

    Combines multiple GNN architectures:
    - GAT (Graph Attention Network): Learns attention weights for neighbors
    - GCN (Graph Convolutional Network): Standard message passing
    - GraphSAGE: Sampling and aggregating node features

    Input features: 28 (15 atomic + 13 quantum)
    """

    def __init__(self, num_node_features=28, hidden_channels=128, num_heads=8,
                 dropout=0.3, num_classes=1):
        super().__init__()

        self.num_node_features = num_node_features
        self.hidden_channels = hidden_channels

        # === Layer 1: GAT (Graph Attention) ===
        self.gat1 = GATConv(
            num_node_features,
            hidden_channels,
            heads=num_heads,
            dropout=dropout,
            concat=True  # Output: hidden_channels * num_heads
        )
        self.bn1 = nn.BatchNorm1d(hidden_channels * num_heads)

        # === Layer 2: GCN (Graph Convolution) ===
        self.gcn1 = GCNConv(hidden_channels * num_heads, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        # === Layer 3: GraphSAGE ===
        self.sage1 = SAGEConv(hidden_channels, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)

        # === Layer 4: Another GAT for refinement ===
        self.gat2 = GATConv(
            hidden_channels,
            hidden_channels,
            heads=4,
            dropout=dropout,
            concat=False  # Output: hidden_channels
        )
        self.bn4 = nn.BatchNorm1d(hidden_channels)

        self.dropout = nn.Dropout(dropout)

        # === Readout and prediction MLPs ===
        # Combine mean and max pooling for richer graph representation
        self.mlp1 = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),  # *2 for concat of mean+max
            nn.ELU(),
            nn.BatchNorm1d(hidden_channels),
            nn.Dropout(dropout)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ELU(),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.Dropout(dropout)
        )

        self.mlp3 = nn.Sequential(
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.ELU(),
            nn.Dropout(dropout / 2)
        )

        # Final output layer - NO sigmoid (BCEWithLogitsLoss expects raw logits)
        self.mlp4 = nn.Sequential(
            nn.Linear(hidden_channels // 4, 32),
            nn.ELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(32, num_classes)
            # No Sigmoid here - BCEWithLogitsLoss expects raw logits
        )

    def forward(self, x, edge_index, batch):
        """
        Forward pass

        Args:
            x: Node features [num_nodes, 28]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment vector [num_nodes]

        Returns:
            Prediction logits [batch_size, 1]
        """
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

        # Layer 3: GraphSAGE
        x = self.sage1(x, edge_index)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.dropout(x)

        # Layer 4: GAT
        x = self.gat2(x, edge_index)
        x = self.bn4(x)
        x = F.elu(x)

        # Graph-level pooling (mean + max for richer representation)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        # MLP for prediction
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)

        return x


def get_model_info_quantum(model):
    """Get model information and parameter count"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    info = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'num_node_features': model.num_node_features,
        'hidden_channels': model.hidden_channels,
    }

    return info


def transfer_weights_from_pretrained(pretrained_path, quantum_model, device='cpu'):
    """
    Transfer weights from pretrained encoder to quantum model.

    Only transfers weights for layers with matching shapes.
    The first GAT layer won't transfer because input dimension changed
    (15 -> 28 features).
    """
    print("Transferring pretrained weights to quantum model...")

    checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
    pretrained_dict = checkpoint['model_state_dict']
    quantum_dict = quantum_model.state_dict()

    transferred = []
    skipped = []

    for name, param in pretrained_dict.items():
        if name in quantum_dict:
            if quantum_dict[name].shape == param.shape:
                quantum_dict[name] = param
                transferred.append(name)
            else:
                skipped.append(f"{name} (shape mismatch: {param.shape} vs {quantum_dict[name].shape})")
        else:
            skipped.append(f"{name} (not in quantum model)")

    quantum_model.load_state_dict(quantum_dict)

    print(f"Transferred {len(transferred)} layers:")
    for name in transferred[:5]:  # Show first 5
        print(f"  + {name}")
    if len(transferred) > 5:
        print(f"  ... and {len(transferred) - 5} more")

    print(f"\nSkipped {len(skipped)} layers (expected - input dimension changed)")

    return quantum_model


if __name__ == "__main__":
    # Test the quantum model
    print("Testing Advanced Hybrid BBB Net with Quantum Descriptors")
    print("=" * 60)

    # Create model
    model = AdvancedHybridBBBNetQuantum(
        num_node_features=28,  # 15 atomic + 13 quantum
        hidden_channels=128,
        num_heads=8,
        dropout=0.3
    )

    # Get model info
    info = get_model_info_quantum(model)
    print(f"\nModel Architecture:")
    print(f"  Input features: {info['num_node_features']}")
    print(f"  Hidden channels: {info['hidden_channels']}")
    print(f"  Total parameters: {info['total_params']:,}")
    print(f"  Trainable parameters: {info['trainable_params']:,}")

    # Test forward pass
    print("\nTesting forward pass...")

    # Create dummy data (10 nodes, 28 features)
    x = torch.randn(10, 28)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                               [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
    batch = torch.zeros(10, dtype=torch.long)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x, edge_index, batch)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output value: {output.item():.4f}")
    print(f"  Probability: {torch.sigmoid(output).item():.4f}")

    print("\nQuantum model working!")

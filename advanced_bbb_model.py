"""
Advanced Hybrid BBB Permeability Predictor
Combining GAT, GraphSAGE, and GCN architectures

Architecture: GAT → GCN → GraphSAGE → GAT → Dual Pooling → MLP
This multi-architecture approach captures:
- Local attention patterns (GAT)
- Graph convolutions (GCN)
- Neighborhood aggregation (SAGE)
- Final attention refinement (GAT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv, GCNConv, SAGEConv,
    global_mean_pool, global_max_pool, global_add_pool
)


class AdvancedHybridBBBNet(nn.Module):
    """
    State-of-the-art hybrid architecture for BBB prediction

    Architecture:
    1. Initial GAT layer (attention-based feature extraction)
    2. GCN layer (spectral graph convolution)
    3. GraphSAGE layer (inductive neighborhood aggregation)
    4. Final GAT layer (attention-based refinement)
    5. Triple pooling (mean + max + sum)
    6. Deep MLP with residual connections
    """

    def __init__(self,
                 num_node_features=15,  # Updated: 9 basic + 6 polarity features
                 hidden_channels=128,
                 num_heads=8,
                 dropout=0.3,
                 num_classes=1):
        super(AdvancedHybridBBBNet, self).__init__()

        # Layer 1: GAT - Attention mechanism for important features
        self.gat1 = GATConv(
            num_node_features,
            hidden_channels,
            heads=num_heads,
            dropout=dropout,
            concat=True
        )

        # Layer 2: GCN - Spectral graph convolution
        self.gcn = GCNConv(
            hidden_channels * num_heads,
            hidden_channels * 2
        )

        # Layer 3: GraphSAGE - Neighborhood aggregation
        self.sage = SAGEConv(
            hidden_channels * 2,
            hidden_channels,
            aggr='mean'
        )

        # Layer 4: GAT - Final attention-based refinement
        self.gat2 = GATConv(
            hidden_channels,
            hidden_channels // 2,
            heads=num_heads,
            dropout=dropout,
            concat=True
        )

        # Normalization layers
        self.norm1 = nn.LayerNorm(hidden_channels * num_heads)
        self.norm2 = nn.LayerNorm(hidden_channels * 2)
        self.norm3 = nn.LayerNorm(hidden_channels)
        self.norm4 = nn.LayerNorm((hidden_channels // 2) * num_heads)

        # Triple pooling features (mean + max + sum)
        pooled_features = (hidden_channels // 2) * num_heads * 3

        # Deep MLP with residual connections
        self.mlp1 = nn.Sequential(
            nn.Linear(pooled_features, 512),
            nn.LayerNorm(512),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.mlp3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Dropout(dropout / 2),
        )

        self.mlp4 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes)
            # No Sigmoid here - BCEWithLogitsLoss expects raw logits
            # Sigmoid is applied externally when needed for predictions
        )

        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        """
        Forward pass through hybrid architecture

        Args:
            x: Node features [num_nodes, num_node_features]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment [num_nodes]

        Returns:
            BBB permeability prediction [batch_size, 1]
        """
        # Layer 1: GAT with multi-head attention
        x = self.gat1(x, edge_index)
        x = self.norm1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2: GCN for spectral features
        x = self.gcn(x, edge_index)
        x = self.norm2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 3: GraphSAGE for neighborhood aggregation
        x = self.sage(x, edge_index)
        x = self.norm3(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 4: Final GAT for attention refinement
        x = self.gat2(x, edge_index)
        x = self.norm4(x)
        x = F.elu(x)

        # Triple global pooling (captures different graph aspects)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        x = torch.cat([x_mean, x_max, x_sum], dim=1)

        # Deep MLP with residual connections
        x1 = self.mlp1(x)
        x2 = self.mlp2(x1)
        x3 = self.mlp3(x2)
        out = self.mlp4(x3)

        return out.squeeze(-1)

    def get_embeddings(self, x, edge_index, batch):
        """Extract graph embeddings for visualization"""
        with torch.no_grad():
            x = self.gat1(x, edge_index)
            x = F.elu(self.norm1(x))
            x = self.gcn(x, edge_index)
            x = F.elu(self.norm2(x))
            x = self.sage(x, edge_index)
            x = F.elu(self.norm3(x))
            x = self.gat2(x, edge_index)
            x = F.elu(self.norm4(x))

            # Pool to get graph-level embeddings
            embedding = global_mean_pool(x, batch)
            return embedding


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model):
    """Get detailed model information"""
    total_params = count_parameters(model)

    info = {
        'total_parameters': total_params,
        'architecture': 'Hybrid GAT+GCN+GraphSAGE',
        'layers': [
            'GAT (8 heads, 128 channels)',
            'GCN (256 channels)',
            'GraphSAGE (128 channels)',
            'GAT (8 heads, 64 channels)',
            'Triple Pooling (mean+max+sum)',
            'MLP (512>256>128>64>1)'
        ],
        'pooling': 'Triple (mean + max + sum)',
        'normalization': 'LayerNorm',
        'activation': 'ELU',
        'dropout': 0.3
    }

    return info


if __name__ == "__main__":
    print("Advanced Hybrid BBB Permeability Predictor")
    print("=" * 70)

    # Initialize model
    model = AdvancedHybridBBBNet(
        num_node_features=15,  # 9 basic + 6 polarity features
        hidden_channels=128,
        num_heads=8,
        dropout=0.3
    )

    # Get model info
    info = get_model_info(model)

    print(f"\nModel: {info['architecture']}")
    print(f"Total Parameters: {info['total_parameters']:,}")
    print(f"\nArchitecture Layers:")
    for i, layer in enumerate(info['layers'], 1):
        print(f"  {i}. {layer}")

    print(f"\nPooling Strategy: {info['pooling']}")
    print(f"Normalization: {info['normalization']}")
    print(f"Activation: {info['activation']}")

    # Test forward pass
    num_nodes = 20
    x = torch.randn(num_nodes, 15)  # 15 features now
    edge_index = torch.randint(0, num_nodes, (2, 40))
    batch = torch.zeros(num_nodes, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        output = model(x, edge_index, batch)
        embedding = model.get_embeddings(x, edge_index, batch)

    print(f"\nTest Forward Pass:")
    print(f"  Input: {num_nodes} nodes with {x.shape[1]} features each")
    print(f"  Output: {output.shape} (BBB permeability score)")
    print(f"  Embedding: {embedding.shape} (graph representation)")
    print(f"  Prediction: {output.item():.4f}")

    print(f"\n✓ Advanced Hybrid Model Ready for Training!")
    print("=" * 70)

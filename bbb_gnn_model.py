import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader


class HybridGATSAGE(nn.Module):
    """
    Hybrid Graph Neural Network combining GAT and GraphSAGE

    Architecture:
    - Layer 1: GAT (attention mechanism for important features)
    - Layer 2: GraphSAGE (neighborhood aggregation)
    - Layer 3: GAT (final refinement with attention)
    - Global pooling: Combines mean and max pooling
    - MLP: Final prediction layers with dropout
    """

    def __init__(self,
                 num_node_features=9,
                 hidden_channels=128,
                 num_heads=8,
                 dropout=0.3):
        super(HybridGATSAGE, self).__init__()

        # GAT Layer 1: Multi-head attention for feature extraction
        self.gat1 = GATConv(
            num_node_features,
            hidden_channels,
            heads=num_heads,
            dropout=dropout,
            concat=True
        )

        # GraphSAGE Layer: Neighborhood aggregation
        self.sage = SAGEConv(
            hidden_channels * num_heads,
            hidden_channels,
            aggr='mean'
        )

        # GAT Layer 2: Attention-based refinement
        self.gat2 = GATConv(
            hidden_channels,
            hidden_channels // 2,
            heads=num_heads,
            dropout=dropout,
            concat=True
        )

        # Layer normalization (works with any batch size including 1)
        self.bn1 = nn.LayerNorm(hidden_channels * num_heads)
        self.bn2 = nn.LayerNorm(hidden_channels)
        self.bn3 = nn.LayerNorm((hidden_channels // 2) * num_heads)

        # MLP for final prediction (mean + max pooling = 2x features)
        pooled_features = (hidden_channels // 2) * num_heads * 2

        self.mlp = nn.Sequential(
            nn.Linear(pooled_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output between 0 and 1 for BBB permeability
        )

        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        """
        Forward pass through the hybrid GNN

        Args:
            x: Node features [num_nodes, num_node_features]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment vector [num_nodes]

        Returns:
            BBB permeability prediction [batch_size, 1]
        """
        # GAT Layer 1 with attention
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # GraphSAGE aggregation
        x = self.sage(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # GAT Layer 2 refinement
        x = self.gat2(x, edge_index)
        x = self.bn3(x)
        x = F.elu(x)

        # Global pooling (combine mean and max)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        # Final prediction through MLP
        x = self.mlp(x)

        return x.squeeze(-1)  # [batch_size]

    def get_attention_weights(self, x, edge_index):
        """
        Extract attention weights from GAT layers for interpretability

        Returns:
            Tuple of attention weights from GAT layers
        """
        with torch.no_grad():
            # First GAT layer attention
            _, (edge_index_gat1, alpha_gat1) = self.gat1(
                x, edge_index, return_attention_weights=True
            )

            # Pass through to second GAT
            x = self.gat1(x, edge_index)
            x = F.elu(x)
            x = self.sage(x, edge_index)
            x = F.elu(x)

            # Second GAT layer attention
            _, (edge_index_gat2, alpha_gat2) = self.gat2(
                x, edge_index, return_attention_weights=True
            )

        return (edge_index_gat1, alpha_gat1), (edge_index_gat2, alpha_gat2)


def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model architecture
    print("Testing Hybrid GAT+SAGE Model")
    print("=" * 60)

    model = HybridGATSAGE(
        num_node_features=9,
        hidden_channels=128,
        num_heads=8,
        dropout=0.3
    )

    print(f"Model Parameters: {count_parameters(model):,}")
    print(f"\nModel Architecture:")
    print(model)

    # Create dummy graph for testing
    num_nodes = 20
    x = torch.randn(num_nodes, 9)  # 9 node features
    edge_index = torch.randint(0, num_nodes, (2, 40))  # Random edges
    batch = torch.zeros(num_nodes, dtype=torch.long)  # Single graph

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x, edge_index, batch)

    print(f"\nTest Forward Pass:")
    print(f"Input nodes: {num_nodes}")
    print(f"Output shape: {output.shape}")
    print(f"Output value: {output.item():.4f}")
    print(f"Output range: [0, 1] (valid BBB permeability)")

    print("\nModel successfully initialized!")

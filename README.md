# BBB Permeability Prediction System

A breakthrough Graph Neural Network (GNN) system for predicting Blood-Brain Barrier (BBB) permeability of chemical compounds using a hybrid GAT+GraphSAGE architecture.

## Overview

This system uses state-of-the-art deep learning to predict whether molecules can cross the blood-brain barrier - a critical property for CNS drug development. The hybrid architecture combines Graph Attention Networks (GAT) for learning important molecular features and GraphSAGE for neighborhood aggregation.

## Architecture

### Hybrid GAT+SAGE Model
- **Layer 1**: GAT with 8 attention heads (feature extraction)
- **Layer 2**: GraphSAGE (neighborhood aggregation)
- **Layer 3**: GAT with 8 attention heads (refinement)
- **Pooling**: Combined mean + max global pooling
- **MLP**: 4-layer prediction head with dropout
- **Total Parameters**: 649,345

### Key Features
- Attention mechanisms for interpretability
- Batch normalization for stable training
- Early stopping to prevent overfitting
- Learning rate scheduling
- Comprehensive evaluation metrics (MAE, RMSE, R²)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Requirements
- PyTorch 2.9+
- PyTorch Geometric 2.7+
- RDKit (for molecular processing)
- scikit-learn
- pandas, numpy
- matplotlib, seaborn

## Dataset

The system includes a curated dataset of 42 compounds with known BBB permeability:
- **BBB+**: 20 compounds (high permeability) - e.g., Cocaine, Caffeine, Propranolol
- **BBB-**: 14 compounds (low/no permeability) - e.g., Glucose, Glutamic acid
- **BBB±**: 8 compounds (moderate permeability)

Permeability scores range from 0.0 (no BBB penetration) to 1.0 (high BBB penetration).

### BBB Compliance Rules
For optimal BBB permeability:
- Molecular Weight: 150-450 Da
- LogP: 1-5
- TPSA (Topological Polar Surface Area): <90 Ų
- H-bond Donors: ≤3
- H-bond Acceptors: ≤7

## Usage

### Web Interface (Recommended)

Launch the beautiful web interface for easy predictions:

```bash
# Option 1: Double-click the launcher
launch_web.bat

# Option 2: Command line
streamlit run app.py
```

The app will open at `http://localhost:8501` with:
- 🎨 Beautiful interactive UI
- 📊 Real-time visualizations
- 🔬 20+ pre-loaded molecules
- 💾 Export results (CSV/JSON)
- 📈 Comprehensive analysis

See [WEB_INTERFACE.md](WEB_INTERFACE.md) for detailed documentation.

### Training the Model

```bash
python train_gnn.py
```

This will:
1. Load and preprocess the BBB dataset
2. Train the hybrid GNN model
3. Save the best model to `models/best_model.pth`
4. Generate training visualizations

Training parameters:
- Epochs: 200 (with early stopping)
- Learning rate: 0.001
- Batch size: 4
- Optimizer: Adam
- Early stopping patience: 20 epochs

### Making Predictions

```python
from predict_bbb import BBBGNNPredictor

# Initialize predictor
predictor = BBBGNNPredictor(model_path='models/best_model.pth')

# Predict for a single molecule
result = predictor.predict('CN1C=NC2=C1C(=O)N(C(=O)N2C)C')  # Caffeine

print(f"BBB Score: {result['bbb_score']:.3f}")
print(f"Category: {result['category']}")  # BBB+, BBB±, or BBB-
print(f"LogP: {result['molecular_descriptors']['logp']:.2f}")
```

### Batch Predictions

```python
smiles_list = ['CCO', 'c1ccccc1', 'CC(=O)O']
results = predictor.predict_batch(smiles_list)

for result in results:
    print(f"{result['smiles']}: {result['bbb_score']:.3f} ({result['category']})")
```

### Command-line Testing

```bash
# Test with pre-defined compounds
python predict_bbb.py

# Test specific molecules
python test_cocaine.py
```

## Project Structure

```
BBB_System/
├── bbb_gnn_model.py         # Hybrid GAT+SAGE architecture
├── mol_to_graph.py          # SMILES to graph conversion
├── bbb_dataset.py           # Dataset loader with 42 compounds
├── train_gnn.py             # Training pipeline
├── predict_bbb.py           # Prediction interface
├── simple_bbb.py            # Baseline Random Forest model
├── test_cocaine.py          # Test script for various compounds
├── requirements.txt         # Dependencies
├── models/                  # Trained model checkpoints
│   ├── best_model.pth
│   ├── training_history.png
│   └── predictions.png
└── README.md
```

## Model Features

### Molecular Graph Representation
Each molecule is represented as a graph where:
- **Nodes**: Atoms with 9 features (atomic number, degree, charge, hybridization, aromaticity, etc.)
- **Edges**: Chemical bonds (bidirectional)

### Node Features (9 total)
1. Atomic number (normalized)
2. Degree (number of bonds)
3. Formal charge
4. Hybridization type
5. Aromaticity (binary)
6. In ring (binary)
7. Implicit valence
8. Explicit valence
9. Atomic mass (normalized)

## Performance

The model is evaluated on:
- **MAE (Mean Absolute Error)**: Average prediction error
- **RMSE (Root Mean Squared Error)**: Penalizes large errors
- **R² Score**: Variance explained by the model

Training includes:
- 80/20 train/validation split
- Early stopping with 20-epoch patience
- Learning rate reduction on plateau
- Gradient clipping for stability

## Molecular Descriptors

The system calculates traditional drug-likeness descriptors:
- Molecular Weight
- LogP (lipophilicity)
- TPSA (Topological Polar Surface Area)
- H-bond donors/acceptors
- Rotatable bonds
- Aromatic rings
- Lipinski's Rule of 5 violations

## Example Results

```
Cocaine:
  BBB Score: 0.892
  Category: BBB+ (HIGH BBB permeability)
  Molecular Weight: 275.3 Da
  LogP: 2.04
  TPSA: 38.8 Ų
  BBB Rule Compliant: True

Glucose:
  BBB Score: 0.105
  Category: BBB- (LOW BBB permeability)
  Molecular Weight: 180.2 Da
  LogP: -3.24
  TPSA: 110.4 Ų
  BBB Rule Compliant: False
  Warning: High TPSA (>90 Ų)
```

## Baseline Comparison

The system includes a baseline Random Forest model ([simple_bbb.py](simple_bbb.py)) using molecular descriptors. The GNN model learns directly from molecular structure and typically outperforms descriptor-based methods.

## Interpretability

The GAT layers provide attention weights showing which molecular substructures are important for BBB permeability predictions:

```python
# Extract attention weights (for analysis)
attention = model.get_attention_weights(x, edge_index)
```

## Contributing

Key areas for improvement:
1. Expand dataset with more diverse compounds
2. Implement external dataset loaders (e.g., BBBP from MoleculeNet)
3. Add molecular fingerprint fusion
4. Experiment with different GNN architectures (GCN, GIN, etc.)
5. Ensemble methods

## References

- Graph Attention Networks (GAT): Veličković et al., ICLR 2018
- GraphSAGE: Hamilton et al., NeurIPS 2017
- PyTorch Geometric: Fey & Lenssen, 2019
- RDKit: Open-source cheminformatics toolkit

## License

This is a research/educational project for blood-brain barrier permeability prediction.

## Citation

If you use this system in your research:

```bibtex
@software{bbb_gnn_predictor,
  title = {BBB Permeability Prediction System},
  author = {N Yasini-Ardekani},
  year = {2025},
  description = {Hybrid GAT+SAGE GNN for Blood-Brain Barrier Permeability Prediction}
}
```

---

**Built with PyTorch Geometric** | **Powered by Deep Learning** | **For CNS Drug Discovery**

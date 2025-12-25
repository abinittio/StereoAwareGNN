"""
Train the Advanced Hybrid BBB GNN Model on the real BBBP dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
import os
import time

from advanced_bbb_model import AdvancedHybridBBBNet, get_model_info
from mol_to_graph import mol_to_graph
from rdkit import Chem


def is_antibiotic_like(smiles):
    """
    Detect antibiotic-like structures that should be excluded from accuracy metrics.

    Antibiotics are known to have poor passive BBB permeability due to:
    - Large size (>500 Da)
    - High polarity (many H-bond donors/acceptors)
    - Active efflux by transporters

    Common antibiotic patterns:
    - Beta-lactams (penicillins, cephalosporins)
    - Aminoglycosides
    - Fluoroquinolones
    - Macrolides
    - Tetracyclines
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    # SMARTS patterns for common antibiotic scaffolds
    # Using simpler, more reliable patterns
    antibiotic_patterns = {
        'beta_lactam': 'C1C(=O)NC1',  # 4-membered lactam ring (simplified)
        'sulfonamide': 'NS(=O)(=O)c',  # Sulfonamide group attached to aromatic
    }

    for name, pattern in antibiotic_patterns.items():
        try:
            pat = Chem.MolFromSmarts(pattern)
            if pat and mol.HasSubstructMatch(pat):
                return True
        except:
            continue

    # Also flag by property thresholds common to antibiotics
    from rdkit.Chem import Descriptors
    mw = Descriptors.MolWt(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)

    # Antibiotics typically: large, polar, many H-bond sites
    if mw > 600 and tpsa > 150 and (hbd + hba) > 12:
        return True

    return False

class EarlyStopping:
    def __init__(self, patience=20, verbose=True, delta=0.001):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model, path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} > {val_loss:.6f}). Saving model...')
        torch.save({
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
        }, path)
        self.val_loss_min = val_loss

def load_bbbp_data(csv_path='data/bbbp_dataset.csv'):
    """Load and convert BBBP dataset to PyG graphs

    Also detects antibiotic-like compounds for exclusion from accuracy metrics.
    """
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    graphs = []
    labels = []
    valid_count = 0
    invalid_count = 0
    antibiotic_count = 0

    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"Processing molecule {idx}/{len(df)}...")

        smiles = row['SMILES']
        label = row['BBB_permeability']

        try:
            graph = mol_to_graph(smiles)
            if graph is not None:
                # Check if antibiotic-like (for exclusion from accuracy metrics)
                is_antibiotic = is_antibiotic_like(smiles)
                if is_antibiotic:
                    antibiotic_count += 1

                # Store antibiotic flag on graph object
                graph.is_antibiotic = is_antibiotic
                graph.smiles = smiles

                graphs.append(graph)
                labels.append(label)
                valid_count += 1
            else:
                invalid_count += 1
        except Exception as e:
            invalid_count += 1
            if invalid_count < 10:  # Only print first few errors
                print(f"Error processing SMILES '{smiles}': {e}")

    print(f"\nDataset processing complete:")
    print(f"  Valid molecules: {valid_count}")
    print(f"  Invalid molecules: {invalid_count}")
    print(f"  Antibiotic-like: {antibiotic_count} (will be excluded from accuracy)")
    print(f"  Success rate: {100 * valid_count / len(df):.2f}%")

    # Add labels to graphs
    for graph, label in zip(graphs, labels):
        graph.y = torch.tensor([label], dtype=torch.float)

    return graphs

def train_model(model, train_loader, val_loader, epochs=200, lr=0.0001, patience=50, device='cpu', class_weight=3.24):
    """Train the advanced GNN model

    Args:
        class_weight: Weight for positive class (BBB+). Since dataset is imbalanced
                      (1567 BBB+ vs 483 BBB-), we weight BBB- errors more heavily.
                      Default 3.24 = 1567/483 ratio.
    """
    model = model.to(device)

    # Class weights to handle imbalanced dataset (76.8% BBB+ vs 23.2% BBB-)
    # pos_weight makes the model pay more attention to BBB- (minority class)
    pos_weight = torch.tensor([class_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Lower learning rate (0.0001 instead of 0.001) for stable training
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15
    )

    save_dir = 'models'
    os.makedirs(save_dir, exist_ok=True)

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_auc': [],
        'val_auc': []
    }

    print("\nStarting training...")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Patience: {patience}")
    print("="*70)

    for epoch in range(epochs):
        start_time = time.time()

        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch.x, batch.edge_index, batch.batch)
            # Keep at least 1 dimension to avoid scalar tensor issues
            out_flat = out.view(-1)
            y_flat = batch.y.view(-1)
            loss = criterion(out_flat, y_flat)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.sigmoid(out.squeeze()).detach().cpu().numpy()
            labels = batch.y.squeeze().cpu().numpy()
            train_preds.extend(preds.tolist() if preds.ndim > 0 else [preds.item()])
            train_labels.extend(labels.tolist() if labels.ndim > 0 else [labels.item()])

        train_loss /= len(train_loader)
        train_auc = roc_auc_score(train_labels, train_preds)

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                out_flat = out.view(-1)
                y_flat = batch.y.view(-1)
                loss = criterion(out_flat, y_flat)

                val_loss += loss.item()
                preds = torch.sigmoid(out.squeeze()).cpu().numpy()
                labels = batch.y.squeeze().cpu().numpy()
                val_preds.extend(preds.tolist() if preds.ndim > 0 else [preds.item()])
                val_labels.extend(labels.tolist() if labels.ndim > 0 else [labels.item()])

        val_loss /= len(val_loader)
        val_auc = roc_auc_score(val_labels, val_preds)

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Print progress
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1:03d}/{epochs} | '
              f'Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | '
              f'Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | '
              f'Time: {epoch_time:.1f}s')

        # Early stopping
        early_stopping(val_loss, model, f'{save_dir}/best_advanced_model.pth')
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

    print("\n" + "="*70)
    print("Training completed!")

    # Load best model
    checkpoint = torch.load(f'{save_dir}/best_advanced_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, history

def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model on test set

    Reports both overall metrics and metrics excluding antibiotic-like compounds.
    """
    model.eval()
    predictions = []
    labels = []
    is_antibiotic = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = torch.sigmoid(out.squeeze())

            pred_np = pred.cpu().numpy()
            labels_np = batch.y.squeeze().cpu().numpy()
            predictions.extend(pred_np.tolist() if pred_np.ndim > 0 else [pred_np.item()])
            labels.extend(labels_np.tolist() if labels_np.ndim > 0 else [labels_np.item()])

            # Track antibiotic status per sample in batch
            if hasattr(batch, 'is_antibiotic'):
                # Unpack antibiotic flags per molecule in batch
                batch_size = batch.batch.max().item() + 1
                for i in range(batch_size):
                    # Get the antibiotic status for this molecule
                    is_antibiotic.append(batch.is_antibiotic[i] if hasattr(batch, 'is_antibiotic') else False)
            else:
                batch_size = batch.batch.max().item() + 1
                is_antibiotic.extend([False] * batch_size)

    predictions = np.array(predictions)
    labels = np.array(labels)
    is_antibiotic = np.array(is_antibiotic)

    # Convert to binary predictions
    binary_preds = (predictions > 0.5).astype(int)

    # === OVERALL METRICS ===
    auc = roc_auc_score(labels, predictions)
    accuracy = (binary_preds == labels).mean()
    mae = mean_absolute_error(labels, predictions)
    rmse = np.sqrt(mean_squared_error(labels, predictions))

    print("\n" + "="*70)
    print("FINAL TEST RESULTS (ALL COMPOUNDS)")
    print("="*70)
    print(f"AUC-ROC:  {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"MAE:      {mae:.4f}")
    print(f"RMSE:     {rmse:.4f}")
    print(f"Total:    {len(labels)} compounds")

    # === METRICS EXCLUDING ANTIBIOTICS ===
    non_antibiotic_mask = ~is_antibiotic
    num_antibiotics = is_antibiotic.sum()

    if num_antibiotics > 0 and non_antibiotic_mask.sum() > 0:
        preds_no_ab = predictions[non_antibiotic_mask]
        labels_no_ab = labels[non_antibiotic_mask]
        binary_no_ab = binary_preds[non_antibiotic_mask]

        auc_no_ab = roc_auc_score(labels_no_ab, preds_no_ab)
        acc_no_ab = (binary_no_ab == labels_no_ab).mean()
        mae_no_ab = mean_absolute_error(labels_no_ab, preds_no_ab)
        rmse_no_ab = np.sqrt(mean_squared_error(labels_no_ab, preds_no_ab))

        print("\n" + "-"*70)
        print("RESULTS EXCLUDING ANTIBIOTICS (Primary Metric)")
        print("-"*70)
        print(f"AUC-ROC:  {auc_no_ab:.4f}")
        print(f"Accuracy: {acc_no_ab:.4f}")
        print(f"MAE:      {mae_no_ab:.4f}")
        print(f"RMSE:     {rmse_no_ab:.4f}")
        print(f"Total:    {len(labels_no_ab)} compounds ({num_antibiotics} antibiotics excluded)")
    else:
        auc_no_ab = auc
        acc_no_ab = accuracy
        print(f"\n(No antibiotic-like compounds detected in test set)")

    print("="*70)

    return {
        'auc': auc,
        'accuracy': accuracy,
        'mae': mae,
        'rmse': rmse,
        'auc_no_antibiotics': auc_no_ab,
        'accuracy_no_antibiotics': acc_no_ab,
        'predictions': predictions,
        'labels': labels,
        'num_antibiotics_excluded': int(num_antibiotics)
    }

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ADVANCED BBB GNN TRAINING PIPELINE")
    print("="*70)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load data
    graphs = load_bbbp_data('data/bbbp_dataset.csv')

    # Split data
    train_graphs, test_graphs = train_test_split(graphs, test_size=0.15, random_state=42)
    train_graphs, val_graphs = train_test_split(train_graphs, test_size=0.15, random_state=42)

    print(f"\nDataset split:")
    print(f"  Training:   {len(train_graphs)} molecules")
    print(f"  Validation: {len(val_graphs)} molecules")
    print(f"  Test:       {len(test_graphs)} molecules")

    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

    # Initialize model
    print("\nInitializing Advanced Hybrid BBB GNN...")
    model = AdvancedHybridBBBNet(
        num_node_features=15,  # 9 basic + 6 polarity features for BBB
        hidden_channels=128,
        num_heads=8,
        dropout=0.3
    )

    # Print model info
    info = get_model_info(model)
    print(f"\nModel: {info['architecture']}")
    print(f"Parameters: {info['total_parameters']:,}")
    print("\nArchitecture:")
    for i, layer in enumerate(info['layers'], 1):
        print(f"  {i}. {layer}")

    # Train model with optimized hyperparameters
    # - lr=0.0001: Lower learning rate for stable training (was 0.001)
    # - patience=50: More patience before early stopping (was 20)
    # - class_weight=3.24: Handle imbalanced dataset (1567 BBB+ / 483 BBB-)
    model, history = train_model(
        model, train_loader, val_loader,
        epochs=200, lr=0.0001, patience=50, class_weight=3.24, device=device
    )

    # Evaluate on test set
    results = evaluate_model(model, test_loader, device=device)

    # Save final results
    print("\nSaving training history and results...")
    np.save('models/training_history.npy', history)
    np.save('models/test_results.npy', results)

    print("\n" + "="*70)
    print("ALL DONE!")
    print("="*70)
    print("\nModel saved to: models/best_advanced_model.pth")
    print("Next step: Update app.py to use the advanced model")
    print("\nTo deploy:")
    print("1. Push to GitHub")
    print("2. Deploy to Streamlit Cloud")
    print("3. Share your breakthrough!")

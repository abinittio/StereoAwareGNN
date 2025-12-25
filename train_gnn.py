import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from bbb_gnn_model import HybridGATSAGE, count_parameters
from bbb_dataset import load_bbb_dataset


class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience=15, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'  EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print('  Early stopping triggered!')
        else:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict()
            self.counter = 0


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []

    for batch in train_loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        output = model(batch.x, batch.edge_index, batch.batch)

        loss = criterion(output, batch.y)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        all_preds.extend(output.detach().cpu().numpy())
        all_targets.extend(batch.y.detach().cpu().numpy())

    avg_loss = total_loss / len(train_loader.dataset)

    # Calculate MAE (Mean Absolute Error)
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))

    return avg_loss, mae


def evaluate(model, val_loader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.batch)

            loss = criterion(output, batch.y)

            total_loss += loss.item() * batch.num_graphs
            all_preds.extend(output.cpu().numpy())
            all_targets.extend(batch.y.cpu().numpy())

    avg_loss = total_loss / len(val_loader.dataset)

    # Calculate metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))

    # R² score
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return avg_loss, mae, rmse, r2, all_preds, all_targets


def plot_training_history(history, save_path='training_history.png'):
    """Plot training metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MAE plot
    axes[1].plot(history['train_mae'], label='Train MAE', marker='o')
    axes[1].plot(history['val_mae'], label='Val MAE', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Mean Absolute Error')
    axes[1].set_title('Training and Validation MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")


def plot_predictions(y_true, y_pred, save_path='predictions.png'):
    """Plot predicted vs actual values"""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, s=100, edgecolors='black')
    plt.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect prediction')

    plt.xlabel('Actual BBB Permeability', fontsize=12)
    plt.ylabel('Predicted BBB Permeability', fontsize=12)
    plt.title('GNN Predictions vs Actual Values', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Predictions plot saved to {save_path}")


def train_model(model,
                train_loader,
                val_loader,
                epochs=200,
                lr=0.001,
                weight_decay=1e-5,
                patience=15,
                device='cpu',
                save_dir='models'):
    """
    Complete training pipeline with early stopping and checkpointing
    """
    Path(save_dir).mkdir(exist_ok=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': [],
    }

    print(f"\nStarting training...")
    print(f"Device: {device}")
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Learning rate: {lr}")
    print(f"Weight decay: {weight_decay}")
    print("=" * 70)

    best_val_mae = float('inf')
    best_epoch = 0

    for epoch in range(epochs):
        # Train
        train_loss, train_mae = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_mae, val_rmse, val_r2, val_preds, val_targets = evaluate(
            model, val_loader, criterion, device
        )

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} MAE: {train_mae:.4f} | "
                  f"Val Loss: {val_loss:.4f} MAE: {val_mae:.4f} RMSE: {val_rmse:.4f} R²: {val_r2:.3f}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae,
                'val_loss': val_loss,
                'val_r2': val_r2,
            }, f'{save_dir}/best_model.pth')

        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"\nTraining stopped early at epoch {epoch+1}")
            break

    print("=" * 70)
    print(f"\nTraining completed!")
    print(f"Best validation MAE: {best_val_mae:.4f} at epoch {best_epoch+1}")

    # Load best model
    checkpoint = torch.load(f'{save_dir}/best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final evaluation
    val_loss, val_mae, val_rmse, val_r2, val_preds, val_targets = evaluate(
        model, val_loader, criterion, device
    )

    print(f"\nFinal Validation Metrics:")
    print(f"  MAE:  {val_mae:.4f}")
    print(f"  RMSE: {val_rmse:.4f}")
    print(f"  R²:   {val_r2:.4f}")

    # Plot results
    plot_training_history(history, save_path=f'{save_dir}/training_history.png')
    plot_predictions(val_targets, val_preds, save_path=f'{save_dir}/predictions.png')

    return model, history


if __name__ == "__main__":
    print("BBB GNN Training Pipeline")
    print("=" * 70)

    # Load dataset
    train_graphs, val_graphs, df = load_bbb_dataset(validation_split=0.2)

    # Create data loaders (drop_last=True to avoid batch size 1)
    train_loader = DataLoader(train_graphs, batch_size=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_graphs, batch_size=4, shuffle=False, drop_last=True)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridGATSAGE(
        num_node_features=9,
        hidden_channels=128,
        num_heads=8,
        dropout=0.3
    ).to(device)

    print(f"\nModel architecture:")
    print(f"  Total parameters: {count_parameters(model):,}")
    print(f"  Device: {device}")

    # Train model
    trained_model, history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=200,
        lr=0.001,
        weight_decay=1e-5,
        patience=20,
        device=device
    )

    print("\nTraining complete! Model saved to models/best_model.pth")

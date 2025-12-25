"""
Save the best model so far from the 4-model comparison
"""
import shutil
import os
from datetime import datetime

models_dir = 'models'
backup_dir = 'models/backup_best'

# Create backup directory
os.makedirs(backup_dir, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Models to backup
models_to_backup = [
    'best_baseline_v2.pth',
    'best_pretrained_v2.pth',
    'best_quantum_v2.pth',
    'best_combined_v2.pth',
    'full_comparison_results_v2.npy'
]

print(f"Backing up models at {timestamp}")
print("="*60)

for model_file in models_to_backup:
    src = os.path.join(models_dir, model_file)
    if os.path.exists(src):
        # Create timestamped backup
        dst = os.path.join(backup_dir, f"{timestamp}_{model_file}")
        shutil.copy2(src, dst)
        print(f"Saved: {model_file}")

        # Also keep a "latest" copy
        latest = os.path.join(backup_dir, f"latest_{model_file}")
        shutil.copy2(src, latest)
    else:
        print(f"Not found yet: {model_file}")

print("="*60)
print(f"Backup complete! Files saved to {backup_dir}")

# Show current best results if available
results_file = os.path.join(models_dir, 'full_comparison_results_v2.npy')
if os.path.exists(results_file):
    import numpy as np
    results = np.load(results_file, allow_pickle=True).item()
    print("\nCurrent Results:")
    for key, data in results.items():
        if 'test_auc' in data:
            print(f"  {key}: Test AUC = {data['test_auc']:.4f}")

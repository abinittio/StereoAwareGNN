import numpy as np

# Load comparison results
results = np.load('models/full_comparison_results.npy', allow_pickle=True).item()

baseline = results['baseline']['test_metrics']
pretrained = results['pretrained']['test_metrics']
quantum = results['quantum']['test_metrics']

print("\n" + "="*65)
print("           FULL BBB MODEL COMPARISON RESULTS")
print("="*65)

print("\nModel           | Test AUC | Accuracy | Precision | Recall | F1")
print("-"*65)
print(f"Baseline        | {baseline['auc']:.4f}   | {baseline['accuracy']*100:.1f}%    | {baseline['precision']:.4f}    | {baseline['recall']:.4f} | {baseline['f1']:.4f}")
print(f"Pretrained      | {pretrained['auc']:.4f}   | {pretrained['accuracy']*100:.1f}%    | {pretrained['precision']:.4f}    | {pretrained['recall']:.4f} | {pretrained['f1']:.4f}")
print(f"Quantum         | {quantum['auc']:.4f}   | {quantum['accuracy']*100:.1f}%    | {quantum['precision']:.4f}    | {quantum['recall']:.4f} | {quantum['f1']:.4f}")
print("-"*65)

# Improvements
pt_improv = ((pretrained['auc'] - baseline['auc']) / baseline['auc']) * 100
q_improv = ((quantum['auc'] - baseline['auc']) / baseline['auc']) * 100

print(f"\nImprovement over Baseline:")
print(f"  Pretrained: +{pt_improv:.2f}% AUC")
print(f"  Quantum:    +{q_improv:.2f}% AUC")

# Best model
models = [("Baseline", baseline['auc']), ("Pretrained", pretrained['auc']), ("Quantum", quantum['auc'])]
best = max(models, key=lambda x: x[1])

print("\n" + "="*65)
print(f"  BEST MODEL: {best[0].upper()} with Test AUC = {best[1]:.4f}")
print("="*65)

# Validation AUC comparison
print("\nValidation AUC (during training):")
print(f"  Baseline:   {results['baseline']['best_val_auc']:.4f}")
print(f"  Pretrained: {results['pretrained']['best_val_auc']:.4f}")
print(f"  Quantum:    {results['quantum']['best_val_auc']:.4f}")

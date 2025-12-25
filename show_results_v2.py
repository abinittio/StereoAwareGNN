import numpy as np
import os

# Load results
results_file = 'models/full_comparison_results.npy'
if os.path.exists(results_file):
    results = np.load(results_file, allow_pickle=True).item()

    print("\n" + "="*80)
    print("4-MODEL COMPARISON RESULTS: BBB Permeability Prediction")
    print("="*80)

    model_names = ['Baseline', 'Pretrained', 'Quantum-only', 'Pretrained+Quantum']

    # Print detailed results for each model
    for i, name in enumerate(model_names, 1):
        key = f'model{i}'
        if key in results:
            print(f"\n{'─'*80}")
            print(f"MODEL {i}: {name}")
            print(f"{'─'*80}")

            model_results = results[key]

            print(f"  Test AUC:       {model_results['test_auc']:.4f}")
            print(f"  Test Accuracy:  {model_results['test_acc']:.4f} ({model_results['test_acc']*100:.1f}%)")
            print(f"  Precision:      {model_results['precision']:.4f}")
            print(f"  Recall:         {model_results['recall']:.4f}")
            print(f"  F1 Score:       {model_results['f1']:.4f}")
            print(f"  Best Val AUC:   {model_results['best_val_auc']:.4f} (epoch {model_results['best_epoch']})")

    # Find winners for each metric
    print(f"\n{'='*80}")
    print("METRIC WINNERS")
    print(f"{'='*80}")

    metrics = {
        'Best Overall (AUC)': 'test_auc',
        'Best Recall': 'recall',
        'Best Precision': 'precision',
        'Best F1 Score': 'f1',
        'Best Accuracy': 'test_acc'
    }

    for metric_name, metric_key in metrics.items():
        scores = []
        for i in range(1, 5):
            key = f'model{i}'
            if key in results:
                scores.append((model_names[i-1], results[key][metric_key]))

        winner = max(scores, key=lambda x: x[1])
        print(f"  {metric_name:20s}: {winner[0]:20s} ({winner[1]:.4f})")

    # Calculate improvements
    print(f"\n{'='*80}")
    print("IMPROVEMENTS OVER BASELINE")
    print(f"{'='*80}")

    if 'model1' in results:
        baseline_auc = results['model1']['test_auc']

        for i, name in enumerate(model_names[1:], 2):
            key = f'model{i}'
            if key in results:
                model_auc = results[key]['test_auc']
                improvement = ((model_auc - baseline_auc) / baseline_auc) * 100
                abs_improvement = model_auc - baseline_auc
                print(f"  {name:20s}: {improvement:+6.2f}% ({abs_improvement:+.4f} AUC points)")

    print(f"\n{'='*80}\n")

else:
    print(f"Results file not found: {results_file}")

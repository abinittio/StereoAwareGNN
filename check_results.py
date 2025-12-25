import numpy as np
import os

results_file = 'models/full_comparison_results.npy'
if os.path.exists(results_file):
    results = np.load(results_file, allow_pickle=True).item()
    print("Keys in results:", results.keys())
    print("\nFull results:")
    for key, value in results.items():
        print(f"\n{key}:")
        print(value)
else:
    print("Results file not found")

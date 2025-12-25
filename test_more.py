from simple_bbb import BBBPredictor
import pandas as pd

# Test with more diverse molecules
test_molecules = [
    ('CCO', 'Ethanol', 'Expected: High BBB'),
    ('c1ccccc1', 'Benzene', 'Expected: High BBB'),  
    ('CC(=O)O', 'Acetic acid', 'Expected: Low BBB'),
    ('CCN(CC)CC', 'Triethylamine', 'Expected: High BBB'),
    ('C(C(=O)O)C(=O)O', 'Malonic acid', 'Expected: Very Low BBB')
]

predictor = BBBPredictor()

# Quick training data
data = {'SMILES': ['CCO', 'c1ccccc1', 'CC(=O)O'], 'BBB_permeability': [0.8, 0.9, 0.3]}
predictor.prepare_data(pd.DataFrame(data))
predictor.train()

print("🧬 BBB PERMEABILITY PREDICTIONS:")
print("=" * 50)

for smiles, name, expected in test_molecules:
    result = predictor.predict(smiles)
    if result:
        print(f"{name:15} | BBB: {result['prediction']:.3f} | {expected}")
        print(f"                MW: {result['molecular_descriptors']['mol_weight']:.1f} | LogP: {result['molecular_descriptors']['logp']:.2f}")
    else:
        print(f"{name:15} | FAILED TO PROCESS")
    print("-" * 50)
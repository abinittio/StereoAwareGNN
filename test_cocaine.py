from simple_bbb import BBBPredictor
import pandas as pd

# Initialize predictor with training data
predictor = BBBPredictor()
data = {'SMILES': ['CCO', 'c1ccccc1', 'CC(=O)O'], 'BBB_permeability': [0.8, 0.9, 0.3]}
predictor.prepare_data(pd.DataFrame(data))
predictor.train()

# Test cocaine and related compounds
test_compounds = [
    ('COC(=O)C1C(CC2CC1N2C)c3cccc(c3)OC', 'Cocaine', 'Known BBB+'),
    ('CCO', 'Ethanol', 'Known BBB+'),
    ('CC(=O)O', 'Acetic Acid', 'Known BBB-'),
    ('c1ccc(cc1)CCN', 'Phenethylamine', 'Known BBB+'),
    ('C1CCC(CC1)C(C2CCCCC2)N', 'Phencyclidine skeleton', 'BBB+')
]

print("🧬 BBB PERMEABILITY ANALYSIS")
print("=" * 70)

for smiles, name, known_status in test_compounds:
    result = predictor.predict(smiles)
    if result:
        bbb_score = result['prediction']
        descriptors = result['molecular_descriptors']
        
        print(f"\n{name} ({known_status}):")
        print(f"  BBB Prediction: {bbb_score:.3f}")
        print(f"  Molecular Weight: {descriptors['mol_weight']:.1f} Da")
        print(f"  LogP: {descriptors['logp']:.2f}")
        print(f"  TPSA: {descriptors['tpsa']:.1f} Ų")
        print(f"  H-bond donors: {descriptors['num_hbd']}")
        print(f"  H-bond acceptors: {descriptors['num_hba']}")
        
        # Interpretation
        if bbb_score > 0.6:
            interpretation = "HIGH BBB permeability predicted ✅"
        elif bbb_score > 0.4:
            interpretation = "MODERATE BBB permeability predicted ⚠️"
        else:
            interpretation = "LOW BBB permeability predicted ❌"
        
        print(f"  → {interpretation}")
    else:
        print(f"\n{name}: FAILED TO PROCESS SMILES")
    
    print("-" * 70)

print("\n📊 MOLECULAR PROPERTY ANALYSIS:")
print("For BBB permeability, optimal ranges are typically:")
print("• Molecular Weight: 150-450 Da")
print("• LogP: 1-3")
print("• TPSA: <90 Ų")
print("• H-bond donors: ≤3")
print("• H-bond acceptors: ≤7")
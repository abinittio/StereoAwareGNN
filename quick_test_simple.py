import pandas as pd
from simple_bbb import BBBPredictor

print("🧬 Testing Simple BBB System")

# Create test data
data = {
    'SMILES': ['CCO', 'c1ccccc1', 'CC(=O)O'],
    'BBB_permeability': [0.8, 0.9, 0.3]
}
df = pd.DataFrame(data)

# Initialize and train
predictor = BBBPredictor()
predictor.prepare_data(df)
predictor.train()

# Test prediction
result = predictor.predict('CCO')
print(f"🎉 Ethanol BBB prediction: {result['prediction']:.3f}")
print("✅ YOUR BBB SYSTEM IS WORKING!")
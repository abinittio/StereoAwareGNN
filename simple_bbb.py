import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors

class SimpleBBBPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.trained = False
        print("Simple BBB Predictor initialized!")
    
    def smiles_to_features(self, smiles):
        """Convert SMILES to simple molecular features"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            features = [
                Descriptors.MolWt(mol),           # Molecular weight
                Descriptors.MolLogP(mol),         # LogP
                Descriptors.TPSA(mol),            # Polar surface area
                Descriptors.NumHDonors(mol),      # H-bond donors
                Descriptors.NumHAcceptors(mol),   # H-bond acceptors
                mol.GetNumAtoms(),                # Number of atoms
                mol.GetNumBonds(),                # Number of bonds
            ]
            return features
        except:
            return None
    
    def prepare_data(self, df, smiles_col='SMILES', target_col='BBB_permeability'):
        X = []
        y = []
        
        for idx, row in df.iterrows():
            features = self.smiles_to_features(row[smiles_col])
            if features is not None:
                X.append(features)
                y.append(row[target_col])
        
        self.X = np.array(X)
        self.y = np.array(y)
        print(f"Prepared {len(self.X)} molecules for training")
        return self
    
    def train(self, **kwargs):
        """Train the model"""
        if hasattr(self, 'X') and hasattr(self, 'y'):
            self.model.fit(self.X, self.y)
            self.trained = True
            print("Training completed!")
            return [], []
        else:
            print("No data prepared for training!")
            return [], []
    
    def predict(self, smiles):
        """Predict BBB permeability"""
        if not self.trained:
            return {'prediction': 0.5, 'molecular_descriptors': {}}
        
        features = self.smiles_to_features(smiles)
        if features is None:
            return None
        
        prediction = self.model.predict([features])[0]
        
        return {
            'prediction': prediction,
            'molecular_descriptors': {
                'mol_weight': features[0],
                'logp': features[1],
                'tpsa': features[2],
                'num_hbd': features[3],
                'num_hba': features[4]
            }
        }

# Alias for compatibility
BBBPredictor = SimpleBBBPredictor

print("Simple BBB system loaded successfully!")
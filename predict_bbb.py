import torch
from pathlib import Path

from advanced_bbb_model import AdvancedHybridBBBNet
from mol_to_graph import mol_to_graph, get_molecular_descriptors


class BBBGNNPredictor:
    """
    Production-ready BBB permeability predictor using trained GNN model
    Uses the Advanced Hybrid Architecture (GAT+GCN+GraphSAGE) trained on 2,039 compounds
    """

    def __init__(self, model_path='models/best_advanced_model.pth', device=None):
        """
        Initialize the predictor with a trained model

        Args:
            model_path: Path to saved model checkpoint
            device: torch device (auto-detects if None)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Initialize advanced hybrid model (GAT+GCN+GraphSAGE, 1.37M params)
        self.model = AdvancedHybridBBBNet(
            num_node_features=15,  # 9 basic + 6 polarity features for BBB
            hidden_channels=128,
            num_heads=8,
            dropout=0.3
        ).to(self.device)

        # Load trained weights
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.trained = True
            print(f"Loaded trained model from {model_path}")
            val_loss = checkpoint.get('val_loss', 'N/A')
            if isinstance(val_loss, float):
                print(f"  Validation Loss: {val_loss:.4f}")
            else:
                print(f"  Validation Loss: {val_loss}")
        else:
            self.trained = False
            print(f"Warning: Model file not found at {model_path}")
            print("Model initialized but not trained. Predictions will be random.")

    def predict(self, smiles, return_details=True):
        """
        Predict BBB permeability for a molecule

        Args:
            smiles: SMILES string of molecule
            return_details: If True, return detailed analysis

        Returns:
            dict with prediction and optional details
        """
        if not self.trained:
            print("Warning: Using untrained model!")

        # Convert SMILES to graph
        graph = mol_to_graph(smiles)

        if graph is None:
            return {
                'success': False,
                'error': 'Invalid SMILES string',
                'smiles': smiles
            }

        # Move to device
        graph = graph.to(self.device)

        # Create batch tensor for single molecule
        batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=self.device)

        # Predict (ensure model is in eval mode)
        self.model.eval()
        with torch.no_grad():
            # Temporarily disable all batch norm layers for single-molecule prediction
            for module in self.model.modules():
                if isinstance(module, torch.nn.BatchNorm1d):
                    module.training = False

            logits = self.model(graph.x, graph.edge_index, batch)
            # Apply sigmoid since model outputs raw logits (for BCEWithLogitsLoss compatibility)
            prediction = torch.sigmoid(logits)

        bbb_score = prediction.item()

        # Categorize prediction
        if bbb_score >= 0.6:
            category = 'BBB+'
            interpretation = 'HIGH BBB permeability'
        elif bbb_score >= 0.4:
            category = 'BBB±'
            interpretation = 'MODERATE BBB permeability'
        else:
            category = 'BBB-'
            interpretation = 'LOW BBB permeability'

        result = {
            'success': True,
            'smiles': smiles,
            'bbb_score': bbb_score,
            'category': category,
            'interpretation': interpretation,
        }

        if return_details:
            # Get molecular descriptors
            descriptors = get_molecular_descriptors(smiles)

            if descriptors:
                result['molecular_descriptors'] = descriptors

                # Check BBB rules compliance
                result['bbb_rule_compliant'] = descriptors['bbb_rule_compliant']

                # Add warnings if any
                warnings = []
                if descriptors['molecular_weight'] > 450:
                    warnings.append('High molecular weight (>450 Da)')
                if descriptors['tpsa'] > 90:
                    warnings.append('High TPSA (>90 A^2)')
                if descriptors['num_h_donors'] > 3:
                    warnings.append('High H-bond donors (>3)')
                if descriptors['logp'] < 1 or descriptors['logp'] > 5:
                    warnings.append(f'LogP outside optimal range (1-5): {descriptors["logp"]:.2f}')

                result['warnings'] = warnings

        return result

    def predict_batch(self, smiles_list):
        """
        Predict BBB permeability for multiple molecules

        Args:
            smiles_list: List of SMILES strings

        Returns:
            List of prediction results
        """
        results = []
        for smiles in smiles_list:
            result = self.predict(smiles, return_details=True)
            results.append(result)
        return results


def format_prediction_output(result):
    """Pretty print prediction results"""
    if not result['success']:
        print(f"FAILED: {result['error']}")
        return

    print(f"\nSMILES: {result['smiles']}")
    print(f"BBB Permeability Score: {result['bbb_score']:.3f}")
    print(f"Category: {result['category']} ({result['interpretation']})")

    if 'molecular_descriptors' in result:
        desc = result['molecular_descriptors']
        print(f"\nMolecular Properties:")
        print(f"  Molecular Weight: {desc['molecular_weight']:.1f} Da")
        print(f"  LogP: {desc['logp']:.2f}")
        print(f"  TPSA: {desc['tpsa']:.1f} A^2")
        print(f"  H-bond Donors: {desc['num_h_donors']}")
        print(f"  H-bond Acceptors: {desc['num_h_acceptors']}")
        print(f"  BBB Rule Compliant: {desc['bbb_rule_compliant']}")

    if result.get('warnings'):
        print(f"\nWarnings:")
        for warning in result['warnings']:
            print(f"  - {warning}")

    print("-" * 70)


if __name__ == "__main__":
    print("BBB GNN Predictor - Testing")
    print("=" * 70)

    # Initialize predictor
    predictor = BBBGNNPredictor()

    # Test compounds
    test_compounds = [
        ('COC(=O)C1C(CC2CC1N2C)c3cccc(c3)OC', 'Cocaine'),
        ('CCO', 'Ethanol'),
        ('CC(=O)O', 'Acetic Acid'),
        ('c1ccc(cc1)CCN', 'Phenethylamine'),
        ('CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'Caffeine'),
        ('C(C(=O)O)N', 'Glycine'),
        ('c1ccccc1', 'Benzene'),
        ('CC(C)NCC(COc1ccccc1)O', 'Propranolol'),
    ]

    print(f"\nTesting {len(test_compounds)} compounds:")
    print("=" * 70)

    for smiles, name in test_compounds:
        print(f"\n{name}:")
        result = predictor.predict(smiles, return_details=True)
        format_prediction_output(result)

    # Batch prediction
    print("\n\nBatch Prediction Test:")
    print("=" * 70)
    smiles_batch = [s for s, _ in test_compounds[:3]]
    batch_results = predictor.predict_batch(smiles_batch)

    print(f"\nBatch results:")
    for i, (result, (_, name)) in enumerate(zip(batch_results, test_compounds[:3])):
        print(f"{i+1}. {name}: {result['bbb_score']:.3f} ({result['category']})")

    print("\nPrediction system ready!")

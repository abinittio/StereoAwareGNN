"""
Save processed ZINC graphs with stereoisomers for resumable pretraining.

This script:
1. Loads ZINC molecules
2. Expands with stereoisomers
3. Converts to graphs with 21 features
4. Saves to disk as a pickle file

This allows resuming pretraining without re-doing graph conversion.
"""

import torch
import pickle
import os
import sys
from rdkit import Chem
from rdkit.Chem import Descriptors

from zinc_stereo_pretraining import (
    load_zinc_subset,
    expand_zinc_with_stereoisomers,
    count_stereocenters
)
from mol_to_graph_enhanced import batch_smiles_to_graphs_enhanced


def save_zinc_stereo_graphs(num_molecules=50000, output_path='data/zinc_stereo_graphs.pkl'):
    """
    Process and save ZINC graphs with stereoisomers.
    """
    print("=" * 70)
    print("SAVING ZINC STEREO GRAPHS FOR RESUMABLE PRETRAINING")
    print("=" * 70)
    sys.stdout.flush()

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load ZINC
    zinc_smiles = load_zinc_subset(num_molecules)
    if not zinc_smiles:
        print("ERROR: Could not load ZINC")
        return None

    # Expand with stereoisomers
    print("\nExpanding with stereoisomers...")
    expanded_smiles = expand_zinc_with_stereoisomers(zinc_smiles, max_isomers_per_mol=4)

    # Convert to graphs
    print("\nConverting to graphs (21 features)...")
    sys.stdout.flush()

    graphs = batch_smiles_to_graphs_enhanced(
        expanded_smiles,
        y_list=None,
        include_quantum=False,
        include_stereo=True,
        use_dft=False,
        verbose=True
    )

    # Add self-supervised targets
    print("\nAdding self-supervised targets...")
    valid_graphs = []

    for i, (smiles, graph) in enumerate(zip(expanded_smiles, graphs)):
        if graph is None:
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        # Targets
        graph.mol_weight = torch.tensor([Descriptors.MolWt(mol) / 500.0], dtype=torch.float)
        graph.atom_count = torch.tensor([mol.GetNumAtoms() / 50.0], dtype=torch.float)

        chiral, ez = count_stereocenters(smiles)
        has_stereo = 1.0 if (chiral > 0 or ez > 0) else 0.0
        graph.has_stereo = torch.tensor([has_stereo], dtype=torch.float)
        graph.smiles = smiles

        valid_graphs.append(graph)

        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1}/{len(expanded_smiles)}")
            sys.stdout.flush()

    print(f"\nValid graphs: {len(valid_graphs)}")

    # Save to disk
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump({
            'graphs': valid_graphs,
            'num_original': num_molecules,
            'num_expanded': len(expanded_smiles),
            'num_valid': len(valid_graphs),
        }, f)

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved! File size: {file_size:.1f} MB")

    return output_path


def load_zinc_stereo_graphs(input_path='data/zinc_stereo_graphs.pkl'):
    """Load saved ZINC graphs."""
    print(f"Loading graphs from {input_path}...")

    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Loaded {data['num_valid']} graphs")
    print(f"  Original ZINC: {data['num_original']}")
    print(f"  After expansion: {data['num_expanded']}")

    return data['graphs']


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true', help='Save graphs')
    parser.add_argument('--load', action='store_true', help='Load and verify')
    parser.add_argument('--num', type=int, default=50000, help='Number of ZINC molecules')
    args = parser.parse_args()

    if args.save:
        save_zinc_stereo_graphs(args.num)
    elif args.load:
        graphs = load_zinc_stereo_graphs()
        print(f"\nSample graph:")
        print(f"  Features: {graphs[0].x.shape}")
        print(f"  SMILES: {graphs[0].smiles}")

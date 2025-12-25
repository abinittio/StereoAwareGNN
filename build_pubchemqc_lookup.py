"""
Build PubChemQC Lookup for BBBP Dataset

This script:
1. Loads all SMILES from the BBBP dataset
2. Streams through PubChemQC B3LYP/6-31G* database
3. Caches matches for use in training

The PubChemQC database contains 86 million molecules with real DFT-computed
quantum properties (HOMO, LUMO, dipole moment, etc.) from B3LYP/6-31G* calculations.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pubchemqc_integration import PubChemQCIntegration, StereochemistryEncoder


def load_bbbp_smiles():
    """Load all SMILES from BBBP dataset"""
    data_paths = [
        'data/bbbp_dataset.csv',
        'data/BBBP.csv',
        'data/bbbp.csv',
        'BBBP.csv'
    ]

    for path in data_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Find SMILES column
            smiles_col = None
            for col in df.columns:
                if 'smiles' in col.lower():
                    smiles_col = col
                    break

            if smiles_col:
                smiles_list = df[smiles_col].dropna().unique().tolist()
                print(f"Loaded {len(smiles_list)} unique SMILES from {path}")
                return smiles_list

    raise FileNotFoundError("Could not find BBBP dataset")


def analyze_stereochemistry_in_bbbp():
    """Analyze E-Z isomers and chiral centers in BBBP dataset"""
    smiles_list = load_bbbp_smiles()
    stereo = StereochemistryEncoder()

    stats = {
        'total': len(smiles_list),
        'has_double_bonds': 0,
        'has_ez_centers': 0,
        'has_chiral_centers': 0,
        'total_ez_centers': 0,
        'total_e': 0,
        'total_z': 0,
        'total_chiral': 0,
        'total_r': 0,
        'total_s': 0
    }

    print(f"\nAnalyzing stereochemistry in {len(smiles_list)} BBBP molecules...")

    for smiles in smiles_list:
        features = stereo.get_ez_isomer_features(smiles)

        if features['has_double_bonds']:
            stats['has_double_bonds'] += 1
        if features['num_ez_centers'] > 0:
            stats['has_ez_centers'] += 1
            stats['total_ez_centers'] += features['num_ez_centers']
            stats['total_e'] += features['e_count']
            stats['total_z'] += features['z_count']
        if features['num_chiral_centers'] > 0:
            stats['has_chiral_centers'] += 1
            stats['total_chiral'] += features['num_chiral_centers']
            stats['total_r'] += features['r_count']
            stats['total_s'] += features['s_count']

    print("\n" + "=" * 60)
    print("BBBP STEREOCHEMISTRY ANALYSIS")
    print("=" * 60)
    print(f"Total molecules: {stats['total']}")
    print(f"\nDouble Bonds:")
    print(f"  Molecules with C=C: {stats['has_double_bonds']} ({100*stats['has_double_bonds']/stats['total']:.1f}%)")
    print(f"\nE-Z Isomers (geometric):")
    print(f"  Molecules with E-Z centers: {stats['has_ez_centers']} ({100*stats['has_ez_centers']/stats['total']:.1f}%)")
    print(f"  Total E-Z stereocenters: {stats['total_ez_centers']}")
    print(f"    E (trans) configurations: {stats['total_e']}")
    print(f"    Z (cis) configurations: {stats['total_z']}")
    print(f"\nChiral Centers (R/S):")
    print(f"  Molecules with chiral centers: {stats['has_chiral_centers']} ({100*stats['has_chiral_centers']/stats['total']:.1f}%)")
    print(f"  Total chiral centers: {stats['total_chiral']}")
    print(f"    R configurations: {stats['total_r']}")
    print(f"    S configurations: {stats['total_s']}")
    print("=" * 60)

    return stats


def build_pubchemqc_lookup(subset: str = "b3lyp_pm6_chon500nosalt", max_scan: int = 1000000):
    """
    Build lookup table for BBBP molecules from PubChemQC.

    Args:
        subset: PubChemQC subset to use
        max_scan: Maximum number of entries to scan (for testing)
    """
    # Load BBBP SMILES
    smiles_list = load_bbbp_smiles()

    # Initialize PubChemQC integration
    pubchemqc = PubChemQCIntegration()

    print(f"\n{'='*60}")
    print("BUILDING PUBCHEMQC LOOKUP")
    print(f"{'='*60}")
    print(f"BBBP molecules to find: {len(smiles_list)}")
    print(f"PubChemQC subset: {subset}")
    print(f"Max entries to scan: {max_scan:,}")

    # Initialize dataset
    pubchemqc.initialize_dataset(subset)

    # Build lookup (this can take a while)
    print("\nStarting lookup... (press Ctrl+C to stop early)")
    found = pubchemqc.build_lookup_index(smiles_list)

    print(f"\n{'='*60}")
    print(f"LOOKUP COMPLETE")
    print(f"{'='*60}")
    print(f"Found {found}/{len(smiles_list)} molecules ({100*found/len(smiles_list):.1f}%)")
    print(f"Cache saved to: {pubchemqc.cache_file}")

    return pubchemqc


def test_lookup():
    """Test the cached lookup with some molecules"""
    pubchemqc = PubChemQCIntegration()

    test_smiles = [
        "CCO",  # Ethanol
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
    ]

    print("\nTesting cached lookups:")
    for smiles in test_smiles:
        result = pubchemqc.get_quantum_descriptors(smiles)
        if result:
            print(f"\n{smiles}:")
            print(f"  HOMO: {result.get('homo_ev', 'N/A'):.2f} eV")
            print(f"  LUMO: {result.get('lumo_ev', 'N/A'):.2f} eV")
            print(f"  Gap: {result.get('gap_ev', 'N/A'):.2f} eV")
            print(f"  χ (electronegativity): {result.get('electronegativity', 'N/A'):.2f} eV")
            print(f"  η (hardness): {result.get('chemical_hardness', 'N/A'):.2f} eV")
            print(f"  Source: {result.get('source', 'unknown')}")
        else:
            print(f"\n{smiles}: Not found in cache")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build PubChemQC lookup for BBBP")
    parser.add_argument('--action', choices=['analyze', 'build', 'test'], default='analyze',
                       help='Action to perform')
    parser.add_argument('--subset', default='b3lyp_pm6_chon500nosalt',
                       help='PubChemQC subset to use')
    parser.add_argument('--max-scan', type=int, default=1000000,
                       help='Maximum entries to scan')

    args = parser.parse_args()

    if args.action == 'analyze':
        analyze_stereochemistry_in_bbbp()
    elif args.action == 'build':
        build_pubchemqc_lookup(args.subset, args.max_scan)
    elif args.action == 'test':
        test_lookup()

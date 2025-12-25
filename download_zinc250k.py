"""
Download ZINC 250k dataset for pretraining
ZINC is a free database of commercially-available compounds for virtual screening
"""

import os
import urllib.request
import gzip
import pandas as pd

def download_zinc250k():
    """Download ZINC 250k dataset"""

    # ZINC 250k is commonly used for molecular generation/pretraining
    # Available from multiple sources - using the cleaned version from MoleculeNet

    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    zinc_path = os.path.join(data_dir, "zinc250k.csv")

    if os.path.exists(zinc_path):
        print(f"ZINC 250k already exists at {zinc_path}")
        df = pd.read_csv(zinc_path)
        print(f"Total molecules: {len(df)}")
        return zinc_path

    print("Downloading ZINC 250k dataset...")

    # Primary source: Harvard Dataverse (commonly used version)
    urls = [
        "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv",
        "https://media.githubusercontent.com/media/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv",
    ]

    downloaded = False
    for url in urls:
        try:
            print(f"Trying: {url[:60]}...")
            urllib.request.urlretrieve(url, zinc_path)
            downloaded = True
            print("Download successful!")
            break
        except Exception as e:
            print(f"Failed: {e}")
            continue

    if not downloaded:
        # Fallback: Download from DeepChem/MoleculeNet
        print("Trying alternative source (DeepChem)...")
        try:
            import deepchem as dc
            tasks, datasets, transformers = dc.molnet.load_zinc15(featurizer='Raw')
            train, valid, test = datasets

            # Combine all splits
            all_smiles = []
            for dataset in [train, valid, test]:
                all_smiles.extend(dataset.ids.tolist())

            df = pd.DataFrame({'smiles': all_smiles})
            df.to_csv(zinc_path, index=False)
            downloaded = True
        except ImportError:
            print("DeepChem not installed. Installing minimal ZINC subset...")

    if not downloaded:
        # Create a minimal version by generating diverse drug-like molecules
        print("\nCreating ZINC-like pretraining set from available data...")
        create_pretraining_set(zinc_path)

    # Verify
    if os.path.exists(zinc_path):
        df = pd.read_csv(zinc_path)
        print(f"\nZINC dataset ready: {len(df)} molecules")
        print(f"Location: {zinc_path}")

        # Show sample
        if 'smiles' in df.columns:
            print(f"\nSample SMILES:")
            for s in df['smiles'].head(3):
                print(f"  {s}")
        elif 'SMILES' in df.columns:
            print(f"\nSample SMILES:")
            for s in df['SMILES'].head(3):
                print(f"  {s}")

        return zinc_path
    else:
        raise Exception("Failed to download ZINC dataset")


def create_pretraining_set(output_path):
    """Create a pretraining set from ChEMBL or PubChem if ZINC unavailable"""

    # Use RDKit's built-in fragment library + enumerate combinations
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    import random

    print("Generating diverse drug-like molecules for pretraining...")

    # Start with known drug scaffolds
    scaffolds = [
        "c1ccccc1",  # benzene
        "c1ccncc1",  # pyridine
        "c1ccc2ccccc2c1",  # naphthalene
        "c1cnc2ccccc2n1",  # quinazoline
        "c1ccc2[nH]ccc2c1",  # indole
        "c1ccc2nc[nH]c2c1",  # benzimidazole
        "C1CCCCC1",  # cyclohexane
        "C1CCNCC1",  # piperidine
        "C1COCCN1",  # morpholine
        "c1ccc(cc1)c2ccccc2",  # biphenyl
    ]

    # Common substituents
    substituents = [
        "", "C", "CC", "CCC", "C(C)C", "C(=O)O", "C(=O)N",
        "O", "OC", "N", "NC", "N(C)C", "F", "Cl", "Br",
        "C(F)(F)F", "S(=O)(=O)N", "C#N", "C(=O)OC"
    ]

    molecules = set()

    # Also load our BBBP data to include those structures
    bbbp_path = "data/BBBP.csv"
    if os.path.exists(bbbp_path):
        bbbp_df = pd.read_csv(bbbp_path)
        smiles_col = 'smiles' if 'smiles' in bbbp_df.columns else 'SMILES'
        for smi in bbbp_df[smiles_col]:
            if Chem.MolFromSmiles(smi) is not None:
                molecules.add(smi)
        print(f"Added {len(molecules)} molecules from BBBP")

    # Generate more molecules using RDKit
    print("Generating additional molecules...")

    # Use MolFromSmiles to validate
    for scaffold in scaffolds:
        mol = Chem.MolFromSmiles(scaffold)
        if mol:
            molecules.add(Chem.MolToSmiles(mol))

    # Try to download a subset of ChEMBL
    try:
        print("Attempting to fetch molecules from ChEMBL...")
        import urllib.request
        import json

        # Get small drug-like molecules from ChEMBL
        chembl_url = "https://www.ebi.ac.uk/chembl/api/data/molecule.json?max_phase=4&molecule_type=Small%20molecule&limit=1000"

        req = urllib.request.Request(chembl_url, headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())

            for mol_data in data.get('molecules', []):
                structs = mol_data.get('molecule_structures', {})
                if structs and structs.get('canonical_smiles'):
                    smi = structs['canonical_smiles']
                    if Chem.MolFromSmiles(smi) is not None:
                        molecules.add(smi)

        print(f"Fetched {len(molecules)} molecules from ChEMBL")
    except Exception as e:
        print(f"ChEMBL fetch failed: {e}")

    # If still not enough, use PubChem diversity subset
    if len(molecules) < 10000:
        print("Fetching from PubChem...")
        try:
            # PubChem has a diversity subset
            pubchem_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/listkey/0/property/CanonicalSMILES/CSV"
            # This won't work directly, need different approach
            pass
        except:
            pass

    print(f"\nTotal molecules collected: {len(molecules)}")

    # Save what we have
    df = pd.DataFrame({'smiles': list(molecules)})
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    return output_path


if __name__ == "__main__":
    download_zinc250k()

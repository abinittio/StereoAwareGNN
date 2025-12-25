"""
Download and prepare the BBBP dataset from MoleculeNet
"""

import pandas as pd
import os

def download_bbbp_dataset():
    """
    Download the BBBP (Blood-Brain Barrier Penetration) dataset
    from MoleculeNet (2039 compounds)
    """
    print("Downloading BBBP dataset from MoleculeNet...")

    # MoleculeNet BBBP dataset URL
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"

    try:
        # Download dataset
        df = pd.read_csv(url)
        print(f"Downloaded {len(df)} compounds")

        # Inspect the dataset
        print("\nDataset columns:", df.columns.tolist())
        print("\nFirst few rows:")
        print(df.head())

        # The BBBP dataset typically has columns: 'smiles', 'p_np' (binary classification)
        # We need to convert it to our format with continuous BBB permeability scores

        if 'smiles' in df.columns and 'p_np' in df.columns:
            # Rename columns to match our format
            df_processed = pd.DataFrame({
                'SMILES': df['smiles'],
                'BBB_permeability': df['p_np'].astype(float),  # 1 = permeable, 0 = not permeable
                'compound_name': df['name'] if 'name' in df.columns else ['Unknown'] * len(df)
            })

            # Save processed dataset
            os.makedirs('data', exist_ok=True)
            output_path = 'data/bbbp_dataset.csv'
            df_processed.to_csv(output_path, index=False)
            print(f"\nProcessed dataset saved to {output_path}")
            print(f"Total compounds: {len(df_processed)}")
            print(f"BBB+ (permeable): {(df_processed['BBB_permeability'] == 1).sum()}")
            print(f"BBB- (not permeable): {(df_processed['BBB_permeability'] == 0).sum()}")

            return df_processed
        else:
            print("ERROR: Dataset format not as expected")
            print(f"Available columns: {df.columns.tolist()}")
            return None

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nTrying alternative source...")

        # Alternative: Use DeepChem library
        try:
            import deepchem as dc
            tasks, datasets, transformers = dc.molnet.load_bbbp(featurizer='Raw')
            train_dataset, valid_dataset, test_dataset = datasets

            # Combine all splits
            all_smiles = []
            all_labels = []

            for dataset in [train_dataset, valid_dataset, test_dataset]:
                all_smiles.extend(dataset.ids)
                all_labels.extend(dataset.y.flatten())

            df_processed = pd.DataFrame({
                'SMILES': all_smiles,
                'BBB_permeability': all_labels,
                'compound_name': ['Unknown'] * len(all_smiles)
            })

            # Save
            os.makedirs('data', exist_ok=True)
            output_path = 'data/bbbp_dataset.csv'
            df_processed.to_csv(output_path, index=False)
            print(f"\nDataset saved to {output_path}")
            print(f"Total compounds: {len(df_processed)}")

            return df_processed

        except ImportError:
            print("DeepChem not installed. Install with: pip install deepchem")
            return None
        except Exception as e2:
            print(f"Error with alternative method: {e2}")
            return None

if __name__ == "__main__":
    dataset = download_bbbp_dataset()

    if dataset is not None:
        print("\n" + "="*50)
        print("SUCCESS: BBBP dataset downloaded and ready!")
        print("="*50)
        print("\nNext steps:")
        print("1. Review the dataset: data/bbbp_dataset.csv")
        print("2. Train the advanced model: python train_advanced.py")
        print("3. Update app.py to use the new model")
    else:
        print("\n" + "="*50)
        print("FAILED: Could not download dataset")
        print("="*50)
        print("\nManual download:")
        print("1. Visit: https://moleculenet.org/datasets-1")
        print("2. Download BBBP.csv")
        print("3. Place in data/bbbp_dataset.csv")

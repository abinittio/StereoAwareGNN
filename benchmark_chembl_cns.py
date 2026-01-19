"""
ChEMBL CNS Benchmark for BBB Permeability Model
================================================

Uses ChEMBL bioactivity data for CNS vs peripheral targets.
For CNS targets, active compounds should be BBB+.

This is a more reliable alternative to DUD-E (which is currently down).
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import pickle

sys.path.insert(0, str(Path(__file__).parent))

# ChEMBL target IDs for CNS vs peripheral targets
CNS_TARGETS = {
    'CHEMBL217': 'Dopamine D2 receptor',
    'CHEMBL224': 'Serotonin 1A (5-HT1A) receptor',
    'CHEMBL225': 'Serotonin 2A (5-HT2A) receptor',
    'CHEMBL228': 'GABA-A receptor',
    'CHEMBL233': 'Mu opioid receptor',
    'CHEMBL236': 'Kappa opioid receptor',
    'CHEMBL251': 'Adenosine A2A receptor',
    'CHEMBL1867': 'Acetylcholinesterase',
}

PERIPHERAL_TARGETS = {
    'CHEMBL203': 'Epidermal growth factor receptor',
    'CHEMBL279': 'Vascular endothelial growth factor receptor 2',
    'CHEMBL4036': 'Hepatitis C virus NS5B',
    'CHEMBL244': 'Coagulation factor X',
}


def fetch_chembl_actives(target_id, max_compounds=500, cache_dir=None):
    """Fetch active compounds for a target from ChEMBL."""
    from chembl_webresource_client.new_client import new_client

    if cache_dir:
        cache_file = cache_dir / f'{target_id}_actives.pkl'
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

    activity = new_client.activity

    # Get high-affinity binders (Ki < 100 nM)
    results = activity.filter(
        target_chembl_id=target_id,
        standard_type__in=['Ki', 'IC50', 'Kd'],
        standard_units='nM',
        standard_value__lte=100,
        standard_relation='='
    ).only(['molecule_chembl_id', 'canonical_smiles', 'standard_value'])

    smiles_list = []
    seen = set()

    for r in results:
        smi = r.get('canonical_smiles')
        if smi and smi not in seen:
            smiles_list.append(smi)
            seen.add(smi)
            if len(smiles_list) >= max_compounds:
                break

    if cache_dir:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(smiles_list, f)

    return smiles_list


def load_bbb_model():
    """Load the BBB model ensemble using BBBStereoV2Model from bbb_stereo_v2.py."""
    from zinc_stereo_pretraining import StereoAwareEncoder
    from bbb_stereo_v2 import BBBStereoV2Model

    models = []
    for fold in range(1, 6):
        model_path = Path(__file__).parent / f'models/bbb_stereo_v2_fold{fold}_best.pth'
        if model_path.exists():
            try:
                encoder = StereoAwareEncoder(node_features=21, hidden_dim=128, num_layers=4)
                model = BBBStereoV2Model(encoder, hidden_dim=128)
                state = torch.load(model_path, map_location='cpu')
                model.load_state_dict(state, strict=True)
                model.eval()
                models.append(model)
                print(f"  Loaded fold {fold}")
            except Exception as e:
                print(f"  Failed fold {fold}: {e}")
    return models


def predict_bbb_batch(smiles_list, models):
    """Predict BBB permeability for a batch of SMILES."""
    from mol_to_graph_enhanced import mol_to_graph_enhanced
    from torch_geometric.loader import DataLoader

    graphs = []
    valid_indices = []

    for i, smi in enumerate(smiles_list):
        try:
            graph = mol_to_graph_enhanced(smi, y=0.0, include_quantum=False, include_stereo=True, use_dft=False)
            if graph is not None and hasattr(graph, 'x') and graph.x.shape[1] == 21:
                graphs.append(graph)
                valid_indices.append(i)
        except:
            pass

    if not graphs:
        return np.array([]), valid_indices

    loader = DataLoader(graphs, batch_size=64)
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            batch_preds = []
            for model in models:
                # BBBStereoV2Model returns (logBB, classification_prob)
                logBB, prob = model(batch.x, batch.edge_index, batch.batch)
                probs = torch.sigmoid(prob).cpu().numpy().flatten()
                batch_preds.append(probs)
            all_preds.extend(np.mean(batch_preds, axis=0))

    return np.array(all_preds), valid_indices


def main():
    print("=" * 70)
    print("ChEMBL CNS BENCHMARK FOR BBB PERMEABILITY MODEL")
    print("=" * 70)
    print("\nHypothesis: Actives for CNS targets should be BBB+")
    print()

    cache_dir = Path(__file__).parent / 'data' / 'chembl_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Loading BBB model...")
    models = load_bbb_model()
    if not models:
        print("ERROR: Could not load BBB model!")
        return
    print(f"  Loaded {len(models)} model(s)")

    results = []

    # CNS targets
    print("\n" + "=" * 60)
    print("CNS TARGETS (actives expected to be BBB+)")
    print("=" * 60)

    for target_id, description in CNS_TARGETS.items():
        print(f"\n{target_id} - {description}")

        try:
            actives = fetch_chembl_actives(target_id, max_compounds=300, cache_dir=cache_dir)
            if not actives:
                print("  No actives found")
                continue

            print(f"  Found {len(actives)} actives")
            preds, valid_idx = predict_bbb_batch(actives, models)

            if len(preds) == 0:
                print("  No valid predictions")
                continue

            bbb_plus_rate = (preds > 0.5).mean()
            mean_prob = preds.mean()

            print(f"  BBB+ rate: {bbb_plus_rate:.1%}")
            print(f"  Mean probability: {mean_prob:.3f}")

            results.append({
                'target': target_id,
                'type': 'CNS',
                'description': description,
                'n_compounds': len(preds),
                'bbb_plus_rate': bbb_plus_rate,
                'mean_prob': mean_prob,
            })

        except Exception as e:
            print(f"  Error: {e}")

    # Peripheral targets
    print("\n" + "=" * 60)
    print("PERIPHERAL TARGETS (lower BBB+ rate expected)")
    print("=" * 60)

    for target_id, description in PERIPHERAL_TARGETS.items():
        print(f"\n{target_id} - {description}")

        try:
            actives = fetch_chembl_actives(target_id, max_compounds=300, cache_dir=cache_dir)
            if not actives:
                print("  No actives found")
                continue

            print(f"  Found {len(actives)} actives")
            preds, valid_idx = predict_bbb_batch(actives, models)

            if len(preds) == 0:
                print("  No valid predictions")
                continue

            bbb_plus_rate = (preds > 0.5).mean()
            mean_prob = preds.mean()

            print(f"  BBB+ rate: {bbb_plus_rate:.1%}")
            print(f"  Mean probability: {mean_prob:.3f}")

            results.append({
                'target': target_id,
                'type': 'Peripheral',
                'description': description,
                'n_compounds': len(preds),
                'bbb_plus_rate': bbb_plus_rate,
                'mean_prob': mean_prob,
            })

        except Exception as e:
            print(f"  Error: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    df = pd.DataFrame(results)

    if len(df) > 0:
        cns_df = df[df['type'] == 'CNS']
        periph_df = df[df['type'] == 'Peripheral']

        print(f"\nCNS Targets ({len(cns_df)} tested):")
        if len(cns_df) > 0:
            print(f"  Mean BBB+ rate:    {cns_df['bbb_plus_rate'].mean():.1%}")
            print(f"  Mean probability:  {cns_df['mean_prob'].mean():.3f}")
            print(f"  Total compounds:   {cns_df['n_compounds'].sum()}")

        print(f"\nPeripheral Targets ({len(periph_df)} tested):")
        if len(periph_df) > 0:
            print(f"  Mean BBB+ rate:    {periph_df['bbb_plus_rate'].mean():.1%}")
            print(f"  Mean probability:  {periph_df['mean_prob'].mean():.3f}")
            print(f"  Total compounds:   {periph_df['n_compounds'].sum()}")

        # Statistical test
        if len(cns_df) > 2 and len(periph_df) > 2:
            from scipy import stats
            t_stat, p_val = stats.ttest_ind(
                cns_df['bbb_plus_rate'].values,
                periph_df['bbb_plus_rate'].values
            )
            print(f"\nCNS vs Peripheral BBB+ rate comparison:")
            print(f"  t-statistic: {t_stat:.2f}")
            print(f"  p-value:     {p_val:.4f}")
            if p_val < 0.05 and t_stat > 0:
                print("  SIGNIFICANT: CNS target actives are more BBB+ than peripheral")
            elif p_val < 0.05 and t_stat < 0:
                print("  UNEXPECTED: Peripheral higher than CNS")
            else:
                print("  Not significant")

        # Save results
        output_path = Path(__file__).parent / 'chembl_bbb_results.csv'
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
Expected results if BBB model is working correctly:
- CNS target actives: >70% BBB+ (drugs need to cross BBB)
- Peripheral target actives: <50% BBB+ (no CNS requirement)
- Significant difference between CNS and peripheral

This validates that the model correctly predicts CNS drug-likeness.
""")


if __name__ == "__main__":
    main()

"""
DUD-E Benchmark for BBB Permeability Model
===========================================

DUD-E (Database of Useful Decoys: Enhanced) contains actives and decoys
for ~100 protein targets. For BBB validation, we use CNS-relevant targets
where actives SHOULD be BBB-permeable.

Hypothesis: For CNS targets (dopamine receptors, serotonin receptors, etc.),
active compounds should have higher BBB+ rates than decoys.

This serves as an indirect validation of BBB prediction quality.
"""

import os
import sys
import gzip
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from io import StringIO
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

# CNS-relevant targets in DUD-E (using actual DUD-E target names)
CNS_TARGETS = {
    'drd3': 'Dopamine D3 receptor',
    'aa2ar': 'Adenosine A2A receptor',
    'adrb1': 'Beta-1 adrenergic receptor',
    'adrb2': 'Beta-2 adrenergic receptor',
    'aces': 'Acetylcholinesterase',
    'gria2': 'Glutamate receptor 2',
    'hdac2': 'Histone deacetylase 2',
    'lck': 'Lymphocyte kinase',
}

# Non-CNS targets (actives should NOT necessarily be BBB+)
PERIPHERAL_TARGETS = {
    'cxcr4': 'CXCR4 chemokine receptor',
    'hivpr': 'HIV protease',
    'hivrt': 'HIV reverse transcriptase',
    'kpcb': 'Protein kinase C beta',
}

DUDE_BASE_URL = "https://dude.docking.org/targets"
DUDE_ZENODO_URL = "https://zenodo.org/records/7954048/files"  # Alternative mirror


def download_dude_target(target_name, data_dir):
    """Download actives and decoys for a DUD-E target."""
    target_dir = data_dir / target_name
    target_dir.mkdir(parents=True, exist_ok=True)

    actives_file = target_dir / 'actives_final.smi'
    decoys_file = target_dir / 'decoys_final.smi'

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    # Download actives
    if not actives_file.exists():
        # Try main URL first, then alternative
        urls_to_try = [
            f"{DUDE_BASE_URL}/{target_name}/actives_final.ism",
            f"http://dude.docking.org/targets/{target_name}/actives_final.ism",
        ]

        downloaded = False
        for url in urls_to_try:
            try:
                r = requests.get(url, timeout=30, headers=headers)
                if r.status_code == 200 and len(r.text) > 10:
                    with open(actives_file, 'w') as f:
                        f.write(r.text)
                    print(f"  Downloaded actives for {target_name}")
                    downloaded = True
                    break
            except Exception as e:
                continue

        if not downloaded:
            print(f"  Failed to download actives for {target_name}")
            return None, None

    # Download decoys
    if not decoys_file.exists():
        urls_to_try = [
            f"{DUDE_BASE_URL}/{target_name}/decoys_final.ism",
            f"http://dude.docking.org/targets/{target_name}/decoys_final.ism",
        ]

        downloaded = False
        for url in urls_to_try:
            try:
                r = requests.get(url, timeout=60, headers=headers)
                if r.status_code == 200 and len(r.text) > 10:
                    with open(decoys_file, 'w') as f:
                        f.write(r.text)
                    print(f"  Downloaded decoys for {target_name}")
                    downloaded = True
                    break
            except Exception as e:
                continue

        if not downloaded:
            print(f"  Failed to download decoys for {target_name}")
            return None, None

    # Parse files
    actives = []
    with open(actives_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                actives.append(parts[0])

    decoys = []
    with open(decoys_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                decoys.append(parts[0])

    return actives, decoys


def load_bbb_model():
    """Load the best BBB model using the correct architecture from app.py."""
    from torch_geometric.nn import GATv2Conv, TransformerConv, global_mean_pool, global_max_pool
    from mol_to_graph_enhanced import mol_to_graph_enhanced

    class StereoAwareEncoder(nn.Module):
        """Stereo-aware molecular encoder using GATv2 + Transformer (from app.py)."""

        def __init__(self, node_features=21, hidden_dim=128, num_layers=4, heads=4, dropout=0.1):
            super().__init__()
            self.node_features = node_features
            self.hidden_dim = hidden_dim

            # Input projection
            self.input_proj = nn.Sequential(
                nn.Linear(node_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

            # GATv2 layers
            self.gat_layers = nn.ModuleList()
            self.gat_norms = nn.ModuleList()

            for i in range(num_layers):
                in_channels = hidden_dim
                out_channels = hidden_dim // heads
                self.gat_layers.append(
                    GATv2Conv(in_channels, out_channels, heads=heads, dropout=dropout, add_self_loops=True)
                )
                self.gat_norms.append(nn.LayerNorm(hidden_dim))

            # Transformer layer
            self.transformer = TransformerConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
            self.transformer_norm = nn.LayerNorm(hidden_dim)

        def forward(self, x, edge_index, batch, edge_attr=None):
            # Initial projection
            x = self.input_proj(x)

            # GAT layers with residual
            for gat, norm in zip(self.gat_layers, self.gat_norms):
                x_res = x
                x = gat(x, edge_index)
                x = norm(x + x_res)

            # Transformer layer
            x_res = x
            x = self.transformer(x, edge_index)
            x = self.transformer_norm(x + x_res)

            # Pool
            mean_pool = global_mean_pool(x, batch)
            max_pool = global_max_pool(x, batch)
            return torch.cat([mean_pool, max_pool], dim=-1)

    class BBBStereoClassifier(nn.Module):
        def __init__(self, encoder, hidden_dim=128):
            super().__init__()
            self.encoder = encoder
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, 1)
            )

        def forward(self, x, edge_index, batch):
            graph_embed = self.encoder(x, edge_index, batch)
            return self.classifier(graph_embed)

    # Try to load ensemble of fold models
    models = []
    for fold in range(1, 6):
        model_path = Path(__file__).parent / f'models/bbb_stereo_fold{fold}_best.pth'
        if model_path.exists():
            try:
                encoder = StereoAwareEncoder(node_features=21, hidden_dim=128, num_layers=4)
                model = BBBStereoClassifier(encoder, hidden_dim=128)
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()
                models.append(model)
                print(f"  Loaded fold {fold} model")
            except Exception as e:
                print(f"  Failed to load fold {fold}: {e}")

    if not models:
        # Try v2 models
        for fold in range(1, 6):
            model_path = Path(__file__).parent / f'models/bbb_stereo_v2_fold{fold}_best.pth'
            if model_path.exists():
                try:
                    encoder = StereoAwareEncoder(node_features=21, hidden_dim=128, num_layers=4)
                    model = BBBStereoClassifier(encoder, hidden_dim=128)
                    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
                    model.eval()
                    models.append(model)
                    print(f"  Loaded v2 fold {fold} model")
                except Exception as e:
                    print(f"  Failed to load v2 fold {fold}: {e}")

    return models


def predict_bbb_batch(smiles_list, models):
    """Predict BBB permeability for a batch of SMILES."""
    from mol_to_graph_enhanced import mol_to_graph_enhanced
    from torch_geometric.loader import DataLoader
    from torch_geometric.data import Batch

    # Convert to graphs
    graphs = []
    valid_indices = []

    for i, smi in enumerate(smiles_list):
        try:
            graph = mol_to_graph_enhanced(
                smi,
                y=0.0,
                include_quantum=False,
                include_stereo=True,
                use_dft=False
            )
            if graph is not None and hasattr(graph, 'x') and graph.x.shape[1] == 21:
                graphs.append(graph)
                valid_indices.append(i)
        except:
            pass

    if not graphs:
        return np.array([]), valid_indices

    # Batch predict
    loader = DataLoader(graphs, batch_size=64)
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            batch_preds = []
            for model in models:
                out = model(batch.x, batch.edge_index, batch.batch)
                probs = torch.sigmoid(out).cpu().numpy().flatten()
                batch_preds.append(probs)

            # Ensemble average
            avg_preds = np.mean(batch_preds, axis=0)
            all_preds.extend(avg_preds)

    return np.array(all_preds), valid_indices


def main():
    print("=" * 70)
    print("DUD-E BENCHMARK FOR BBB PERMEABILITY MODEL")
    print("=" * 70)
    print("\nHypothesis: For CNS targets, actives should be more BBB+ than decoys")
    print()

    data_dir = Path(__file__).parent / 'data' / 'dude'
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load BBB model
    print("Loading BBB model...")
    models = load_bbb_model()
    if not models:
        print("ERROR: Could not load BBB model!")
        return
    print(f"  Loaded {len(models)} model(s)")

    results = []

    # Test CNS targets
    print("\n" + "=" * 60)
    print("CNS TARGETS (actives expected to be BBB+)")
    print("=" * 60)

    for target, description in CNS_TARGETS.items():
        print(f"\n{target.upper()} - {description}")

        actives, decoys = download_dude_target(target, data_dir)
        if actives is None:
            continue

        print(f"  Actives: {len(actives)}, Decoys: {len(decoys)}")

        # Predict BBB for actives
        print("  Predicting BBB for actives...")
        active_preds, active_valid = predict_bbb_batch(actives, models)

        # Predict BBB for decoys (sample if too many)
        decoy_sample = decoys[:min(len(decoys), 1000)]  # Limit to 1000 decoys
        print(f"  Predicting BBB for decoys (n={len(decoy_sample)})...")
        decoy_preds, decoy_valid = predict_bbb_batch(decoy_sample, models)

        if len(active_preds) == 0 or len(decoy_preds) == 0:
            print("  Skipping - not enough valid molecules")
            continue

        # Calculate metrics
        active_bbb_rate = (active_preds > 0.5).mean()
        decoy_bbb_rate = (decoy_preds > 0.5).mean()
        active_mean = active_preds.mean()
        decoy_mean = decoy_preds.mean()

        # Enrichment: how much more likely are actives to be BBB+
        enrichment = active_bbb_rate / decoy_bbb_rate if decoy_bbb_rate > 0 else float('inf')

        # AUC treating "is_active" as label
        labels = np.concatenate([np.ones(len(active_preds)), np.zeros(len(decoy_preds))])
        preds = np.concatenate([active_preds, decoy_preds])
        try:
            auc = roc_auc_score(labels, preds)
        except:
            auc = 0.5

        print(f"  Active BBB+ rate: {active_bbb_rate:.1%} (mean prob: {active_mean:.3f})")
        print(f"  Decoy BBB+ rate:  {decoy_bbb_rate:.1%} (mean prob: {decoy_mean:.3f})")
        print(f"  Enrichment:       {enrichment:.2f}x")
        print(f"  AUC (active vs decoy): {auc:.3f}")

        results.append({
            'target': target,
            'type': 'CNS',
            'description': description,
            'n_actives': len(active_preds),
            'n_decoys': len(decoy_preds),
            'active_bbb_rate': active_bbb_rate,
            'decoy_bbb_rate': decoy_bbb_rate,
            'enrichment': enrichment,
            'auc': auc,
        })

    # Test peripheral targets for comparison
    print("\n" + "=" * 60)
    print("PERIPHERAL TARGETS (control - no BBB enrichment expected)")
    print("=" * 60)

    for target, description in PERIPHERAL_TARGETS.items():
        print(f"\n{target.upper()} - {description}")

        actives, decoys = download_dude_target(target, data_dir)
        if actives is None:
            continue

        print(f"  Actives: {len(actives)}, Decoys: {len(decoys)}")

        print("  Predicting BBB for actives...")
        active_preds, active_valid = predict_bbb_batch(actives, models)

        decoy_sample = decoys[:min(len(decoys), 1000)]
        print(f"  Predicting BBB for decoys (n={len(decoy_sample)})...")
        decoy_preds, decoy_valid = predict_bbb_batch(decoy_sample, models)

        if len(active_preds) == 0 or len(decoy_preds) == 0:
            print("  Skipping - not enough valid molecules")
            continue

        active_bbb_rate = (active_preds > 0.5).mean()
        decoy_bbb_rate = (decoy_preds > 0.5).mean()
        enrichment = active_bbb_rate / decoy_bbb_rate if decoy_bbb_rate > 0 else float('inf')

        labels = np.concatenate([np.ones(len(active_preds)), np.zeros(len(decoy_preds))])
        preds = np.concatenate([active_preds, decoy_preds])
        try:
            auc = roc_auc_score(labels, preds)
        except:
            auc = 0.5

        print(f"  Active BBB+ rate: {active_bbb_rate:.1%}")
        print(f"  Decoy BBB+ rate:  {decoy_bbb_rate:.1%}")
        print(f"  Enrichment:       {enrichment:.2f}x")

        results.append({
            'target': target,
            'type': 'Peripheral',
            'description': description,
            'n_actives': len(active_preds),
            'n_decoys': len(decoy_preds),
            'active_bbb_rate': active_bbb_rate,
            'decoy_bbb_rate': decoy_bbb_rate,
            'enrichment': enrichment,
            'auc': auc,
        })

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
            print(f"  Mean BBB+ rate (actives): {cns_df['active_bbb_rate'].mean():.1%}")
            print(f"  Mean BBB+ rate (decoys):  {cns_df['decoy_bbb_rate'].mean():.1%}")
            print(f"  Mean enrichment:          {cns_df['enrichment'].mean():.2f}x")
            print(f"  Mean AUC:                 {cns_df['auc'].mean():.3f}")

        print(f"\nPeripheral Targets ({len(periph_df)} tested):")
        if len(periph_df) > 0:
            print(f"  Mean BBB+ rate (actives): {periph_df['active_bbb_rate'].mean():.1%}")
            print(f"  Mean BBB+ rate (decoys):  {periph_df['decoy_bbb_rate'].mean():.1%}")
            print(f"  Mean enrichment:          {periph_df['enrichment'].mean():.2f}x")

        # Statistical test
        if len(cns_df) > 0 and len(periph_df) > 0:
            from scipy import stats
            t_stat, p_val = stats.ttest_ind(
                cns_df['enrichment'].values,
                periph_df['enrichment'].values
            )
            print(f"\nCNS vs Peripheral enrichment difference:")
            print(f"  t-statistic: {t_stat:.2f}")
            print(f"  p-value:     {p_val:.4f}")
            if p_val < 0.05:
                print("  SIGNIFICANT: CNS target actives are more BBB+ than peripheral")

        # Save results
        output_path = Path(__file__).parent / 'dude_bbb_results.csv'
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
If the model is working correctly:
- CNS target actives should have HIGH BBB+ rates (>70%)
- CNS enrichment should be >1.5x (actives more BBB+ than decoys)
- Peripheral targets should show lower/no enrichment

This validates that the BBB model correctly identifies CNS-penetrant drugs.
""")


if __name__ == "__main__":
    main()

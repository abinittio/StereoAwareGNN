"""
ChEMBL BBB & Kinetics Data Curation Script
==========================================
Fetches and processes BBB permeability, transporter kinetics, and
membrane permeability data from ChEMBL for GNN training.

Target assay types:
- BBB permeability (in vivo, in vitro)
- P-gp efflux ratios
- BCRP transport
- Caco-2 permeability (Papp)
- PAMPA permeability
- Brain-to-plasma ratios (Kp, Kp,uu)
- LogBB values

Author: BBB_System
Date: December 2025
"""

import os
import json
import time
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import urllib.request
import urllib.parse

# Try to import chembl_webresource_client (preferred)
try:
    from chembl_webresource_client.new_client import new_client
    CHEMBL_CLIENT_AVAILABLE = True
except ImportError:
    CHEMBL_CLIENT_AVAILABLE = False
    print("chembl_webresource_client not available, using REST API")


@dataclass
class BBBDataPoint:
    """Single BBB/permeability data point"""
    chembl_id: str
    smiles: str
    assay_type: str  # BBB, P-gp, Caco2, PAMPA, etc.
    standard_type: str  # Papp, ER, LogBB, Kp, etc.
    standard_value: float
    standard_units: str
    standard_relation: str  # =, <, >, etc.
    assay_description: str
    target_name: Optional[str]
    pchembl_value: Optional[float]
    document_chembl_id: str
    year: Optional[int]


class ChEMBLDataCurator:
    """
    Curates BBB and kinetics data from ChEMBL database.
    """

    BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"

    # Assay keywords for BBB-related data
    BBB_KEYWORDS = [
        "blood-brain barrier",
        "blood brain barrier",
        "BBB",
        "brain penetration",
        "brain permeability",
        "CNS penetration",
        "brain-to-plasma",
        "brain to plasma",
        "brain/plasma",
        "LogBB",
        "log BB",
        "Kp,uu",
        "Kpuu",
    ]

    # P-glycoprotein/efflux transporter keywords
    PGP_KEYWORDS = [
        "P-glycoprotein",
        "P-gp",
        "Pgp",
        "MDR1",
        "ABCB1",
        "efflux ratio",
        "efflux transport",
    ]

    # BCRP transporter keywords
    BCRP_KEYWORDS = [
        "BCRP",
        "ABCG2",
        "breast cancer resistance protein",
    ]

    # Caco-2 permeability keywords
    CACO2_KEYWORDS = [
        "Caco-2",
        "Caco2",
        "intestinal permeability",
    ]

    # PAMPA keywords
    PAMPA_KEYWORDS = [
        "PAMPA",
        "parallel artificial membrane",
        "artificial membrane permeability",
    ]

    # Standard types we're interested in
    RELEVANT_STANDARD_TYPES = {
        # Permeability
        "Papp", "Pe", "Permeability", "LogPapp",
        # Efflux
        "ER", "Efflux ratio", "Efflux Ratio",
        # Brain penetration
        "LogBB", "log BB", "Kp", "Kp,uu", "Kpuu",
        "Brain/Plasma", "B/P ratio",
        # IC50/Ki for transporters
        "IC50", "Ki", "Km",
        # Binary
        "Inhibition", "Activity",
    }

    def __init__(self, cache_dir: str = "data/chembl_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_points: List[BBBDataPoint] = []
        self.session_stats = {
            "api_calls": 0,
            "cached_hits": 0,
            "total_records": 0,
        }

    def _api_request(self, endpoint: str, params: Dict, max_retries: int = 3) -> Dict:
        """Make API request with caching and retry logic"""
        # Create cache key
        cache_key = f"{endpoint}_{hash(frozenset(params.items()))}"
        cache_file = self.cache_dir / f"{cache_key}.json"

        # Check cache
        if cache_file.exists():
            self.session_stats["cached_hits"] += 1
            with open(cache_file, 'r') as f:
                return json.load(f)

        # Build URL
        param_str = urllib.parse.urlencode(params)
        url = f"{self.BASE_URL}/{endpoint}.json?{param_str}"

        # Make request with retries
        for attempt in range(max_retries):
            self.session_stats["api_calls"] += 1
            req = urllib.request.Request(url, headers={
                'Accept': 'application/json',
                'User-Agent': 'BBB_System/1.0'
            })

            try:
                with urllib.request.urlopen(req, timeout=120) as response:
                    data = json.loads(response.read().decode())

                # Cache result
                with open(cache_file, 'w') as f:
                    json.dump(data, f)

                time.sleep(0.3)  # Rate limiting
                return data

            except Exception as e:
                wait_time = (attempt + 1) * 5
                print(f"  Attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"  Retrying in {wait_time}s...")
                    time.sleep(wait_time)

        print(f"All retries failed for {endpoint}")
        return {"activities": [], "assays": []}

    def _search_assays_by_keywords(self, keywords: List[str],
                                    assay_type: str) -> List[str]:
        """Search for assay ChEMBL IDs matching keywords"""
        assay_ids = set()

        for keyword in keywords:
            print(f"  Searching for '{keyword}'...")

            if CHEMBL_CLIENT_AVAILABLE:
                assay = new_client.assay
                results = assay.filter(
                    description__icontains=keyword
                ).only(['assay_chembl_id', 'description', 'assay_type'])

                for r in results:
                    assay_ids.add(r['assay_chembl_id'])
            else:
                # REST API fallback
                params = {
                    'description__icontains': keyword,
                    'limit': 1000,
                }
                data = self._api_request('assay', params)

                for assay in data.get('assays', []):
                    assay_ids.add(assay.get('assay_chembl_id'))

        print(f"  Found {len(assay_ids)} assays for {assay_type}")
        return list(assay_ids)

    def _fetch_activities_for_assays(self, assay_ids: List[str],
                                      assay_type: str) -> List[BBBDataPoint]:
        """Fetch activity data for given assay IDs with checkpointing"""
        data_points = []

        # Checkpoint file for this assay type
        checkpoint_file = self.cache_dir / f"checkpoint_{assay_type}.pkl"
        start_idx = 0

        # Load checkpoint if exists
        if checkpoint_file.exists():
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
                data_points = checkpoint.get('data_points', [])
                start_idx = checkpoint.get('last_idx', 0) + 1
                print(f"  Resuming from checkpoint: {start_idx}/{len(assay_ids)}")

        for i, assay_id in enumerate(assay_ids[start_idx:], start=start_idx):
            if i % 50 == 0:
                print(f"  Processing assay {i+1}/{len(assay_ids)}...")
                # Save checkpoint every 50 assays
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump({'data_points': data_points, 'last_idx': i}, f)

            if CHEMBL_CLIENT_AVAILABLE:
                activity = new_client.activity
                results = activity.filter(
                    assay_chembl_id=assay_id
                ).only([
                    'molecule_chembl_id', 'canonical_smiles',
                    'standard_type', 'standard_value', 'standard_units',
                    'standard_relation', 'assay_description',
                    'target_pref_name', 'pchembl_value',
                    'document_chembl_id', 'document_year'
                ])

                for r in results:
                    if r.get('canonical_smiles') and r.get('standard_value'):
                        try:
                            dp = BBBDataPoint(
                                chembl_id=r.get('molecule_chembl_id', ''),
                                smiles=r['canonical_smiles'],
                                assay_type=assay_type,
                                standard_type=r.get('standard_type', ''),
                                standard_value=float(r['standard_value']),
                                standard_units=r.get('standard_units', ''),
                                standard_relation=r.get('standard_relation', '='),
                                assay_description=r.get('assay_description', '')[:500],
                                target_name=r.get('target_pref_name'),
                                pchembl_value=float(r['pchembl_value']) if r.get('pchembl_value') else None,
                                document_chembl_id=r.get('document_chembl_id', ''),
                                year=r.get('document_year'),
                            )
                            data_points.append(dp)
                        except (ValueError, TypeError):
                            continue
            else:
                # REST API
                params = {
                    'assay_chembl_id': assay_id,
                    'limit': 1000,
                }
                data = self._api_request('activity', params)

                for act in data.get('activities', []):
                    if act.get('canonical_smiles') and act.get('standard_value'):
                        try:
                            dp = BBBDataPoint(
                                chembl_id=act.get('molecule_chembl_id', ''),
                                smiles=act['canonical_smiles'],
                                assay_type=assay_type,
                                standard_type=act.get('standard_type', ''),
                                standard_value=float(act['standard_value']),
                                standard_units=act.get('standard_units', ''),
                                standard_relation=act.get('standard_relation', '='),
                                assay_description=act.get('assay_description', '')[:500],
                                target_name=act.get('target_pref_name'),
                                pchembl_value=float(act['pchembl_value']) if act.get('pchembl_value') else None,
                                document_chembl_id=act.get('document_chembl_id', ''),
                                year=act.get('document_year'),
                            )
                            data_points.append(dp)
                        except (ValueError, TypeError):
                            continue

        # Clear checkpoint on completion
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print(f"  Completed {assay_type}, checkpoint cleared")

        return data_points

    def fetch_bbb_data(self) -> int:
        """Fetch BBB permeability data"""
        print("\n[1/5] Fetching BBB permeability data...")
        assay_ids = self._search_assays_by_keywords(self.BBB_KEYWORDS, "BBB")
        data = self._fetch_activities_for_assays(assay_ids, "BBB")
        self.data_points.extend(data)
        print(f"  Collected {len(data)} BBB data points")
        return len(data)

    def fetch_pgp_data(self) -> int:
        """Fetch P-glycoprotein efflux data"""
        print("\n[2/5] Fetching P-gp transporter data...")
        assay_ids = self._search_assays_by_keywords(self.PGP_KEYWORDS, "P-gp")
        data = self._fetch_activities_for_assays(assay_ids, "P-gp")
        self.data_points.extend(data)
        print(f"  Collected {len(data)} P-gp data points")
        return len(data)

    def fetch_bcrp_data(self) -> int:
        """Fetch BCRP transporter data"""
        print("\n[3/5] Fetching BCRP transporter data...")
        assay_ids = self._search_assays_by_keywords(self.BCRP_KEYWORDS, "BCRP")
        data = self._fetch_activities_for_assays(assay_ids, "BCRP")
        self.data_points.extend(data)
        print(f"  Collected {len(data)} BCRP data points")
        return len(data)

    def fetch_caco2_data(self) -> int:
        """Fetch Caco-2 permeability data"""
        print("\n[4/5] Fetching Caco-2 permeability data...")
        assay_ids = self._search_assays_by_keywords(self.CACO2_KEYWORDS, "Caco-2")
        data = self._fetch_activities_for_assays(assay_ids, "Caco-2")
        self.data_points.extend(data)
        print(f"  Collected {len(data)} Caco-2 data points")
        return len(data)

    def fetch_pampa_data(self) -> int:
        """Fetch PAMPA permeability data"""
        print("\n[5/5] Fetching PAMPA permeability data...")
        assay_ids = self._search_assays_by_keywords(self.PAMPA_KEYWORDS, "PAMPA")
        data = self._fetch_activities_for_assays(assay_ids, "PAMPA")
        self.data_points.extend(data)
        print(f"  Collected {len(data)} PAMPA data points")
        return len(data)

    def fetch_all(self) -> pd.DataFrame:
        """Fetch all BBB-related data from ChEMBL"""
        print("="*60)
        print("ChEMBL BBB & Kinetics Data Curation")
        print("="*60)

        self.fetch_bbb_data()
        self.fetch_pgp_data()
        self.fetch_bcrp_data()
        self.fetch_caco2_data()
        self.fetch_pampa_data()

        # Convert to DataFrame
        df = pd.DataFrame([asdict(dp) for dp in self.data_points])

        print("\n" + "="*60)
        print(f"Total raw data points: {len(df)}")
        print(f"API calls made: {self.session_stats['api_calls']}")
        print(f"Cache hits: {self.session_stats['cached_hits']}")

        return df

    def process_and_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and normalize the curated data"""
        print("\nProcessing and normalizing data...")

        if df.empty:
            print("No data to process!")
            return df

        # Remove duplicates (same molecule, same assay type, similar value)
        df = df.drop_duplicates(subset=['chembl_id', 'assay_type', 'standard_type'])
        print(f"After deduplication: {len(df)} records")

        # Filter for relevant standard types
        relevant_types = list(self.RELEVANT_STANDARD_TYPES)
        df_filtered = df[df['standard_type'].isin(relevant_types)].copy()
        print(f"After filtering standard types: {len(df_filtered)} records")

        # Normalize permeability values to log scale where appropriate
        def normalize_papp(row):
            """Normalize Papp values to log scale (cm/s)"""
            value = row['standard_value']
            units = str(row['standard_units']).lower()

            if 'nm/s' in units or 'nm s-1' in units:
                # Convert nm/s to cm/s
                value_cm = value * 1e-7
            elif 'um/s' in units or 'µm/s' in units:
                value_cm = value * 1e-4
            elif 'cm/s' in units:
                value_cm = value
            else:
                return value  # Return as-is if units unclear

            # Return log10(Papp)
            if value_cm > 0:
                return np.log10(value_cm)
            return value

        # Apply normalization for Papp values
        papp_mask = df_filtered['standard_type'].isin(['Papp', 'Pe', 'Permeability'])
        if papp_mask.any():
            df_filtered.loc[papp_mask, 'normalized_value'] = df_filtered[papp_mask].apply(
                normalize_papp, axis=1
            )

        # For other types, just copy the value
        df_filtered['normalized_value'] = df_filtered.apply(
            lambda row: row.get('normalized_value', row['standard_value']),
            axis=1
        )

        # Add BBB classification based on various criteria
        def classify_bbb(row):
            """Classify as BBB+ or BBB- based on assay type and value"""
            assay_type = row['assay_type']
            std_type = row['standard_type']
            value = row['standard_value']

            # LogBB classification
            if std_type in ['LogBB', 'log BB']:
                return 'BBB+' if value > -0.3 else 'BBB-'

            # Efflux ratio (P-gp)
            if std_type in ['ER', 'Efflux ratio', 'Efflux Ratio']:
                return 'BBB-' if value > 2.0 else 'BBB+'  # ER > 2 = efflux substrate

            # Papp classification (cm/s)
            if std_type in ['Papp', 'Pe']:
                # High permeability: Papp > 1e-6 cm/s
                if row.get('normalized_value', value) > -6:  # log scale
                    return 'BBB+'
                return 'BBB-'

            # Kp,uu
            if std_type in ['Kp', 'Kp,uu', 'Kpuu']:
                return 'BBB+' if value > 0.3 else 'BBB-'

            return None  # Unknown

        df_filtered['bbb_class'] = df_filtered.apply(classify_bbb, axis=1)

        # Remove rows with no classification
        df_final = df_filtered[df_filtered['bbb_class'].notna()].copy()
        print(f"After BBB classification: {len(df_final)} records")

        # Summary statistics
        print("\nData summary by assay type:")
        print(df_final.groupby('assay_type').size())
        print("\nBBB class distribution:")
        print(df_final['bbb_class'].value_counts())

        return df_final

    def merge_with_existing(self, df: pd.DataFrame,
                            existing_path: str = "data/bbbp_dataset.csv") -> pd.DataFrame:
        """Merge with existing BBBP dataset"""
        print(f"\nMerging with existing dataset: {existing_path}")

        if not os.path.exists(existing_path):
            print(f"Existing dataset not found at {existing_path}")
            return df

        existing_df = pd.read_csv(existing_path)
        print(f"Existing dataset: {len(existing_df)} records")

        # Standardize column names
        df_standardized = df[['smiles', 'bbb_class', 'assay_type',
                              'standard_value', 'normalized_value']].copy()
        df_standardized.columns = ['SMILES', 'BBB_class', 'source',
                                   'raw_value', 'normalized_value']
        df_standardized['source'] = 'ChEMBL_' + df_standardized['source']

        # Add source to existing
        if 'source' not in existing_df.columns:
            existing_df['source'] = 'BBBP_original'

        # Combine
        combined = pd.concat([existing_df, df_standardized], ignore_index=True)

        # Remove duplicates by SMILES (keep first = original)
        combined = combined.drop_duplicates(subset=['SMILES'], keep='first')
        print(f"Combined dataset: {len(combined)} unique molecules")

        return combined

    def save(self, df: pd.DataFrame, output_dir: str = "data"):
        """Save curated data"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save full data
        csv_path = output_path / f"chembl_bbb_kinetics_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved to: {csv_path}")

        # Save pickle for faster loading
        pkl_path = output_path / f"chembl_bbb_kinetics_{timestamp}.pkl"
        df.to_pickle(pkl_path)

        # Save summary stats
        stats = {
            "total_records": len(df),
            "unique_molecules": df['smiles'].nunique() if 'smiles' in df.columns else df['SMILES'].nunique(),
            "by_assay_type": df['assay_type'].value_counts().to_dict() if 'assay_type' in df.columns else {},
            "by_bbb_class": df['bbb_class'].value_counts().to_dict() if 'bbb_class' in df.columns else df['BBB_class'].value_counts().to_dict(),
            "timestamp": timestamp,
        }

        stats_path = output_path / f"chembl_curation_stats_{timestamp}.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        return csv_path


def main():
    """Main curation workflow"""
    curator = ChEMBLDataCurator()

    # Fetch all data
    raw_df = curator.fetch_all()

    if raw_df.empty:
        print("\nNo data fetched. Check network connection or try again later.")
        return

    # Process and normalize
    processed_df = curator.process_and_normalize(raw_df)

    # Merge with existing
    merged_df = curator.merge_with_existing(processed_df)

    # Save
    output_path = curator.save(merged_df)

    print("\n" + "="*60)
    print("Curation complete!")
    print(f"Output: {output_path}")
    print("="*60)

    return merged_df


if __name__ == "__main__":
    main()

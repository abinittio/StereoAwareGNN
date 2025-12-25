import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, AllChem
from torch_geometric.data import Data


# Electronegativity values (Pauling scale) for common atoms
ELECTRONEGATIVITY = {
    1: 2.20,   # H
    6: 2.55,   # C
    7: 3.04,   # N
    8: 3.44,   # O
    9: 3.98,   # F
    15: 2.19,  # P
    16: 2.58,  # S
    17: 3.16,  # Cl
    35: 2.96,  # Br
    53: 2.66,  # I
}

# Polar atoms that contribute to TPSA and reduce BBB permeability
POLAR_ATOMS = {7, 8, 15, 16}  # N, O, P, S


def get_atom_features(atom):
    """
    Extract comprehensive features for a single atom

    Features (15 total) - Enhanced for BBB polarity prediction:

    Basic features (9):
    1. Atomic number (normalized)
    2. Degree (number of bonded atoms)
    3. Formal charge
    4. Hybridization (encoded as number)
    5. Is aromatic (binary)
    6. Is in ring (binary)
    7. Implicit valence
    8. Explicit valence
    9. Mass (normalized)

    Polarity features (6) - NEW for BBB:
    10. Electronegativity (normalized)
    11. Is polar atom (N, O, P, S - binary)
    12. Is H-bond donor (binary)
    13. Is H-bond acceptor (binary)
    14. Gasteiger partial charge (polarity indicator)
    15. Is in polar functional group (binary)
    """
    features = []

    # === BASIC FEATURES (1-9) ===

    # 1. Atomic number (normalized by 100, typical max for organic molecules)
    features.append(atom.GetAtomicNum() / 100.0)

    # 2. Degree (number of bonds)
    features.append(atom.GetDegree())

    # 3. Formal charge
    features.append(atom.GetFormalCharge())

    # 4. Hybridization (encoded as number: SP=1, SP2=2, SP3=3, etc.)
    hybridization_map = {
        Chem.HybridizationType.S: 0,
        Chem.HybridizationType.SP: 1,
        Chem.HybridizationType.SP2: 2,
        Chem.HybridizationType.SP3: 3,
        Chem.HybridizationType.SP3D: 4,
        Chem.HybridizationType.SP3D2: 5,
    }
    features.append(hybridization_map.get(atom.GetHybridization(), 0))

    # 5. Aromatic
    features.append(1 if atom.GetIsAromatic() else 0)

    # 6. In ring
    features.append(1 if atom.IsInRing() else 0)

    # 7. Implicit valence
    features.append(atom.GetTotalValence() - atom.GetTotalDegree())

    # 8. Total valence (replaces explicit valence)
    features.append(atom.GetTotalValence())

    # 9. Atomic mass (normalized by 200)
    features.append(atom.GetMass() / 200.0)

    # === POLARITY FEATURES (10-15) - Critical for BBB ===

    atomic_num = atom.GetAtomicNum()

    # 10. Electronegativity (normalized by 4.0, max is F at 3.98)
    electronegativity = ELECTRONEGATIVITY.get(atomic_num, 2.5)
    features.append(electronegativity / 4.0)

    # 11. Is polar atom (N, O, P, S contribute to TPSA)
    is_polar = 1 if atomic_num in POLAR_ATOMS else 0
    features.append(is_polar)

    # 12. Is H-bond donor (N-H or O-H)
    is_h_donor = 0
    if atomic_num in [7, 8]:  # N or O
        total_h = atom.GetTotalNumHs()
        if total_h > 0:
            is_h_donor = 1
    features.append(is_h_donor)

    # 13. Is H-bond acceptor (N, O with lone pairs)
    is_h_acceptor = 0
    if atomic_num == 7:  # Nitrogen
        # Check if nitrogen has lone pair available
        if atom.GetDegree() < 4 and atom.GetFormalCharge() <= 0:
            is_h_acceptor = 1
    elif atomic_num == 8:  # Oxygen
        # Oxygen typically always an acceptor unless positively charged
        if atom.GetFormalCharge() <= 0:
            is_h_acceptor = 1
    features.append(is_h_acceptor)

    # 14. Gasteiger partial charge (computed at molecule level, use approximation)
    # Approximation based on electronegativity difference from C
    c_en = 2.55  # Carbon electronegativity
    charge_approx = (electronegativity - c_en) / 2.0  # Normalized diff
    features.append(charge_approx)

    # 15. Is in polar functional group
    # Check if atom is part of common polar groups (COOH, OH, NH2, C=O, etc.)
    in_polar_group = 0
    if atomic_num in POLAR_ATOMS:
        # Check neighbors for polar group patterns
        for neighbor in atom.GetNeighbors():
            neighbor_num = neighbor.GetAtomicNum()
            if neighbor_num in POLAR_ATOMS or neighbor_num == 6:
                # C=O, N-H, O-H patterns
                bond = atom.GetOwningMol().GetBondBetweenAtoms(
                    atom.GetIdx(), neighbor.GetIdx()
                )
                if bond and bond.GetBondTypeAsDouble() >= 2.0:
                    in_polar_group = 1
                    break
        # Also flag if polar atom with H
        if atom.GetTotalNumHs() > 0:
            in_polar_group = 1
    features.append(in_polar_group)

    return features


def mol_to_graph(smiles, y=None):
    """
    Convert SMILES string to PyTorch Geometric graph Data object

    Args:
        smiles: SMILES string representation of molecule
        y: Optional target value (BBB permeability)

    Returns:
        PyTorch Geometric Data object with:
        - x: Node features [num_atoms, 15] (9 basic + 6 polarity features)
        - edge_index: Graph connectivity [2, num_bonds*2] (bidirectional)
        - y: Target value (if provided)
        - smiles: Original SMILES string
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))

    x = torch.tensor(atom_features, dtype=torch.float)

    # Get edges (bonds) - create bidirectional edges
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # Add both directions
        edge_indices.append([i, j])
        edge_indices.append([j, i])

    if len(edge_indices) == 0:
        # Handle single-atom molecules (rare but possible)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    # Create Data object
    data = Data(x=x, edge_index=edge_index)

    # Add target if provided
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float)

    # Store SMILES for reference
    data.smiles = smiles

    return data


def estimate_logd(mol, logp, ph=7.4):
    """
    Estimate LogD at physiological pH (7.4)

    LogD accounts for ionization state, which is critical for BBB permeability.
    At pH 7.4, basic amines are protonated (cationic) and acids are deprotonated (anionic).
    Ionized molecules have much lower BBB permeability.

    LogD = LogP - log(1 + 10^(pKa - pH)) for acids
    LogD = LogP - log(1 + 10^(pH - pKa)) for bases

    Since we don't have exact pKa values, we estimate based on functional groups.
    """
    # Count ionizable groups
    num_basic_n = 0  # Basic nitrogens (likely protonated at pH 7.4)
    num_acidic = 0   # Acidic groups (likely deprotonated at pH 7.4)

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 7:  # Nitrogen
            # Primary/secondary amines and non-aromatic tertiary amines are basic
            if atom.GetTotalNumHs() > 0 or (atom.GetDegree() == 3 and not atom.GetIsAromatic()):
                # Check if it's an amide (not basic)
                is_amide = False
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetAtomicNum() == 6:  # Carbon
                        for n_neighbor in neighbor.GetNeighbors():
                            if n_neighbor.GetAtomicNum() == 8:
                                bond = mol.GetBondBetweenAtoms(neighbor.GetIdx(), n_neighbor.GetIdx())
                                if bond and bond.GetBondTypeAsDouble() == 2.0:
                                    is_amide = True
                                    break
                if not is_amide:
                    num_basic_n += 1

        elif atom.GetAtomicNum() == 8:  # Oxygen
            # Carboxylic acids are acidic
            if atom.GetTotalNumHs() > 0:  # O-H
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetAtomicNum() == 6:  # C-OH
                        for n_neighbor in neighbor.GetNeighbors():
                            if n_neighbor.GetAtomicNum() == 8 and n_neighbor.GetIdx() != atom.GetIdx():
                                bond = mol.GetBondBetweenAtoms(neighbor.GetIdx(), n_neighbor.GetIdx())
                                if bond and bond.GetBondTypeAsDouble() == 2.0:
                                    num_acidic += 1  # COOH pattern
                                    break

    # Estimate LogD adjustment
    # Each ionizable group reduces effective lipophilicity
    # Typical pKa for amines ~9-10, so at pH 7.4, ~98% are protonated
    # Typical pKa for carboxylic acids ~4-5, so at pH 7.4, ~99% are deprotonated
    logd_adjustment = 0
    if num_basic_n > 0:
        logd_adjustment -= num_basic_n * 1.5  # Protonated amines are much less lipophilic
    if num_acidic > 0:
        logd_adjustment -= num_acidic * 2.0  # Deprotonated acids are very hydrophilic

    return logp + logd_adjustment


def detect_amphetamine_pattern(mol):
    """
    Detect amphetamine-like structures with neighboring functional groups
    that may require further research for BBB prediction.

    Amphetamine core: phenethylamine with alpha-methyl
    Pattern: phenyl-CH2-CH(CH3)-NH2

    Returns dict with detection results
    """
    result = {
        'is_amphetamine_like': False,
        'has_functional_neighbors': False,
        'functional_groups': [],
        'needs_further_research': False
    }

    # SMARTS patterns for amphetamine-like structures
    # Basic phenethylamine: c1ccccc1CCN
    phenethylamine_pattern = Chem.MolFromSmarts('c1ccccc1CCN')
    # Amphetamine (alpha-methyl): c1ccccc1CC(C)N
    amphetamine_pattern = Chem.MolFromSmarts('c1ccccc1CC(C)N')

    if mol.HasSubstructMatch(amphetamine_pattern):
        result['is_amphetamine_like'] = True
    elif mol.HasSubstructMatch(phenethylamine_pattern):
        result['is_amphetamine_like'] = True

    if result['is_amphetamine_like']:
        # Check for neighboring functional groups that complicate BBB prediction
        functional_patterns = {
            'hydroxyl': Chem.MolFromSmarts('[OH]'),
            'methoxy': Chem.MolFromSmarts('[OX2]C'),
            'halogen': Chem.MolFromSmarts('[F,Cl,Br,I]'),
            'methylenedioxy': Chem.MolFromSmarts('OCO'),
            'nitro': Chem.MolFromSmarts('[N+](=O)[O-]'),
            'amino': Chem.MolFromSmarts('[NH2]'),
            'carbonyl': Chem.MolFromSmarts('C=O'),
            'sulfonyl': Chem.MolFromSmarts('S(=O)(=O)'),
        }

        for name, pattern in functional_patterns.items():
            if pattern and mol.HasSubstructMatch(pattern):
                result['functional_groups'].append(name)
                result['has_functional_neighbors'] = True

        # Flag for further research if amphetamine with functional modifications
        if result['has_functional_neighbors']:
            result['needs_further_research'] = True

    return result


def get_molecular_descriptors(smiles):
    """
    Calculate molecular descriptors for BBB permeability prediction

    Enhanced with:
    - LogD (pH 7.4) instead of just LogP
    - Size vs polarity relationship metrics
    - Amphetamine detection for further research flagging

    Returns dict with key descriptors
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    # Basic descriptors
    logp = Descriptors.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    tpsa = Descriptors.TPSA(mol)
    num_h_donors = Descriptors.NumHDonors(mol)
    num_h_acceptors = Descriptors.NumHAcceptors(mol)

    descriptors = {
        'molecular_weight': mw,
        'logp': logp,
        'tpsa': tpsa,
        'num_h_donors': num_h_donors,
        'num_h_acceptors': num_h_acceptors,
        'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
        'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
        'num_atoms': mol.GetNumAtoms(),
        'num_bonds': mol.GetNumBonds(),
        'num_heteroatoms': Descriptors.NumHeteroatoms(mol),
        'fraction_csp3': Descriptors.FractionCSP3(mol),
        'num_rings': Descriptors.RingCount(mol),
    }

    # === NEW: LogD at pH 7.4 (better than LogP for BBB) ===
    descriptors['logd'] = estimate_logd(mol, logp, ph=7.4)

    # === NEW: Size vs Polarity relationship features ===

    # Polarity density: TPSA per unit molecular weight
    # Higher = more polar per size = worse BBB permeability
    descriptors['polarity_density'] = tpsa / mw if mw > 0 else 0

    # Hydrophilicity per size: H-bond sites per MW
    hb_total = num_h_donors + num_h_acceptors
    descriptors['hb_density'] = hb_total / mw * 100 if mw > 0 else 0

    # Size-adjusted LogD: LogD normalized by size
    # Larger molecules need higher lipophilicity to cross BBB
    descriptors['logd_per_100da'] = descriptors['logd'] / (mw / 100) if mw > 0 else 0

    # Polarity-lipophilicity balance
    # Optimal BBB: low polarity, moderate lipophilicity
    if tpsa > 0:
        descriptors['logd_tpsa_ratio'] = descriptors['logd'] / tpsa
    else:
        descriptors['logd_tpsa_ratio'] = descriptors['logd']

    # === NEW: Amphetamine detection ===
    amphetamine_info = detect_amphetamine_pattern(mol)
    descriptors['is_amphetamine_like'] = amphetamine_info['is_amphetamine_like']
    descriptors['amphetamine_needs_research'] = amphetamine_info['needs_further_research']
    descriptors['amphetamine_functional_groups'] = amphetamine_info['functional_groups']

    # Lipinski's Rule of 5 violations (drug-likeness)
    lipinski_violations = 0
    if descriptors['molecular_weight'] > 500:
        lipinski_violations += 1
    if descriptors['logp'] > 5:
        lipinski_violations += 1
    if descriptors['num_h_donors'] > 5:
        lipinski_violations += 1
    if descriptors['num_h_acceptors'] > 10:
        lipinski_violations += 1

    descriptors['lipinski_violations'] = lipinski_violations

    # BBB-specific rules (using LogD instead of LogP)
    bbb_compliant = (
        descriptors['molecular_weight'] <= 450 and
        descriptors['logd'] >= 0.5 and descriptors['logd'] <= 4.5 and  # LogD range adjusted
        descriptors['tpsa'] <= 90 and
        descriptors['num_h_donors'] <= 3 and
        descriptors['num_h_acceptors'] <= 7
    )

    descriptors['bbb_rule_compliant'] = bbb_compliant

    return descriptors


def batch_smiles_to_graphs(smiles_list, y_list=None):
    """
    Convert multiple SMILES to graph Data objects

    Args:
        smiles_list: List of SMILES strings
        y_list: Optional list of target values

    Returns:
        List of Data objects (skips invalid SMILES)
    """
    graphs = []

    for i, smiles in enumerate(smiles_list):
        y = y_list[i] if y_list is not None else None
        graph = mol_to_graph(smiles, y)

        if graph is not None:
            graphs.append(graph)

    return graphs


if __name__ == "__main__":
    # Test molecule-to-graph conversion
    print("Testing Molecule-to-Graph Conversion")
    print("=" * 60)

    test_molecules = [
        ('CCO', 'Ethanol'),
        ('c1ccccc1', 'Benzene'),
        ('CC(=O)O', 'Acetic Acid'),
        ('COC(=O)C1C(CC2CC1N2C)c3cccc(c3)OC', 'Cocaine'),
    ]

    for smiles, name in test_molecules:
        print(f"\n{name} ({smiles}):")

        # Convert to graph
        graph = mol_to_graph(smiles, y=0.8)

        if graph is not None:
            print(f"  Nodes (atoms): {graph.x.shape[0]}")
            print(f"  Node features: {graph.x.shape[1]}")
            print(f"  Edges (bonds): {graph.edge_index.shape[1] // 2}")
            print(f"  Target value: {graph.y.item():.2f}")

            # Get molecular descriptors
            descriptors = get_molecular_descriptors(smiles)
            print(f"  Molecular Weight: {descriptors['molecular_weight']:.1f}")
            print(f"  LogP: {descriptors['logp']:.2f}")
            print(f"  TPSA: {descriptors['tpsa']:.1f}")
            print(f"  BBB Rule Compliant: {descriptors['bbb_rule_compliant']}")
        else:
            print("  FAILED to convert!")

    # Test batch conversion
    print("\n" + "=" * 60)
    print("Testing Batch Conversion:")
    smiles_batch = ['CCO', 'c1ccccc1', 'CC(=O)O']
    y_batch = [0.8, 0.9, 0.3]

    graphs = batch_smiles_to_graphs(smiles_batch, y_batch)
    print(f"Successfully converted {len(graphs)}/{len(smiles_batch)} molecules")

    print("\nMolecule-to-Graph conversion working!")

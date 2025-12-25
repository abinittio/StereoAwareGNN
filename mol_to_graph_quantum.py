"""
Enhanced Molecule-to-Graph Conversion with Quantum Descriptors

This version adds quantum chemical descriptors as graph-level features
that are broadcast to influence node representations.

Features:
- 15 node features (same as before)
- 13 quantum descriptors as graph-level features
- Total: 28 features per node (15 atomic + 13 quantum broadcast)
"""

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from torch_geometric.data import Data

from quantum_descriptors import QuantumDescriptorCalculator

# Initialize quantum calculator (once)
QUANTUM_CALC = QuantumDescriptorCalculator()


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
    Extract comprehensive features for a single atom (15 features)
    Same as original mol_to_graph.py
    """
    features = []

    # === BASIC FEATURES (1-9) ===
    features.append(atom.GetAtomicNum() / 100.0)
    features.append(atom.GetDegree())
    features.append(atom.GetFormalCharge())

    hybridization_map = {
        Chem.HybridizationType.S: 0,
        Chem.HybridizationType.SP: 1,
        Chem.HybridizationType.SP2: 2,
        Chem.HybridizationType.SP3: 3,
        Chem.HybridizationType.SP3D: 4,
        Chem.HybridizationType.SP3D2: 5,
    }
    features.append(hybridization_map.get(atom.GetHybridization(), 0))

    features.append(1 if atom.GetIsAromatic() else 0)
    features.append(1 if atom.IsInRing() else 0)
    features.append(atom.GetTotalValence() - atom.GetTotalDegree())
    features.append(atom.GetTotalValence())
    features.append(atom.GetMass() / 200.0)

    # === POLARITY FEATURES (10-15) ===
    atomic_num = atom.GetAtomicNum()

    electronegativity = ELECTRONEGATIVITY.get(atomic_num, 2.5)
    features.append(electronegativity / 4.0)

    is_polar = 1 if atomic_num in POLAR_ATOMS else 0
    features.append(is_polar)

    is_h_donor = 0
    if atomic_num in [7, 8]:
        if atom.GetTotalNumHs() > 0:
            is_h_donor = 1
    features.append(is_h_donor)

    is_h_acceptor = 0
    if atomic_num == 7:
        if atom.GetDegree() < 4 and atom.GetFormalCharge() <= 0:
            is_h_acceptor = 1
    elif atomic_num == 8:
        if atom.GetFormalCharge() <= 0:
            is_h_acceptor = 1
    features.append(is_h_acceptor)

    c_en = 2.55
    charge_approx = (electronegativity - c_en) / 2.0
    features.append(charge_approx)

    in_polar_group = 0
    if atomic_num in POLAR_ATOMS:
        for neighbor in atom.GetNeighbors():
            neighbor_num = neighbor.GetAtomicNum()
            if neighbor_num in POLAR_ATOMS or neighbor_num == 6:
                bond = atom.GetOwningMol().GetBondBetweenAtoms(
                    atom.GetIdx(), neighbor.GetIdx()
                )
                if bond and bond.GetBondTypeAsDouble() >= 2.0:
                    in_polar_group = 1
                    break
        if atom.GetTotalNumHs() > 0:
            in_polar_group = 1
    features.append(in_polar_group)

    return features


def mol_to_graph_quantum(smiles, y=None, include_quantum=True):
    """
    Convert SMILES to graph with quantum descriptors

    Args:
        smiles: SMILES string
        y: Optional target value
        include_quantum: Whether to include quantum descriptors (default True)

    Returns:
        Data object with:
        - x: Node features [num_atoms, 28] (15 atomic + 13 quantum)
        - edge_index: Graph connectivity
        - quantum_features: Graph-level quantum descriptors [13]
        - y: Target value
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    # Get atom features (15 features per atom)
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))

    atom_features = np.array(atom_features, dtype=np.float32)
    num_atoms = atom_features.shape[0]

    # Get quantum descriptors (13 features)
    if include_quantum:
        quantum_vec = QUANTUM_CALC.calculate_vector(smiles)
        if quantum_vec is None:
            quantum_vec = np.zeros(13, dtype=np.float32)

        # Normalize quantum features
        quantum_vec = normalize_quantum_features(quantum_vec)

        # Broadcast quantum features to all atoms
        # Each atom gets the same molecular-level quantum properties
        quantum_broadcast = np.tile(quantum_vec, (num_atoms, 1))

        # Concatenate: [15 atomic + 13 quantum] = 28 features
        x = np.concatenate([atom_features, quantum_broadcast], axis=1)
    else:
        x = atom_features

    x = torch.tensor(x, dtype=torch.float)

    # Get edges (bonds)
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])

    if len(edge_indices) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    # Create Data object
    data = Data(x=x, edge_index=edge_index)

    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float)

    data.smiles = smiles

    # Store quantum features separately for analysis
    if include_quantum:
        data.quantum_features = torch.tensor(quantum_vec, dtype=torch.float)

    return data


def normalize_quantum_features(quantum_vec):
    """
    Normalize quantum features to reasonable ranges for neural network

    Feature order from quantum_descriptors.py:
    0: homo_approx (-12 to -4 eV) -> normalize to -1 to 1
    1: lumo_approx (-5 to 2 eV) -> normalize to -1 to 1
    2: homo_lumo_gap (0 to 10 eV) -> normalize to 0 to 1
    3: ionization_potential (4 to 12 eV) -> normalize to 0 to 1
    4: electron_affinity (-2 to 5 eV) -> normalize to -1 to 1
    5: electronegativity (2 to 8 eV) -> normalize to 0 to 1
    6: hardness (1 to 5 eV) -> normalize to 0 to 1
    7: softness (0.1 to 5) -> normalize to 0 to 1
    8: electrophilicity (0 to 10 eV) -> normalize to 0 to 1
    9: dipole_moment (0 to 15 D) -> normalize to 0 to 1
    10: polarizability (0 to 50 Å³) -> normalize to 0 to 1
    11: max_partial_charge (-1 to 1) -> keep as is
    12: min_partial_charge (-1 to 1) -> keep as is
    """
    normalized = quantum_vec.copy()

    # HOMO: [-12, -4] -> [-1, 1]
    normalized[0] = (quantum_vec[0] + 8) / 4.0

    # LUMO: [-5, 2] -> [-1, 1]
    normalized[1] = (quantum_vec[1] + 1.5) / 3.5

    # HOMO-LUMO gap: [0, 10] -> [0, 1]
    normalized[2] = quantum_vec[2] / 10.0

    # Ionization potential: [4, 12] -> [0, 1]
    normalized[3] = (quantum_vec[3] - 4) / 8.0

    # Electron affinity: [-2, 5] -> [-1, 1]
    normalized[4] = (quantum_vec[4] + 2) / 7.0 * 2 - 1

    # Electronegativity: [2, 8] -> [0, 1]
    normalized[5] = (quantum_vec[5] - 2) / 6.0

    # Hardness: [1, 5] -> [0, 1]
    normalized[6] = (quantum_vec[6] - 1) / 4.0

    # Softness: [0.1, 5] -> [0, 1]
    normalized[7] = min(quantum_vec[7], 5.0) / 5.0

    # Electrophilicity: [0, 10] -> [0, 1]
    normalized[8] = min(quantum_vec[8], 10.0) / 10.0

    # Dipole moment: [0, 15] -> [0, 1]
    normalized[9] = min(quantum_vec[9], 15.0) / 15.0

    # Polarizability: [0, 50] -> [0, 1]
    normalized[10] = min(quantum_vec[10], 50.0) / 50.0

    # Partial charges: already in [-1, 1]
    # Handle NaN values
    normalized[11] = 0.0 if np.isnan(quantum_vec[11]) else np.clip(quantum_vec[11], -1, 1)
    normalized[12] = 0.0 if np.isnan(quantum_vec[12]) else np.clip(quantum_vec[12], -1, 1)

    # Clip all to reasonable range
    normalized = np.clip(normalized, -2, 2)

    return normalized


def batch_smiles_to_graphs_quantum(smiles_list, y_list=None, include_quantum=True):
    """Convert multiple SMILES to graph Data objects with quantum features"""
    graphs = []

    for i, smiles in enumerate(smiles_list):
        y = y_list[i] if y_list is not None else None
        graph = mol_to_graph_quantum(smiles, y, include_quantum=include_quantum)

        if graph is not None:
            graphs.append(graph)

    return graphs


if __name__ == "__main__":
    print("Testing Molecule-to-Graph with Quantum Descriptors")
    print("=" * 70)

    test_molecules = [
        ('CCO', 'Ethanol'),
        ('c1ccccc1', 'Benzene'),
        ('CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'Caffeine'),
        ('CC(C)Cc1ccc(cc1)C(C)C(=O)O', 'Ibuprofen'),
    ]

    for smiles, name in test_molecules:
        print(f"\n{name} ({smiles}):")

        # Without quantum
        graph_basic = mol_to_graph_quantum(smiles, y=0.8, include_quantum=False)
        print(f"  Basic features: {graph_basic.x.shape[1]} per atom")

        # With quantum
        graph_quantum = mol_to_graph_quantum(smiles, y=0.8, include_quantum=True)
        print(f"  With quantum:   {graph_quantum.x.shape[1]} per atom")
        print(f"  Atoms: {graph_quantum.x.shape[0]}")
        print(f"  Bonds: {graph_quantum.edge_index.shape[1] // 2}")

        # Show quantum features
        qf = graph_quantum.quantum_features
        print(f"  Quantum features (normalized):")
        print(f"    HOMO: {qf[0]:.3f}, LUMO: {qf[1]:.3f}, Gap: {qf[2]:.3f}")
        print(f"    Electronegativity: {qf[5]:.3f}, Hardness: {qf[6]:.3f}")
        print(f"    Dipole: {qf[9]:.3f}, Polarizability: {qf[10]:.3f}")

    print("\n" + "=" * 70)
    print("Quantum-enhanced graph conversion working!")

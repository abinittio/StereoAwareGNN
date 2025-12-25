"""
BBB Permeability Prediction - Stereo-Aware GNN Web Application
State-of-the-Art Model: AUC 0.8968 (5-fold CV)

Accepts:
- Molecule names (e.g., "Aspirin", "Caffeine")
- Molecular formulas (e.g., "C9H8O4")
- SMILES strings (e.g., "CC(=O)Oc1ccccc1C(=O)O")

Run: streamlit run bbb_webapp.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import torch
import torch.nn as nn
from pathlib import Path
import sys
import re
from datetime import datetime

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import io
import base64

# Import our stereo-aware model
from zinc_stereo_pretraining import StereoAwareEncoder
from mol_to_graph_enhanced import mol_to_graph_enhanced

# Try to import PubChemPy for name/formula lookup
try:
    import pubchempy as pcp
    PUBCHEM_AVAILABLE = True
except ImportError:
    PUBCHEM_AVAILABLE = False
    print("Warning: pubchempy not installed. Install with: pip install pubchempy")


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="BBB Predictor | Stereo-GNN",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        text-align: center;
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .model-badge {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin: 0 auto;
    }
    .prediction-card {
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    .prediction-positive {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    .prediction-negative {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        color: white;
    }
    .prediction-moderate {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.8rem 1.5rem;
        border-radius: 10px;
        font-size: 1.1rem;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
    }
    .input-resolved {
        background: #e8f5e9;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
    }
    .input-error {
        background: #ffebee;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid #f44336;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# MODEL LOADING
# ============================================================================
class BBBStereoClassifier(nn.Module):
    """BBB classifier with pretrained stereo encoder."""

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


@st.cache_resource
def load_model():
    """Load the stereo-aware BBB model (cached)."""
    try:
        # Load encoder
        encoder = StereoAwareEncoder(node_features=21, hidden_dim=128, num_layers=4)

        # Create classifier
        model = BBBStereoClassifier(encoder, hidden_dim=128)

        # Load best fold weights (fold 4 had highest AUC: 0.9111)
        model_path = Path(__file__).parent / 'models' / 'bbb_stereo_fold4_best.pth'

        if not model_path.exists():
            # Try other folds
            for fold in [5, 3, 1, 2]:
                alt_path = Path(__file__).parent / 'models' / f'bbb_stereo_fold{fold}_best.pth'
                if alt_path.exists():
                    model_path = alt_path
                    break

        if model_path.exists():
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()
            return model, None, str(model_path.name)
        else:
            return None, "Model file not found", None

    except Exception as e:
        return None, str(e), None


# ============================================================================
# MOLECULE INPUT RESOLUTION
# ============================================================================
COMMON_MOLECULES = {
    # CNS Drugs
    "caffeine": ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine"),
    "cocaine": ("COC(=O)[C@H]1[C@@H]2CC[C@H](C2)N1C", "Cocaine"),
    "morphine": ("CN1CC[C@]23[C@H]4Oc5c(O)ccc(C[C@@H]1[C@@H]2C=C[C@@H]4O)c35", "Morphine"),
    "nicotine": ("CN1CCC[C@H]1c2cccnc2", "Nicotine"),
    "aspirin": ("CC(=O)Oc1ccccc1C(=O)O", "Aspirin"),
    "ibuprofen": ("CC(C)Cc1ccc(cc1)[C@H](C)C(=O)O", "Ibuprofen"),
    "acetaminophen": ("CC(=O)Nc1ccc(O)cc1", "Acetaminophen (Paracetamol)"),
    "paracetamol": ("CC(=O)Nc1ccc(O)cc1", "Paracetamol"),
    "propranolol": ("CC(C)NCC(O)COc1cccc2ccccc12", "Propranolol"),
    "diazepam": ("CN1C(=O)CN=C(c2ccccc2)c3cc(Cl)ccc13", "Diazepam (Valium)"),
    "valium": ("CN1C(=O)CN=C(c2ccccc2)c3cc(Cl)ccc13", "Valium"),
    "sertraline": ("CN[C@H]1CC[C@@H](c2ccc(Cl)c(Cl)c2)c3ccccc13", "Sertraline (Zoloft)"),
    "zoloft": ("CN[C@H]1CC[C@@H](c2ccc(Cl)c(Cl)c2)c3ccccc13", "Zoloft"),
    "fluoxetine": ("CNCCC(Oc1ccc(C(F)(F)F)cc1)c2ccccc2", "Fluoxetine (Prozac)"),
    "prozac": ("CNCCC(Oc1ccc(C(F)(F)F)cc1)c2ccccc2", "Prozac"),

    # Amphetamines
    "amphetamine": ("CC(Cc1ccccc1)N", "Amphetamine"),
    "methamphetamine": ("CC(Cc1ccccc1)NC", "Methamphetamine"),
    "mdma": ("CC(Cc1ccc2OCOc2c1)NC", "MDMA (Ecstasy)"),
    "ecstasy": ("CC(Cc1ccc2OCOc2c1)NC", "Ecstasy"),
    "adderall": ("CC(Cc1ccccc1)N", "Adderall"),
    "ritalin": ("COC(=O)[C@H](c1ccccc1)[C@@H]2CCCCN2", "Ritalin (Methylphenidate)"),
    "methylphenidate": ("COC(=O)[C@H](c1ccccc1)[C@@H]2CCCCN2", "Methylphenidate"),

    # Opioids
    "fentanyl": ("CCC(=O)N(c1ccccc1)[C@@H]2CCN(CCc3ccccc3)CC2", "Fentanyl"),
    "oxycodone": ("CN1CC[C@]23[C@@H]4OC(=O)[C@H]1[C@@H]2c1ccc(O)c(OC)c1[C@@H]3O[C@@H]4O", "Oxycodone"),
    "codeine": ("COc1ccc2[C@H]3Oc4c(O)ccc(C[C@@H]5N(C)CC[C@]23[C@@H]4C=C5)c14", "Codeine"),
    "heroin": ("CC(=O)O[C@H]1C=C[C@H]2[C@H]3CC4=C5C(=C(OC(C)=O)C=C4)[C@@]12CCN3C5", "Heroin (Diacetylmorphine)"),

    # Neurotransmitters
    "dopamine": ("NCCc1ccc(O)c(O)c1", "Dopamine"),
    "serotonin": ("NCCc1c[nH]c2ccc(O)cc12", "Serotonin"),
    "gaba": ("NCCCC(=O)O", "GABA"),
    "glutamate": ("N[C@@H](CCC(=O)O)C(=O)O", "Glutamate"),
    "acetylcholine": ("CC(=O)OCC[N+](C)(C)C", "Acetylcholine"),
    "norepinephrine": ("NC[C@H](O)c1ccc(O)c(O)c1", "Norepinephrine"),
    "epinephrine": ("CNC[C@H](O)c1ccc(O)c(O)c1", "Epinephrine (Adrenaline)"),
    "adrenaline": ("CNC[C@H](O)c1ccc(O)c(O)c1", "Adrenaline"),

    # Simple molecules
    "ethanol": ("CCO", "Ethanol"),
    "alcohol": ("CCO", "Ethanol (Alcohol)"),
    "glucose": ("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "Glucose"),
    "water": ("O", "Water"),
    "benzene": ("c1ccccc1", "Benzene"),
    "toluene": ("Cc1ccccc1", "Toluene"),

    # Common drugs
    "melatonin": ("CC(=O)NCCc1c[nH]c2ccc(OC)cc12", "Melatonin"),
    "thc": ("CCCCCc1cc(O)c2[C@@H]3C=C(C)CC[C@H]3C(C)(C)Oc2c1", "THC (Tetrahydrocannabinol)"),
    "cbd": ("CCCCCc1cc(O)c(c(O)c1)[C@H]2C=C(C)CC[C@H]2C(=C)C", "CBD (Cannabidiol)"),
    "lsd": ("CCN(CC)C(=O)[C@H]1CN([C@@H]2Cc3c[nH]c4cccc(C2=C1)c34)C", "LSD"),
    "psilocybin": ("CN(C)CCc1c[nH]c2cccc(OP(=O)(O)O)c12", "Psilocybin"),

    # Antibiotics (typically don't cross BBB)
    "penicillin": ("CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)Cc3ccccc3)C(=O)O)C", "Penicillin G"),
    "amoxicillin": ("CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)[C@@H](c3ccc(O)cc3)N)C(=O)O)C", "Amoxicillin"),
}


def is_smiles(text):
    """Check if text is a valid SMILES string."""
    if not text or len(text) < 1:
        return False
    mol = Chem.MolFromSmiles(text)
    return mol is not None


def is_molecular_formula(text):
    """Check if text looks like a molecular formula."""
    # Pattern: starts with capital letter, contains only element symbols and numbers
    pattern = r'^[A-Z][a-zA-Z0-9]*$'
    if not re.match(pattern, text):
        return False
    # Must have at least one capital and could have numbers
    if not re.search(r'[A-Z]', text):
        return False
    return True


def lookup_pubchem(query, search_type='name'):
    """Look up molecule on PubChem."""
    if not PUBCHEM_AVAILABLE:
        return None, "PubChem lookup not available (install pubchempy)"

    try:
        if search_type == 'name':
            results = pcp.get_compounds(query, 'name')
        elif search_type == 'formula':
            results = pcp.get_compounds(query, 'formula')
        else:
            return None, "Unknown search type"

        if results:
            compound = results[0]
            smiles = compound.canonical_smiles
            name = compound.iupac_name or query
            return smiles, name
        else:
            return None, f"No results found for '{query}'"

    except Exception as e:
        return None, f"PubChem error: {str(e)}"


def resolve_molecule_input(user_input):
    """
    Resolve user input to SMILES string.

    Returns: (smiles, display_name, input_type, message)
    """
    if not user_input:
        return None, None, None, "Please enter a molecule"

    user_input = user_input.strip()

    # 1. Check if it's already a valid SMILES
    if is_smiles(user_input):
        mol = Chem.MolFromSmiles(user_input)
        # Try to get a name from the structure
        return user_input, "Custom Molecule", "smiles", "Valid SMILES string"

    # 2. Check local database (case-insensitive)
    lookup_key = user_input.lower().strip()
    if lookup_key in COMMON_MOLECULES:
        smiles, name = COMMON_MOLECULES[lookup_key]
        return smiles, name, "database", f"Found in local database"

    # 3. Try PubChem name lookup
    if PUBCHEM_AVAILABLE:
        smiles, result = lookup_pubchem(user_input, 'name')
        if smiles:
            return smiles, result, "pubchem_name", f"Found via PubChem"

    # 4. Check if it's a molecular formula and try PubChem
    if is_molecular_formula(user_input) and PUBCHEM_AVAILABLE:
        smiles, result = lookup_pubchem(user_input, 'formula')
        if smiles:
            return smiles, result, "pubchem_formula", f"Found formula match via PubChem"

    # 5. Nothing found
    return None, None, "error", f"Could not resolve '{user_input}'. Try a SMILES string, drug name, or molecular formula."


# ============================================================================
# PREDICTION
# ============================================================================
def predict_bbb(model, smiles):
    """Predict BBB permeability for a SMILES string."""
    try:
        # Convert to stereo-aware graph (21 features)
        graph = mol_to_graph_enhanced(
            smiles,
            y=0,  # Dummy label
            include_quantum=False,
            include_stereo=True,
            use_dft=False
        )

        if graph is None:
            return None, "Failed to convert molecule to graph"

        if graph.x.shape[1] != 21:
            return None, f"Feature mismatch: expected 21, got {graph.x.shape[1]}"

        # Create batch
        graph.batch = torch.zeros(graph.x.shape[0], dtype=torch.long)

        # Predict
        with torch.no_grad():
            logit = model(graph.x, graph.edge_index, graph.batch)
            prob = torch.sigmoid(logit).item()

        return prob, None

    except Exception as e:
        return None, str(e)


def get_molecular_properties(smiles):
    """Calculate molecular properties for display."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    props = {
        'molecular_weight': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'tpsa': Descriptors.TPSA(mol),
        'num_h_donors': Descriptors.NumHDonors(mol),
        'num_h_acceptors': Descriptors.NumHAcceptors(mol),
        'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
        'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
        'num_atoms': mol.GetNumAtoms(),
        'num_heavy_atoms': mol.GetNumHeavyAtoms(),
        'formula': Chem.rdMolDescriptors.CalcMolFormula(mol),
    }

    # BBB rules check (Lipinski-like for CNS)
    props['bbb_rules'] = {
        'mw_ok': 150 <= props['molecular_weight'] <= 500,
        'logp_ok': 0 <= props['logp'] <= 5,
        'tpsa_ok': props['tpsa'] <= 90,
        'hbd_ok': props['num_h_donors'] <= 3,
        'hba_ok': props['num_h_acceptors'] <= 7,
    }
    props['bbb_rules_passed'] = sum(props['bbb_rules'].values())

    return props


def mol_to_image(smiles, size=(400, 300)):
    """Generate molecule image from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Generate 2D coordinates
    AllChem.Compute2DCoords(mol)

    # Draw molecule
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    drawer.drawOptions().addStereoAnnotation = True
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    # Convert to base64
    img_data = drawer.GetDrawingText()
    b64 = base64.b64encode(img_data).decode()

    return f"data:image/png;base64,{b64}"


# ============================================================================
# VISUALIZATION
# ============================================================================
def create_gauge_chart(score):
    """Create a gauge chart for BBB score."""
    # Determine color based on score
    if score >= 0.6:
        bar_color = "#11998e"
    elif score >= 0.4:
        bar_color = "#f093fb"
    else:
        bar_color = "#ee0979"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'font': {'size': 48}, 'valueformat': '.3f'},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "BBB Permeability Score", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 2, 'tickcolor': "#333"},
            'bar': {'color': bar_color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#ccc",
            'steps': [
                {'range': [0, 0.4], 'color': '#ffcdd2'},
                {'range': [0.4, 0.6], 'color': '#fff9c4'},
                {'range': [0.6, 1], 'color': '#c8e6c9'}
            ],
            'threshold': {
                'line': {'color': "#333", 'width': 3},
                'thickness': 0.8,
                'value': score
            }
        }
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=60, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'family': "Inter, sans-serif"}
    )

    return fig


def create_properties_chart(props):
    """Create bar chart for molecular properties."""
    # Normalize for visualization
    data = {
        'Property': ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds'],
        'Value': [
            props['molecular_weight'],
            props['logp'],
            props['tpsa'],
            props['num_h_donors'],
            props['num_h_acceptors'],
            props['num_rotatable_bonds']
        ],
        'Optimal Range': [
            '150-500',
            '0-5',
            '<90',
            '<=3',
            '<=7',
            '<10'
        ]
    }

    df = pd.DataFrame(data)

    # Color based on BBB rules
    colors = []
    rules = props['bbb_rules']
    rule_map = ['mw_ok', 'logp_ok', 'tpsa_ok', 'hbd_ok', 'hba_ok', None]
    for i, rule in enumerate(rule_map):
        if rule and rule in rules:
            colors.append('#4caf50' if rules[rule] else '#f44336')
        else:
            colors.append('#2196f3')

    fig = go.Figure(go.Bar(
        x=df['Property'],
        y=df['Value'],
        marker_color=colors,
        text=[f"{v:.1f}" for v in df['Value']],
        textposition='outside',
        hovertemplate='%{x}<br>Value: %{y:.2f}<br>Optimal: %{customdata}<extra></extra>',
        customdata=df['Optimal Range']
    ))

    fig.update_layout(
        title="Molecular Properties",
        height=300,
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'family': "Inter, sans-serif"},
        yaxis_title="Value",
        showlegend=False
    )

    return fig


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Header
    st.markdown('<h1 class="main-header">BBB Permeability Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Stereo-Aware Graph Neural Network | State-of-the-Art Performance</p>', unsafe_allow_html=True)

    # Model badge
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown('<div style="text-align: center"><span class="model-badge">AUC: 0.8968 | 5-Fold CV</span></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Load model
    model, error, model_name = load_model()

    if error:
        st.error(f"Failed to load model: {error}")
        st.info("Please run the fine-tuning script first: `python finetune_bbb_stereo.py`")
        return

    # Sidebar
    with st.sidebar:
        st.header("Model Information")
        st.success(f"**Model:** {model_name}")

        st.markdown("---")

        st.subheader("Performance Metrics")
        st.metric("Mean AUC", "0.8968", "+6.52% vs baseline")
        st.metric("Mean Accuracy", "85.04%")
        st.metric("Std Dev", "0.0156")

        st.markdown("---")

        st.subheader("Architecture")
        st.markdown("""
        - **Encoder:** StereoAwareEncoder
        - **Features:** 21 (15 atomic + 6 stereo)
        - **Layers:** 4 GATv2 + Transformer
        - **Pretraining:** 322k ZINC molecules
        - **Hidden Dim:** 128
        """)

        st.markdown("---")

        st.subheader("Interpretation")
        st.success("**BBB+** (>=0.6): High permeability")
        st.warning("**BBB+/-** (0.4-0.6): Moderate")
        st.error("**BBB-** (<0.4): Low permeability")

        st.markdown("---")

        st.subheader("Input Types Accepted")
        st.markdown("""
        1. **Drug names:** Aspirin, Caffeine, Morphine...
        2. **Molecular formulas:** C9H8O4, C8H10N4O2...
        3. **SMILES strings:** CC(=O)Oc1ccccc1C(=O)O
        """)

        if not PUBCHEM_AVAILABLE:
            st.warning("Install `pubchempy` for name/formula lookup")

    # Main input area
    st.subheader("Enter Molecule")

    col1, col2 = st.columns([3, 1])

    with col1:
        user_input = st.text_input(
            "Molecule (name, formula, or SMILES)",
            placeholder="e.g., Caffeine, C8H10N4O2, or CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            help="Enter a drug name, molecular formula, or SMILES string",
            label_visibility="collapsed"
        )

    with col2:
        predict_btn = st.button("Predict", type="primary", use_container_width=True)

    # Quick examples
    st.markdown("**Quick examples:**")
    example_cols = st.columns(6)
    examples = ["Caffeine", "Aspirin", "Morphine", "Dopamine", "Glucose", "Ethanol"]

    for i, ex in enumerate(examples):
        with example_cols[i]:
            if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                st.session_state['input'] = ex
                st.rerun()

    # Handle session state for examples
    if 'input' in st.session_state:
        user_input = st.session_state['input']
        del st.session_state['input']
        predict_btn = True

    # Process prediction
    if predict_btn and user_input:
        # Resolve input
        with st.spinner("Resolving molecule..."):
            smiles, display_name, input_type, message = resolve_molecule_input(user_input)

        if smiles is None:
            st.markdown(f'<div class="input-error">{message}</div>', unsafe_allow_html=True)
            return

        # Show resolution result
        st.markdown(f'<div class="input-resolved"><strong>{display_name}</strong> | {message}<br><code>{smiles}</code></div>', unsafe_allow_html=True)

        # Make prediction
        with st.spinner("Analyzing molecular structure..."):
            score, pred_error = predict_bbb(model, smiles)
            props = get_molecular_properties(smiles)
            mol_img = mol_to_image(smiles)

        if pred_error:
            st.error(f"Prediction failed: {pred_error}")
            return

        st.markdown("---")

        # Results header
        st.header(f"Results: {display_name}")

        # Main results row
        col1, col2, col3 = st.columns([1.2, 1, 1])

        with col1:
            # Prediction card
            if score >= 0.6:
                card_class = "prediction-positive"
                category = "BBB+"
                interpretation = "HIGH permeability - likely crosses BBB"
                icon = "white_check_mark"
            elif score >= 0.4:
                card_class = "prediction-moderate"
                category = "BBB+/-"
                interpretation = "MODERATE permeability - may partially cross"
                icon = "warning"
            else:
                card_class = "prediction-negative"
                category = "BBB-"
                interpretation = "LOW permeability - unlikely to cross BBB"
                icon = "x"

            st.markdown(f"""
            <div class="prediction-card {card_class}">
                <h1 style="font-size: 3rem; margin: 0;">:{icon}: {category}</h1>
                <h2 style="font-size: 2.5rem; margin: 0.5rem 0;">{score:.4f}</h2>
                <p style="font-size: 1rem; opacity: 0.9;">{interpretation}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Gauge chart
            st.plotly_chart(create_gauge_chart(score), use_container_width=True)

        with col3:
            # Molecule image
            if mol_img:
                st.markdown(f'<img src="{mol_img}" style="width: 100%; border-radius: 10px; border: 1px solid #ddd;">', unsafe_allow_html=True)
            if props:
                st.markdown(f"**Formula:** {props['formula']}")
                st.markdown(f"**Atoms:** {props['num_atoms']} ({props['num_heavy_atoms']} heavy)")

        # Properties section
        if props:
            st.markdown("---")
            st.subheader("Molecular Properties")

            # Key metrics
            metric_cols = st.columns(6)

            with metric_cols[0]:
                delta_mw = "optimal" if props['bbb_rules']['mw_ok'] else "out of range"
                st.metric("MW (Da)", f"{props['molecular_weight']:.1f}", delta_mw, delta_color="normal" if props['bbb_rules']['mw_ok'] else "inverse")

            with metric_cols[1]:
                delta_logp = "optimal" if props['bbb_rules']['logp_ok'] else "out of range"
                st.metric("LogP", f"{props['logp']:.2f}", delta_logp, delta_color="normal" if props['bbb_rules']['logp_ok'] else "inverse")

            with metric_cols[2]:
                delta_tpsa = "optimal" if props['bbb_rules']['tpsa_ok'] else "too high"
                st.metric("TPSA", f"{props['tpsa']:.1f}", delta_tpsa, delta_color="normal" if props['bbb_rules']['tpsa_ok'] else "inverse")

            with metric_cols[3]:
                delta_hbd = "optimal" if props['bbb_rules']['hbd_ok'] else "too many"
                st.metric("H-Donors", props['num_h_donors'], delta_hbd, delta_color="normal" if props['bbb_rules']['hbd_ok'] else "inverse")

            with metric_cols[4]:
                delta_hba = "optimal" if props['bbb_rules']['hba_ok'] else "too many"
                st.metric("H-Acceptors", props['num_h_acceptors'], delta_hba, delta_color="normal" if props['bbb_rules']['hba_ok'] else "inverse")

            with metric_cols[5]:
                st.metric("BBB Rules", f"{props['bbb_rules_passed']}/5", "passed")

            # Properties chart
            st.plotly_chart(create_properties_chart(props), use_container_width=True)

            # BBB Rules explanation
            with st.expander("BBB Permeability Rules (CNS Drug-likeness)"):
                st.markdown("""
                The blood-brain barrier has specific permeability requirements:

                | Property | Optimal Range | Your Molecule |
                |----------|--------------|---------------|
                | Molecular Weight | 150-500 Da | {:.1f} Da {} |
                | LogP (lipophilicity) | 0-5 | {:.2f} {} |
                | TPSA (polar surface) | <90 A^2 | {:.1f} A^2 {} |
                | H-bond Donors | <=3 | {} {} |
                | H-bond Acceptors | <=7 | {} {} |
                """.format(
                    props['molecular_weight'],
                    "yes" if props['bbb_rules']['mw_ok'] else "no",
                    props['logp'],
                    "yes" if props['bbb_rules']['logp_ok'] else "no",
                    props['tpsa'],
                    "yes" if props['bbb_rules']['tpsa_ok'] else "no",
                    props['num_h_donors'],
                    "yes" if props['bbb_rules']['hbd_ok'] else "no",
                    props['num_h_acceptors'],
                    "yes" if props['bbb_rules']['hba_ok'] else "no",
                ))

        # Download section
        st.markdown("---")

        report_data = {
            'Molecule': display_name,
            'SMILES': smiles,
            'Input Type': input_type,
            'BBB Score': score,
            'Category': category,
            'Interpretation': interpretation,
            'Timestamp': datetime.now().isoformat()
        }

        if props:
            report_data.update({
                'Formula': props['formula'],
                'Molecular Weight': props['molecular_weight'],
                'LogP': props['logp'],
                'TPSA': props['tpsa'],
                'H-Donors': props['num_h_donors'],
                'H-Acceptors': props['num_h_acceptors'],
                'BBB Rules Passed': f"{props['bbb_rules_passed']}/5"
            })

        col1, col2, col3 = st.columns(3)

        with col1:
            df_report = pd.DataFrame([report_data])
            st.download_button(
                "Download CSV",
                df_report.to_csv(index=False),
                f"{display_name.replace(' ', '_')}_BBB_prediction.csv",
                "text/csv",
                use_container_width=True
            )

        with col2:
            import json
            st.download_button(
                "Download JSON",
                json.dumps(report_data, indent=2),
                f"{display_name.replace(' ', '_')}_BBB_prediction.json",
                "application/json",
                use_container_width=True
            )

        with col3:
            st.download_button(
                "Copy SMILES",
                smiles,
                f"{display_name.replace(' ', '_')}.smi",
                "chemical/x-daylight-smiles",
                use_container_width=True
            )


if __name__ == "__main__":
    main()

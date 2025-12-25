# BBB Permeability Web Interface

Beautiful, interactive web application for predicting blood-brain barrier permeability of molecules.

## Features

### 🎨 Beautiful UI
- Modern gradient design
- Responsive layout
- Interactive visualizations
- Real-time predictions

### 📊 Comprehensive Analysis
- **BBB Permeability Score** (0-1 scale)
- **Category Classification** (BBB+, BBB±, BBB-)
- **Molecular Properties** (MW, LogP, TPSA, etc.)
- **Drug-likeness Metrics**
- **BBB Rule Compliance**
- **Warning System** for suboptimal properties

### 🔬 Input Methods
1. **Common Molecules** - Select from 20+ pre-loaded molecules
   - CNS Drugs (Caffeine, Cocaine, Morphine, etc.)
   - Simple Molecules (Ethanol, Benzene, Glucose)
   - Amino Acids (Glycine, Alanine, Tryptophan)
   - Neurotransmitters (Dopamine, Serotonin, GABA)

2. **SMILES String** - Direct SMILES input for any molecule

3. **Molecule Name (Beta)** - Type common drug names

### 📈 Visualizations
- **Gauge Chart** - BBB score visualization
- **Radar Chart** - Drug-likeness profile
- **Bar Chart** - Molecular properties
- **Color-coded Results** - Instant visual feedback

### 💾 Export Options
- CSV export for spreadsheet analysis
- JSON export for programmatic use

## Installation

```bash
# Install required packages
pip install streamlit plotly

# Or install all requirements
pip install -r requirements.txt
```

## Usage

### Launch the Web Interface

```bash
streamlit run app.py
```

Or with environment variable for OpenMP:

```bash
# Windows
set KMP_DUPLICATE_LIB_OK=TRUE
streamlit run app.py

# Linux/Mac
export KMP_DUPLICATE_LIB_OK=TRUE
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### Quick Start Guide

1. **Select Input Mode** in the sidebar
   - Choose "Common Molecules" for quick testing
   - Choose "SMILES String" for custom molecules

2. **Select or Enter Molecule**
   - Browse categories (CNS Drugs, Amino Acids, etc.)
   - Or paste a SMILES string

3. **Click "Predict BBB Permeability"**
   - Get instant results with visualizations

4. **Analyze Results**
   - View BBB score and category
   - Check molecular properties
   - Review warnings if any

5. **Export Results** (optional)
   - Download as CSV or JSON

## Interface Sections

### Sidebar
- **Input Mode Selection**
- **Model Information** (MAE, parameters, architecture)
- **Category Guide** (BBB+, BBB±, BBB-)
- **About Section**

### Main Panel
- **Input Section** - Select/enter molecules
- **Prediction Button** - Trigger analysis
- **Results Display**:
  - Color-coded category box
  - BBB score gauge
  - Drug-likeness radar
  - Property metrics
  - Detailed analysis
  - Warning system
  - Export buttons

## Examples

### Example 1: CNS Drug (Caffeine)
```
Category: BBB+ (High permeability)
Score: 0.782
MW: 194.2 Da
LogP: -1.03
TPSA: 61.8 A^2
```

### Example 2: Amino Acid (Glycine)
```
Category: BBB- (Low permeability)
Score: 0.114
MW: 75.1 Da
LogP: -0.97
TPSA: 63.3 A^2
```

### Example 3: Aromatic (Benzene)
```
Category: BBB+ (High permeability)
Score: 0.802
MW: 78.1 Da
LogP: 1.69
TPSA: 0.0 A^2
```

## Common Molecules Database

The app includes 20+ common molecules:

**CNS Drugs:**
- Caffeine, Cocaine, Morphine, Nicotine
- Aspirin, Ibuprofen, Acetaminophen
- Propranolol

**Simple Molecules:**
- Ethanol, Benzene, Toluene, Glucose

**Amino Acids:**
- Glycine, Alanine, Tryptophan

**Neurotransmitters:**
- Dopamine, Serotonin, GABA

## Technical Details

### Model
- **Architecture:** Hybrid GAT+GraphSAGE GNN
- **Parameters:** 649,345
- **Validation MAE:** 0.0967
- **Training Dataset:** 42 curated compounds

### Visualizations
- **Gauge Chart:** Real-time BBB score with thresholds
- **Radar Chart:** Drug-likeness across 5 properties
- **Bar Chart:** Comprehensive molecular properties

### Color Scheme
- **Green:** BBB+ (High permeability, ≥0.6)
- **Orange:** BBB± (Moderate permeability, 0.4-0.6)
- **Red:** BBB- (Low permeability, <0.4)

## Troubleshooting

### Model Not Found
```
Error: Failed to load model
```
**Solution:** Train the model first:
```bash
python train_gnn.py
```

### OpenMP Error
```
OMP: Error #15: Initializing libiomp5md.dll
```
**Solution:** Set environment variable:
```bash
set KMP_DUPLICATE_LIB_OK=TRUE  # Windows
export KMP_DUPLICATE_LIB_OK=TRUE  # Linux/Mac
```

### Port Already in Use
```
Error: Port 8501 is already in use
```
**Solution:** Specify a different port:
```bash
streamlit run app.py --server.port 8502
```

## Customization

### Add More Molecules
Edit `COMMON_MOLECULES` dictionary in `app.py`:
```python
COMMON_MOLECULES = {
    "Your Molecule": "SMILES_STRING",
    # Add more here
}
```

### Change Theme
Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

### Modify Visualizations
Edit the chart creation functions in `app.py`:
- `create_gauge_chart()` - BBB score gauge
- `create_property_radar()` - Drug-likeness radar
- `create_property_bars()` - Property bars

## Performance

- **Prediction Time:** <1 second per molecule
- **Batch Processing:** Supported via API mode
- **Concurrent Users:** Streamlit caching enables multi-user support

## Future Enhancements

Planned features:
- [ ] Molecule drawing interface (JSME/RDKit)
- [ ] Batch upload (CSV/Excel)
- [ ] 3D molecule visualization
- [ ] Historical predictions tracking
- [ ] Comparison mode (multiple molecules)
- [ ] API endpoint mode
- [ ] Mobile-optimized view
- [ ] Dark theme support

## Screenshots

The interface includes:
1. **Header** - Beautiful gradient title
2. **Sidebar** - Settings and information
3. **Input Section** - Multiple input modes
4. **Results Panel** - Comprehensive analysis
5. **Visualizations** - Interactive charts
6. **Export Options** - Download results

## Support

For issues or questions:
- Check [README.md](README.md) for system documentation
- Review [RESULTS.md](RESULTS.md) for model performance
- See example predictions in `demo.py`

## License

Part of the BBB Permeability Prediction System.

---

**Launch the app:** `streamlit run app.py`

**Enjoy predicting BBB permeability with beautiful visualizations!** 🧬✨

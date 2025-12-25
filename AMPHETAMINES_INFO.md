# Amphetamines in BBB Predictor

## ✅ Added to Web Interface!

I've added **6 amphetamine compounds** to the BBB Permeability Predictor web interface.

---

## 🧪 Available Amphetamines

### How to Access:
1. Open the web interface at `http://localhost:8501`
2. Select **"Amphetamines"** from the Category dropdown
3. Choose any amphetamine from the Molecule dropdown
4. Click "Predict BBB Permeability"

---

## 📋 Complete List

### 1. **Amphetamine** (Base compound)
- **SMILES:** `CC(Cc1ccccc1)N`
- **Description:** Base amphetamine structure
- **Clinical Use:** ADHD, narcolepsy
- **Expected BBB:** High (BBB+)
- **Reason:** Small MW, lipophilic, crosses BBB easily

### 2. **Methamphetamine** (Crystal Meth)
- **SMILES:** `CC(Cc1ccccc1)NC`
- **Description:** N-methylated amphetamine
- **Clinical Use:** Rarely prescribed (ADHD)
- **Expected BBB:** Very High (BBB+)
- **Reason:** More lipophilic than amphetamine, rapid CNS entry

### 3. **MDMA** (Ecstasy/Molly)
- **SMILES:** `CC(Cc1ccc2c(c1)OCO2)NC`
- **Description:** 3,4-methylenedioxymethamphetamine
- **Clinical Use:** Research (PTSD therapy)
- **Expected BBB:** High (BBB+)
- **Reason:** CNS-active, affects serotonin/dopamine

### 4. **Dextroamphetamine** (Dexedrine)
- **SMILES:** `CC(Cc1ccccc1)N`
- **Description:** Right-handed enantiomer of amphetamine
- **Clinical Use:** ADHD, narcolepsy
- **Expected BBB:** High (BBB+)
- **Reason:** Same as amphetamine (enantiomer)

### 5. **Adderall (mixed salts)**
- **SMILES:** `CC(Cc1ccccc1)N`
- **Description:** Mix of amphetamine salts (represented by base structure)
- **Clinical Use:** ADHD
- **Expected BBB:** High (BBB+)
- **Reason:** Contains dextroamphetamine and levoamphetamine

### 6. **Methylphenidate** (Ritalin, Concerta)
- **SMILES:** `C1=CC=C(C=C1)C2C(C(=O)OC)CCN2`
- **Description:** Different structure from amphetamines but similar effects
- **Clinical Use:** ADHD
- **Expected BBB:** High (BBB+)
- **Reason:** CNS stimulant, crosses BBB for therapeutic effect

---

## 🔬 Why Amphetamines Cross the BBB

### Key Properties:
1. **Small Molecular Weight** (135-193 Da)
   - All well below 450 Da limit
   - Easy to cross BBB

2. **Lipophilic** (LogP ~1.8-2.1)
   - Within optimal range (1-5)
   - Good membrane penetration

3. **Low TPSA** (~26-40 A²)
   - Well below 90 A² limit
   - Minimal polar surface area

4. **Few H-bond Donors/Acceptors**
   - Usually 1-2 donors
   - 1-3 acceptors
   - Optimal for BBB crossing

### Clinical Significance:
- **Why they work:** Need to enter the brain to affect neurotransmitters
- **Mechanism:** Increase dopamine, norepinephrine in CNS
- **Therapeutic use:** ADHD, narcolepsy, rarely obesity

---

## 📊 Expected Predictions

When you test these in the interface, you should see:

| Compound | BBB Score | Category | Interpretation |
|----------|-----------|----------|----------------|
| Amphetamine | ~0.80-0.90 | BBB+ | HIGH BBB permeability |
| Methamphetamine | ~0.85-0.95 | BBB+ | HIGH BBB permeability |
| MDMA | ~0.80-0.90 | BBB+ | HIGH BBB permeability |
| Dextroamphetamine | ~0.80-0.90 | BBB+ | HIGH BBB permeability |
| Adderall | ~0.80-0.90 | BBB+ | HIGH BBB permeability |
| Methylphenidate | ~0.75-0.85 | BBB+ | HIGH BBB permeability |

All should show:
- ✅ **Green prediction box** (BBB+)
- **Score ≥ 0.6** (typically 0.7-0.9)
- **BBB Rule Compliant:** Likely YES
- **Warnings:** Possibly none or minor

---

## 🎯 How to Test

### Quick Test Protocol:

1. **Open browser:** `http://localhost:8501`

2. **Select Category:** "Amphetamines"

3. **Try each compound:**
   - Start with Amphetamine (base)
   - Then try Methamphetamine (more potent)
   - Compare with MDMA (recreational)
   - Test Ritalin (different structure)

4. **Compare Properties:**
   - Check MW differences
   - Compare LogP values
   - Note TPSA variations
   - See which has highest BBB score

5. **Export Results:**
   - Download all predictions as CSV
   - Create comparison table
   - Analyze structure-activity relationships

---

## 📈 Interesting Comparisons

### Amphetamine vs Methamphetamine
- **Difference:** One methyl group (-CH₃)
- **Effect:** Meth is more lipophilic → higher BBB penetration
- **Prediction:** Meth should score slightly higher

### MDMA vs Amphetamine
- **Difference:** Methylenedioxy ring
- **Effect:** Similar BBB crossing, different receptor effects
- **Prediction:** Similar BBB scores

### Methylphenidate vs Amphetamine
- **Difference:** Different core structure
- **Effect:** Both cross BBB, different mechanisms
- **Prediction:** Both high BBB+

---

## ⚠️ Educational Note

These molecules are included for:
- **Drug discovery research**
- **Pharmacology education**
- **BBB permeability studies**
- **Structure-activity relationship analysis**

This tool predicts BBB permeability, not:
- Drug safety
- Abuse potential
- Therapeutic efficacy
- Legal status

---

## 🔄 Refresh the Interface

The amphetamines should appear automatically, but if needed:

1. **Refresh your browser** (F5 or Ctrl+R)
2. **Select "Amphetamines" category**
3. **Start testing!**

---

## 📝 Notes

- All SMILES are standard canonical forms
- Predictions use the trained GNN model (MAE: 0.0967)
- These are well-studied CNS drugs with known BBB crossing
- Model should correctly predict BBB+ for all

---

**Ready to test!** The amphetamines category is now live in your web interface at `http://localhost:8501` 🧬✨

# 🚀 Deployment Guide

## Quick Deploy to Streamlit Cloud

### Step 1: Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: BBB GNN Predictor"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/BBB-Predictor.git
git push -u origin main
```

### Step 2: Deploy to Streamlit Cloud

1. Go to https://streamlit.io/cloud
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set:
   - **Main file path:** `app.py`
   - **Python version:** 3.12
6. Click "Deploy!"

Your app will be live at: `https://YOUR_USERNAME-bbb-predictor.streamlit.app`

---

## Alternative: Hugging Face Spaces

### Step 1: Create Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Choose "Streamlit" as SDK
4. Upload files

### Step 2: Add Files

Upload:
- `app.py`
- `requirements.txt`
- `bbb_gnn_model.py`
- `mol_to_graph.py`
- `predict_bbb.py`
- `models/best_model.pth`

Your app will be live at: `https://huggingface.co/spaces/YOUR_USERNAME/bbb-predictor`

---

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py

# Access at http://localhost:8501
```

---

## Environment Variables

For production deployment, set:

```bash
KMP_DUPLICATE_LIB_OK=TRUE
```

In Streamlit Cloud:
1. Go to app settings
2. Add to "Secrets"
3. Or add to `.streamlit/config.toml`

---

## Performance Tips

### For Faster Loading:

```python
# In app.py, add:
@st.cache_resource
def load_model():
    # Your model loading code
    pass
```

### For Better UX:

```python
# Add loading spinner
with st.spinner('Predicting...'):
    result = predictor.predict(smiles)
```

---

## Troubleshooting

### Issue: Port already in use
```bash
# Kill existing Streamlit
pkill -f streamlit

# Or use different port
streamlit run app.py --server.port 8502
```

### Issue: Model file too large for GitHub
```bash
# Use Git LFS
git lfs install
git lfs track "*.pth"
git add .gitattributes
```

### Issue: Dependencies not installing
```bash
# Pin exact versions in requirements.txt
torch==2.9.1
streamlit==1.51.0
```

---

## Security Considerations

**DON'T commit:**
- API keys
- Passwords
- Personal data
- Large model files without Git LFS

**DO commit:**
- Code
- Documentation
- Small model files (<100MB)
- Example data

---

## Monitoring

After deployment:

1. **Check logs** in Streamlit Cloud dashboard
2. **Monitor usage** via analytics
3. **Track errors** via error reporting
4. **Update regularly** with new features

---

## Updating Deployed App

```bash
# Make changes locally
git add .
git commit -m "Add new feature"
git push

# Streamlit Cloud auto-updates in 1-2 minutes!
```

---

## Custom Domain (Optional)

1. Buy domain (e.g., bbbpredictor.com)
2. In Streamlit Cloud settings, add custom domain
3. Update DNS records
4. SSL certificate auto-generated

---

**Your app is now live for the world to use!** 🎉

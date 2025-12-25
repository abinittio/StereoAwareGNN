# Contributing to BBB Permeability Predictor

Thank you for your interest in contributing to the BBB Permeability Predictor project!

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment (OS, Python version, package versions)

### Suggesting Enhancements

We welcome feature suggestions! Please open an issue with:
- Clear description of the feature
- Use case and benefits
- Any implementation ideas

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Add tests if applicable
5. Ensure code follows existing style
6. Commit with clear messages (`git commit -m 'Add AmazingFeature'`)
7. Push to your branch (`git push origin feature/AmazingFeature`)
8. Open a Pull Request

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings to functions and classes
- Comment complex logic

### Testing

- Test your changes locally before submitting
- Ensure the model still loads and predicts correctly
- Test the web interface if you modified it

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/BBB-Predictor.git
cd BBB-Predictor

# Install dependencies
pip install -r requirements.txt

# Run tests
python train_gnn.py  # Verify model training works
streamlit run app.py  # Verify web interface works
```

## Areas for Contribution

- **Dataset Expansion**: Add more validated BBB permeability data
- **Model Improvements**: Experiment with new architectures
- **Visualizations**: Enhance charts and molecular displays
- **Documentation**: Improve guides and tutorials
- **Performance**: Optimize inference speed
- **Features**: Add batch processing, uncertainty quantification, etc.

## Questions?

Open an issue or reach out to the maintainers.

Thank you for contributing!

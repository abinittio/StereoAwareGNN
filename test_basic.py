"""
BBB System Quick Test
Tests basic functionality of all components
"""

import sys
import traceback
import warnings
warnings.filterwarnings('ignore')

def test_dependencies():
    """Check if all required packages are installed"""
    print("📦 Checking dependencies...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('rdkit', 'RDKit'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy')
    ]
    
    missing_packages = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name} installed")
        except ImportError:
            print(f"❌ {name} missing")
            missing_packages.append(name)
    
    return len(missing_packages) == 0

def main():
    print("🧬 BBB System - Quick Test")
    print("=" * 40)
    
    if test_dependencies():
        print("\n🎉 All dependencies found!")
        print("✅ Ready to test your BBB system")
    else:
        print("\n❌ Some dependencies missing")
        print("Install with: pip install torch rdkit-pypi pandas numpy")

if __name__ == "__main__":
    main()
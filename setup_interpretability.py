"""
SETUP SCRIPT FOR STARTUP2VEC INTERPRETABILITY
Quick setup and verification before running interpretability analysis
"""

import os
import sys
import torch
from pathlib import Path

def check_environment():
    """Check if environment is ready for interpretability analysis"""
    
    print("🔍 CHECKING ENVIRONMENT FOR INTERPRETABILITY ANALYSIS")
    print("=" * 60)
    
    issues = []
    
    # 1. Check Python packages
    print("\n📦 Checking Python packages...")
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy', 
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'sklearn': 'Scikit-learn',
        'pytorch_lightning': 'PyTorch Lightning'
    }
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - MISSING")
            issues.append(f"Install {name}: pip install {package}")
    
    # Optional packages
    optional_packages = {
        'umap': 'UMAP (for better visualizations)',
        'plotly': 'Plotly (for 3D plots)'
    }
    
    print("\n📦 Checking optional packages...")
    for package, name in optional_packages.items():
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ⚠️ {name} - Optional but recommended")
            print(f"     Install with: pip install {package}")
    
    # 2. Check CUDA
    print(f"\n�� GPU Status:")
    if torch.cuda.is_available():
        print(f"  ✅ CUDA available")
        print(f"  🎯 GPU: {torch.cuda.get_device_name(0)}")
        print(f"  💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"  ⚠️ CUDA not available - will use CPU (slower)")
    
    # 3. Check file structure
    print(f"\n📁 Checking file structure...")
    
    # Required files
    required_files = {
        'step_4_create_datamodule.py': 'Datamodule script',
        'src/models/survival_model.py': 'Survival model',
        'transformer/transformer.py': 'Transformer architecture'
    }
    
    for file_path, description in required_files.items():
        if os.path.exists(file_path):
            print(f"  ✅ {description}: {file_path}")
        else:
            print(f"  ❌ {description}: {file_path} - MISSING")
            issues.append(f"Missing required file: {file_path}")
    
    # 4. Check checkpoint
    print(f"\n🏁 Checking finetuned checkpoint...")
    checkpoint_path = "survival_checkpoints/fixed-survival-prediction/best-epoch=01-val/auc=0.6709.ckpt"
    
    if os.path.exists(checkpoint_path):
        print(f"  ✅ Best checkpoint found: {checkpoint_path}")
        # Check checkpoint size
        size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f"  📊 Checkpoint size: {size_mb:.1f} MB")
    else:
        print(f"  ❌ Best checkpoint not found: {checkpoint_path}")
        
        # Look for alternative checkpoints
        alt_dirs = [
            "survival_checkpoints/",
            "checkpoints/",
            "."
        ]
        
        found_alt = False
        for alt_dir in alt_dirs:
            if os.path.exists(alt_dir):
                ckpt_files = [f for f in os.listdir(alt_dir) if f.endswith('.ckpt')]
                if ckpt_files:
                    print(f"  📁 Found alternative checkpoints in {alt_dir}:")
                    for ckpt in ckpt_files[:3]:  # Show first 3
                        print(f"     📄 {ckpt}")
                    found_alt = True
                    break
        
        if not found_alt:
            issues.append("No checkpoint file found - you need to run finetuning first")
    
    # 5. Check data directories
    print(f"\n📊 Checking data directories...")
    data_dirs = [
        'data/cleaned/cleaned_startup/',
        'data/processed_startup/'
    ]
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            files = os.listdir(data_dir)
            print(f"  ✅ {data_dir} ({len(files)} files)")
        else:
            print(f"  ⚠️ {data_dir} - not found")
    
    # 6. Summary
    print(f"\n📋 ENVIRONMENT CHECK SUMMARY")
    print("=" * 30)
    
    if not issues:
        print("✅ ALL CHECKS PASSED!")
        print("🚀 Ready to run interpretability analysis")
        return True
    else:
        print(f"❌ Found {len(issues)} issues:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print("\n🔧 Please fix these issues before running interpretability analysis")
        return False

def create_directory_structure():
    """Create necessary directories"""
    
    print("\n📁 Creating directory structure...")
    
    directories = [
        "interpretability_results",
        "interpretability_results/plots",
        "interpretability_results/data"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  📂 {directory}")
    
    print("✅ Directory structure created")

def print_quick_start_guide():
    """Print quick start instructions"""
    
    print(f"\n🚀 QUICK START GUIDE")
    print("=" * 20)
    print()
    print("1️⃣ Extract data from your finetuned model:")
    print("   python extract_startup2vec_data.py")
    print()
    print("2️⃣ Run interpretability analysis:")
    print("   python run_interpretability_analysis.py")
    print()
    print("3️⃣ View results:")
    print("   - Plots will be displayed during analysis")
    print("   - Summary saved to interpretability_results/analysis_summary.txt")
    print("   - Detailed data in interpretability_results/")
    print()
    print("📚 What you'll get:")
    print("   • Performance audit across startup characteristics")
    print("   • Event type contribution analysis")
    print("   • 2D/3D visualization of startup embedding space")
    print("   • Concept influence analysis (TCAV)")
    print()
    print("⚡ Expected runtime: 5-15 minutes")

def test_model_loading():
    """Test if we can load the model"""
    
    print(f"\n🧪 TESTING MODEL LOADING...")
    
    try:
        # Try to import the model classes
        from src.models.survival_model import StartupSurvivalModel
        print("  ✅ StartupSurvivalModel imported successfully")
        
        from transformer.transformer import Transformer
        print("  ✅ Transformer imported successfully")
        
        # Try to import datamodule
        import importlib.util
        spec = importlib.util.spec_from_file_location("step_4_create_datamodule", "step_4_create_datamodule.py")
        if spec is not None:
            print("  ✅ step_4_create_datamodule.py found")
        else:
            print("  ❌ step_4_create_datamodule.py not found")
            return False
        
        # Test checkpoint loading (without actually loading the full model)
        checkpoint_path = "survival_checkpoints/fixed-survival-prediction/best-epoch=01-val/auc=0.6709.ckpt"
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                print("  ✅ Checkpoint loads successfully")
                print(f"     Keys: {list(checkpoint.keys())}")
                return True
            except Exception as e:
                print(f"  ❌ Error loading checkpoint: {e}")
                return False
        else:
            print(f"  ⚠️ Checkpoint not found at expected location")
            return False
            
    except Exception as e:
        print(f"  ❌ Import error: {e}")
        return False

def main():
    """Main setup function"""
    
    print("🔧 STARTUP2VEC INTERPRETABILITY SETUP")
    print("=" * 50)
    
    # Check environment
    env_ok = check_environment()
    
    # Test model loading
    if env_ok:
        print()
        model_ok = test_model_loading()
    else:
        model_ok = False
    
    # Create directories
    create_directory_structure()
    
    # Print status
    print(f"\n🎯 SETUP STATUS")
    print("=" * 15)
    print(f"Environment: {'✅ Ready' if env_ok else '❌ Issues found'}")
    print(f"Model loading: {'✅ Working' if model_ok else '❌ Issues found'}")
    
    if env_ok and model_ok:
        print(f"\n🎉 SETUP COMPLETE!")
        print_quick_start_guide()
        return 0
    else:
        print(f"\n🔧 SETUP INCOMPLETE")
        print("Please fix the issues above before proceeding.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
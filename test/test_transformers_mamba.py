#!/usr/bin/env python3
"""
Simple test for environment setup
"""

def test_basic_setup():
    """Test basic PyTorch and GPU setup"""
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except Exception as e:
        print(f"❌ PyTorch error: {e}")

def test_transformers():
    """Test transformers installation"""
    try:
        import transformers
        print(f"✅ Transformers version: {transformers.__version__}")
        
        # Test if Mamba is available in transformers
        try:
            from transformers import MambaConfig, MambaForCausalLM
            print("✅ Mamba classes available in transformers")
            return True
        except ImportError:
            print("❌ Mamba not available in transformers")
            print("Try: pip install transformers>=4.39.0")
            return False
            
    except ImportError:
        print("❌ Transformers not installed")
        print("Try: pip install transformers")
        return False

def test_other_packages():
    """Test other required packages"""
    packages = ['datasets', 'accelerate', 'tokenizers', 'huggingface_hub']
    
    for package in packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package}: {version}")
        except ImportError:
            print(f"❌ {package} not installed")

if __name__ == "__main__":
    print("🔍 Testing Environment Setup")
    print("=" * 40)
    
    test_basic_setup()
    print()
    
    mamba_available = test_transformers()
    print()
    
    test_other_packages()
    print()
    
    if mamba_available:
        print("🎉 Environment looks good! Ready for Mamba development.")
    else:
        print("⚠️  Install missing packages first.") 
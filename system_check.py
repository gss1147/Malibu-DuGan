# system_check.py - UPDATED
import sys
import os
import subprocess
import ctypes

def check_system():
    print("=== MALIBU DUGAN AI SYSTEM DIAGNOSIS ===")
    
    # Check Python
    print(f"Python: {sys.version}")
    
    # Check DLL loading
    try:
        ctypes.windll.kernel32.GetModuleHandleW(None)
        print("✓ Basic DLL loading: OK")
    except Exception as e:
        print(f"✗ Basic DLL loading: FAILED - {e}")
    
    # Check NumPy first
    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
        
        # Test basic NumPy operations
        arr = np.array([1, 2, 3])
        if arr.sum() == 6:
            print("✓ NumPy operations: OK")
        else:
            print("✗ NumPy operations: FAILED")
            
    except ImportError as e:
        print(f"✗ NumPy: Not installed - {e}")
    except Exception as e:
        print(f"✗ NumPy: Failed to initialize - {e}")
    
    # Check PyTorch with NumPy compatibility
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        
        # Test tensor operations with NumPy interop
        x = torch.tensor([1.0, 2.0, 3.0])
        y = x * 2
        print("✓ PyTorch tensor operations: OK")
        
        # Test PyTorch-NumPy interoperability
        numpy_array = x.numpy()
        if len(numpy_array) == 3:
            print("✓ PyTorch-NumPy interop: OK")
        else:
            print("✗ PyTorch-NumPy interop: FAILED")
        
        # Check CUDA
        if torch.cuda.is_available():
            print("✓ CUDA: Available")
        else:
            print("ℹ CUDA: Not available (using CPU)")
            
    except ImportError as e:
        print(f"✗ PyTorch: Not installed - {e}")
    except Exception as e:
        print(f"✗ PyTorch: Failed to initialize - {e}")
    
    # Check other dependencies
    deps = ['sklearn', 'sqlite3', 'yaml', 'PyQt5']
    for dep in deps:
        try:
            if dep == 'sklearn':
                from sklearn.feature_extraction.text import TfidfVectorizer
                print("✓ sklearn: OK")
            else:
                __import__(dep)
                print(f"✓ {dep}: OK")
        except ImportError as e:
            print(f"✗ {dep}: Missing - {e}")
        except Exception as e:
            print(f"✗ {dep}: Failed to initialize - {e}")
    
    # Check for known compatibility issues
    print("\n=== COMPATIBILITY CHECK ===")
    try:
        import numpy as np
        import torch
        numpy_version = tuple(map(int, np.__version__.split('.')))
        torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
        
        if numpy_version[0] >= 2 and torch_version < (2, 3):
            print("⚠️ WARNING: NumPy 2.x may not be fully compatible with PyTorch < 2.3.0")
            print("   Consider: pip install numpy==1.24.3")
        else:
            print("✓ NumPy and PyTorch versions are compatible")
            
    except Exception as e:
        print(f"⚠️ Compatibility check failed: {e}")
    
    print("=== DIAGNOSIS COMPLETE ===")

if __name__ == "__main__":
    check_system()
    input("Press Enter to close...")
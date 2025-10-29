#!/usr/bin/env python3
"""
Setup script for Smart Investment Planner Backend
"""
import subprocess
import sys
import os

def install_requirements():
    """Install required Python packages"""
    print("Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        return False

def train_ml_model():
    """Train the ML model if it doesn't exist"""
    model_path = "ml_model.pkl"
    if not os.path.exists(model_path):
        print("Training ML model...")
        try:
            from ml_train import train_and_save
            train_and_save(n_rows=50000)  # Smaller dataset for faster training
            print("ML model trained successfully!")
        except Exception as e:
            print(f"Failed to train ML model: {e}")
            return False
    else:
        print("ML model already exists!")
    return True

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    required_modules = [
        'flask', 'flask_cors', 'dotenv', 'requests', 'tinydb',
        'pandas', 'numpy', 'sklearn', 'joblib', 'yfinance', 'plotly'
    ]
    
    failed_imports = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"OK: {module}")
        except ImportError:
            print(f"FAIL: {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\nFailed to import: {', '.join(failed_imports)}")
        return False
    else:
        print("\nAll imports successful!")
        return True

def main():
    print("Setting up Smart Investment Planner Backend...\n")
    
    # Step 1: Install dependencies
    if not install_requirements():
        print("Setup failed at dependency installation")
        return False
    
    # Step 2: Test imports
    if not test_imports():
        print("Setup failed at import testing")
        return False
    
    # Step 3: Train ML model
    if not train_ml_model():
        print("Setup failed at ML model training")
        return False
    
    print("\nSetup completed successfully!")
    print("\nTo run the backend:")
    print("  python app.py")
    print("\nTo run the frontend:")
    print("  cd ../frontend && npm start")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

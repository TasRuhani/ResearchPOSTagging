#!/usr/bin/env python3
"""
Dependency installation script for the POS Tagging project.
This script checks for and installs all missing libraries.
"""

import sys
import subprocess
import os

required_libs = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "sklearn": "scikit-learn",
    "torch": "torch",
    "torchcrf": "pytorch-crf",
    "tqdm": "tqdm",
    "transformers": "transformers",
    "hmmlearn": "hmmlearn"
}

def check_and_install():
    print("Checking dependencies...")
    missing = []
    for module_name, pip_name in required_libs.items():
        try:
            __import__(module_name)
            print(f"  [ OK ] {module_name} is installed")
        except ImportError:
            print(f"  [MISSING] {module_name} is missing (requires {pip_name})")
            missing.append(pip_name)

    if not missing:
        print("\nAll core dependencies are already satisfied.")
        return

    print(f"\nMissing packages detected: {missing}")
    
    # Try to use requirements.txt if present
    req_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    success = False
    
    if os.path.exists(req_path):
        print(f"Found requirements.txt at {req_path}. Trying to install from it...")
        try:
            cmd = [sys.executable, "-m", "pip", "install", "-r", req_path]
            subprocess.check_call(cmd)
            success = True
            print("\nAll dependencies installed successfully from requirements.txt!")
        except Exception as e:
            print(f"\nWarning: Installation from requirements.txt failed: {e}")
            print("Attempting to install only the missing packages directly...")

    if not success:
        print("\nAttempting to install missing packages individually...")
        failed_packages = []
        for pkg in missing:
            cmd = [sys.executable, "-m", "pip", "install", pkg]
            print(f"Running: {' '.join(cmd)}")
            try:
                subprocess.check_call(cmd)
                print(f"Successfully installed {pkg}")
            except Exception as e:
                print(f"Failed to install {pkg}: {e}")
                failed_packages.append(pkg)
        
        if failed_packages:
            print(f"\nSome packages failed to install: {failed_packages}")
            print("For packages requiring compilation (like hmmlearn), please ensure you have Microsoft C++ Build Tools installed, or use a precompiled wheel.")
        else:
            print("\nAll missing packages installed successfully!")

if __name__ == "__main__":
    check_and_install()

#!/bin/bash

# Function to check command existence
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

PYTHON=python3.10

# Check if Python 3 is available
if command_exists $PYTHON; then
    echo "✅ Python3 is installed: $(python3 --version)"
else
    echo "❌ Python3 is not installed."
    exit 1
fi

# Get Python version
PYTHON_VERSION=$($PYTHON --version 2>/dev/null | awk '{print $2}')
# Extract major and minor version numbers
MAJOR_MINOR=$(echo "$PYTHON_VERSION" | awk -F. '{print $1"."$2}')
# Check if the version is 3.10.x or 3.11.x
if [[ "$MAJOR_MINOR" == "3.10" ]]; then
    echo "✅ Python version is $PYTHON_VERSION."
else
    echo "❌ Python version is $PYTHON_VERSION, but 3.10 is required. If you already installed Python 3.10 either edit the PYTHON variable in this script to point to the right binary or use the manual installation."
    exit 1
fi

echo "✅ All checks passed! Trying to install dependencies..."

if [ -d ".venv" ]; then
  echo "Warning: .venv already exists! This script should only be run once."
else
  $PYTHON -m venv .venv                                    
  source .venv/bin/activate
  $PYTHON -m pip install --upgrade pip
  $PYTHON -m pip install -r requirements.txt
fi

$PYTHON 01_preprocess.py
$PYTHON 02_analyze.py
$PYTHON 03_fit_csindy.py
$PYTHON 05_plot_matrix.py
$PYTHON 05_plot_union_fits.py

echo "✅ Reproduction succesful."


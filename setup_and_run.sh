#!/bin/bash

# Print current directory
echo "Current directory: $(pwd)"

# Check if required files exist
echo "Checking required files..."
for file in "sample_qllm.py" "run_quantum_llm.py" "requirements.txt"
do
    if [ -f "$file" ]; then
        echo "✓ Found $file"
    else
        echo "✗ Missing $file"
        exit 1
    fi
done

# Create virtual environment
python -m venv quantum_env

# Activate virtual environment
source quantum_env/bin/activate  # On Windows, use: quantum_env\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Create directories
mkdir -p quantum_model_checkpoints

# Run the model
PYTHONPATH="$(pwd):$PYTHONPATH" python run_quantum_llm.py 
@echo off

:: Print current directory
echo Current directory: %CD%

:: Check if required files exist
echo Checking required files...
for %%F in (sample_qllm.py run_quantum_llm.py requirements.txt) do (
    if exist %%F (
        echo ✓ Found %%F
    ) else (
        echo ✗ Missing %%F
        exit /b 1
    )
)

:: Create virtual environment
python -m venv quantum_env

:: Activate virtual environment
call quantum_env\Scripts\activate

:: Install requirements
pip install -r requirements.txt

:: Create directories
mkdir quantum_model_checkpoints 2>nul

:: Run the model
set PYTHONPATH=%CD%;%PYTHONPATH%
python run_quantum_llm.py 
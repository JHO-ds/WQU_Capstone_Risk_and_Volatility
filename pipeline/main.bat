@echo off

set repo_directory=C:\Users\hp\Documents\WQU\Capstone\Risk_and_Volatility_Project
set model_training_path=%repo_directory%\pipeline\generate_models.py
set model_eval_path=%repo_directory%\pipeline\generate_model_evaluation.py
set python_local_path=C:\Users\hp\anaconda3\python.exe

cd %repo_directory%

echo "Training Models"
%python_local_path% %model_training_path%

echo "Evaluating Models"
%python_local_path% %model_eval_path%

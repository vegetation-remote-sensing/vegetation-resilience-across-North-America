# Time Series Analysis and Machine Learning

This repository contains Python and MATLAB scripts for time series analysis and machine learning model development. It includes STL (Seasonal-Trend decomposition using LOESS) decomposition, 1-lag autocorrelation (AR1) analysis, XGBoost model training with hyperparameter tuning, SHAP (SHapley Additive exPlanations) model interpretation, the Theil–Sen slope estimator, and the Mann–Kendall test.

The main functionalities include:

- STL (Seasonal-Trend decomposition using LOESS) decomposition
- 1-lag autocorrelation (AR1) analysis
- XGBoost model training with hyperparameter tuning
- SHAP (SHapley Additive exPlanations) for model interpretation
- Theil–Sen slope estimation
- Mann–Kendall trend testing


## Project Structure

├── STL_decomposition.py  
├── 1-lag autocorrelation analysis.py  
├── Sen_MK_AR1.m  
├── example_run.m  
├── XGBoost+SHAP+TAC.py  
├── XGBoost+SHAP+deltaTAC.py  
├── input/  
└── output/  


## File Description

### STL_decomposition.py
Performs STL decomposition for time series data.

### 1-lag autocorrelation analysis.py
Conducts 1-lag autocorrelation (AR1) analysis.

### Sen_MK_AR1.m
Implements the Theil–Sen slope estimator and Mann–Kendall trend test.

### example_run.m
Provides an example workflow for running the MATLAB-based analysis.

### XGBoost+SHAP+TAC.py
Trains an XGBoost model and performs SHAP-based feature importance analysis.

### XGBoost+SHAP+deltaTAC.py
Trains an XGBoost model and performs SHAP-based feature importance analysis for deltaTAC-related analysis.

### input/
Directory for input datasets.

### output/
Directory for generated results and outputs.


## Requirements

The project uses both Python and MATLAB. Required Python packages may include:
numpy
pandas
xgboost
shap
statsmodels
scikit-learn

MATLAB is required for running the .m scripts.

"""
Shared parameters and constants for Glucovision project
"""

# Patient IDs
PATIENTS_D1NAMO = ['001', '002', '004', '006', '007', '008']
PATIENTS_AZT1D = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

# Prediction horizons (in 5-minute intervals)
PREDICTION_HORIZONS = [6, 9, 12, 18, 24]  # 30min, 45min, 60min, 90min, 120min
PH_COLUMNS = [f'glucose_{h}' for h in PREDICTION_HORIZONS]

# Feature sets for optimization
OPTIMIZATION_FEATURES_D1NAMO = ['simple_sugars', 'complex_sugars', 'proteins', 'fats', 'dietary_fibers', 'insulin']
OPTIMIZATION_FEATURES_AZT1D = ['carbohydrates', 'insulin']
OPTIMIZATION_FEATURES_BASELINE = ['insulin']  # For baseline models

# LightGBM parameters
LGB_PARAMS = {
    'random_state': 42,
    'deterministic': True,
    'num_threads': 1,
    'verbosity': -1,
    'subsample': 0.8,
    'early_stopping_rounds': 10,
    'reg_lambda': 20,
    'max_depth': 3,
}

# Common feature sets to remove during prediction
FEATURES_TO_REMOVE_D1NAMO = ['datetime', 'hour', 'patient_id'] + PH_COLUMNS
FEATURES_TO_REMOVE_AZT1D = ['datetime', 'hour', 'patient'] + PH_COLUMNS

# Default prediction horizon for analysis (60 minutes)
DEFAULT_PREDICTION_HORIZON = 12

# Glucose conversion factor (mmol/L to mg/dL)
GLUCOSE_CONVERSION_FACTOR = 18.0182

# AZT1D domain knowledge Bezier parameters
AZT1D_BEZIER_PARAMS = {
    'carbohydrates': [0.0, 0.0, 0.5, 1.0, 1.5, 0.0, 3.0, 0.0],  # Peak at 1.5 hours
    'insulin': [0.0, 0.0, 0.25, 1.0, 0.75, 0.0, 2.0, 0.0]      # Peak at 0.75 hours
}

# File paths
D1NAMO_DATA_PATH = "../diabetes_subset_pictures-glucose-food-insulin"
AZT1D_DATA_PATH = "../AZT1D 2025/CGM Records"
FOOD_DATA_PATH = "../food_data/pixtral-large-latest"
RESULTS_PATH = "../results"

# AZT1D column mappings
AZT1D_COLUMNS = {
    'datetime': 'EventDateTime',
    'glucose': 'CGM',
    'carbohydrates': 'CarbSize',
    'insulin': 'TotalBolusInsulinDelivered'
}
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings

from params import *
from processing_functions import *

warnings.filterwarnings('ignore')

def evaluate_generic_optimization(dataset='d1namo'):
    """Evaluate approach 1: Generic optimization using all patient data at once, no monotonic constraints"""
    print(f"Evaluating {dataset.upper()}: Generic Optimization (All Patients)")
    
    # Check if results already exist
    result_file = f'results/ablation_generic_{dataset}_ph24.csv'
    if os.path.exists(result_file):
        df = pd.read_csv(result_file)
        mean_rmse = df['RMSE'].mean()
        std_rmse = df['RMSE'].std()
        print(f"{dataset.upper()} Generic Optimization (Loaded): {mean_rmse:.2f} ± {std_rmse:.2f}")
        return mean_rmse, std_rmse
    
    # Load data and setup (same as main scripts)
    if dataset == 'd1namo':
        patients = PATIENTS_D1NAMO
        opt_features = OPTIMIZATION_FEATURES_D1NAMO
        fast_features = FAST_FEATURES
        features_to_remove = FEATURES_TO_REMOVE_D1NAMO + [f'glucose_{h}' for h in PREDICTION_HORIZONS] + ['patient_id']
        
        # Cache patient data (same as d1namo.py)
        patient_to_data = {}
        for patient in patients:
            g_df, c_df = get_d1namo_data(patient)
            patient_to_data[patient] = (g_df, c_df)
        
        # Collect all training data from all patients for generic optimization
        all_train_data = []
        for p in patients:
            g_df, c_df = patient_to_data[p]
            train_days = g_df['datetime'].dt.day.unique()[:3]
            g_train = g_df[g_df['datetime'].dt.day.isin(train_days)]
            c_train = c_df[c_df['datetime'].dt.day.isin(train_days)]
            all_train_data.append((g_train, c_train))
    else:  # azt1d
        patients = PATIENTS_AZT1D[:5]  # First 5 for ablation
        opt_features = OPTIMIZATION_FEATURES_AZT1D
        fast_features = FAST_FEATURES
        features_to_remove = FEATURES_TO_REMOVE_AZT1D + [f'glucose_{h}' for h in PREDICTION_HORIZONS] + ['patient_id']
        
        # Build training data (same as azt1d.py)
        all_patients = []
        for patient in patients:
            file_path = f"{AZT1D_DATA_PATH}/Subject {patient}/Subject {patient}.csv"
            df = pd.read_csv(file_path)
            df['patient'] = patient
            df['datetime'] = pd.to_datetime(df[AZT1D_COLUMNS['datetime']])
            df['glucose'] = df[AZT1D_COLUMNS['glucose']].fillna(0)
            df['carbohydrates'] = df[AZT1D_COLUMNS['carbohydrates']].fillna(0)
            df['insulin'] = df[AZT1D_COLUMNS['insulin']].fillna(0)
            df['correction'] = df[AZT1D_COLUMNS['correction']].fillna(0)
            df['hour'] = df['datetime'].dt.hour
            df['time'] = df['hour'] + df['datetime'].dt.minute / 60
            df = df[['patient', 'datetime', 'glucose', 'carbohydrates', 'insulin', 'correction', 'hour', 'time']].copy()
            df = df.sort_values('datetime').reset_index(drop=True)
            for horizon in PREDICTION_HORIZONS:
                df[f'glucose_{horizon}'] = df['glucose'].shift(-horizon) - df['glucose']
            df['glucose_change'] = df['glucose'] - df['glucose'].shift(1)
            df['glucose_change_projected'] = df['glucose_change'].rolling(6, min_periods=6).apply(lambda window: get_projected_value(window, 6))
            df['glucose_projected'] = df['glucose'].rolling(6, min_periods=6).apply(lambda window: get_projected_value(window, 6))
            df = df.dropna(subset=[f'glucose_24'])
            all_patients.append(df)
        
        # For AZT1D, we don't do generic optimization - each patient uses their own data only
        # This is consistent with azt1d.py approach
        all_train_data = []  # Will be populated per patient during evaluation
    
    # Check if generic parameters already exist, otherwise optimize
    if dataset == 'd1namo':
        param_file = f'results/bezier_params/{dataset}_generic_all_patients_bezier_params.json'
        if os.path.exists(param_file):
            with open(param_file, 'r') as f:
                params = json.load(f)
            print(f"Loaded existing generic parameters for {dataset}")
        else:
            # Generic optimization using all patient data at once
            try:
                params = optimize_params(f'{dataset}_generic_all_patients', 
                                       opt_features, fast_features, all_train_data, features_to_remove, 
                                       prediction_horizon=DEFAULT_PREDICTION_HORIZON, n_trials=N_TRIALS)
            except Exception as e:
                print(f"Generic optimization failed: {e}")
                return float('nan'), float('nan')
    else:  # azt1d - use generic parameters for all patients
        # Load or create generic parameters optimized on all patients combined
        param_file = f'results/bezier_params/{dataset}_generic_all_patients_bezier_params.json'
        if os.path.exists(param_file):
            with open(param_file, 'r') as f:
                params = json.load(f)
            print(f"Loaded existing generic parameters for {dataset}")
        else:
            # Create generic optimization using all patients' data combined
            print("Creating generic parameters using all patients' data")
            all_train_data = []
        for df in all_patients:
                first_14_dates = sorted(df['datetime'].dt.normalize().unique())[:14]
                g_train = df[df['datetime'].dt.normalize().isin(first_14_dates)].copy()
                c_train = g_train[['datetime', 'carbohydrates', 'insulin', 'correction']].copy()
                all_train_data.append((g_train, c_train))
            
            try:
                params = optimize_params(f'{dataset}_generic_all_patients', 
                                       opt_features, fast_features, all_train_data, features_to_remove, 
                                       prediction_horizon=DEFAULT_PREDICTION_HORIZON, n_trials=N_TRIALS)
                
                # Save generic parameters
                os.makedirs('results/bezier_params', exist_ok=True)
                with open(param_file, 'w') as f:
                    json.dump(params, f, indent=2)
                print("Saved generic Bezier parameters")
            except Exception as e:
                print(f"Generic optimization failed: {e}")
                return float('nan'), float('nan')

    results = []
    prediction_horizon = 24  # 120 minutes
    target_feature = f'glucose_{prediction_horizon}'
    
    # Build PH-aligned datasets per patient (same as main scripts)
    frames_bezier = []
    for p in patients:
        if dataset == 'd1namo':
            g_df, c_df = patient_to_data[p]
            # Apply feature engineering with generic parameters
            d3 = add_temporal_features(params, opt_features, g_df, c_df, prediction_horizon)
        else:  # azt1d
            df = next(df for df in all_patients if int(df['patient'].iloc[0]) == p)
            g_df = df.copy()
            c_df = g_df[['datetime', 'carbohydrates', 'insulin', 'correction']].copy()
            
            # For AZT1D Generic approach, use the same generic parameters for all patients
            d3 = add_temporal_features(params, opt_features, g_df, c_df, prediction_horizon)
        
        d3['patient_id'] = f"patient_{p}"
        frames_bezier.append(d3)

    all_bezier = pd.concat(frames_bezier, ignore_index=True)

    # Evaluate with stepwise retraining (same as main scripts)
    print("Training and predicting with stepwise retraining")
    for p in patients:
        # Get test data (same as main scripts)
        mask3 = all_bezier['patient_id'] == f"patient_{p}"
        
        if dataset == 'd1namo':
            test_days = all_bezier[mask3]['datetime'].dt.day.unique()
            test3 = all_bezier[mask3 & (all_bezier['datetime'].dt.day >= test_days[3])]
        else:
            all_days = sorted(all_bezier[mask3]['datetime'].dt.day.unique())
            test_days = all_days[14:]  # Test on all days after first 14
            test3 = all_bezier[mask3 & (all_bezier['datetime'].dt.day.isin(test_days))]
        
        for start_idx in range(0, len(test3), STEP_SIZE):
            end_idx = min(start_idx + STEP_SIZE, len(test3))
            batch3 = test3.iloc[start_idx:end_idx]
            
            if len(batch3) == 0:
                continue

            # Training data (same as main scripts)
            if dataset == 'd1namo':
                Xdf3 = pd.concat([
                    all_bezier[mask3 & (all_bezier['datetime'] < batch3['datetime'].min())],
                    all_bezier[~mask3],
                ])
                weights_train = [CURRENT_PATIENT_WEIGHT if Xdf3['patient_id'].iloc[idx] == f"patient_{p}" else 1 for idx in range(len(Xdf3))]
                weights_val = weights_train  # Same weights for validation
            else:
                Xdf3 = all_bezier[mask3 & (all_bezier['datetime'] < batch3['datetime'].min())]
                if len(Xdf3) == 0:
                    continue
                weights_train = [1 for _ in range(len(Xdf3))]  # All same patient, so all weight 1
                weights_val = weights_train

            indices = train_test_split(range(len(Xdf3)), test_size=VALIDATION_SIZE, random_state=RANDOM_SEED)
            weights_train_split = [weights_train[idx] for idx in indices[0]]
            weights_val_split = [weights_val[idx] for idx in indices[1]]

            # Approach 3: Glucose + Bezier features (WITHOUT monotonic constraints)
            feats3 = Xdf3.columns.difference(features_to_remove)
            rmse3 = train_and_predict(Xdf3, indices[0], indices[1], target_feature, feats3, batch3, weights_train_split, weights_val_split, use_monotone=False)
            results.append(rmse3)
    
    mean_rmse = np.mean(results) if results else float('nan')
    std_rmse = np.std(results) if results else float('nan')
    print(f"{dataset.upper()} Generic Optimization: {mean_rmse:.2f} ± {std_rmse:.2f}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results_df = pd.DataFrame(results, columns=['RMSE'])
    results_df.to_csv(result_file, index=False)
    
    return mean_rmse, std_rmse

def evaluate_individual_optimization(dataset='d1namo'):
    """Evaluate approach 2: Individual optimization but no monotonic constraints"""
    print(f"Evaluating {dataset.upper()}: Individual Optimization (No Monotonic)")
    
    # Check if results already exist
    result_file = f'results/ablation_individual_no_monotonic_{dataset}_ph24.csv'
    if os.path.exists(result_file):
        df = pd.read_csv(result_file)
        mean_rmse = df['RMSE'].mean()
        std_rmse = df['RMSE'].std()
        print(f"{dataset.upper()} Individual No Monotonic (Loaded): {mean_rmse:.2f} ± {std_rmse:.2f}")
        return mean_rmse, std_rmse
    
    # Load data and setup (same as main scripts)
    if dataset == 'd1namo':
        patients = PATIENTS_D1NAMO
        opt_features = OPTIMIZATION_FEATURES_D1NAMO
        fast_features = FAST_FEATURES
        features_to_remove = FEATURES_TO_REMOVE_D1NAMO + [f'glucose_{h}' for h in PREDICTION_HORIZONS] + ['patient_id']
        
        # Cache patient data (same as d1namo.py)
        patient_to_data = {}
        for patient in patients:
            g_df, c_df = get_d1namo_data(patient)
            patient_to_data[patient] = (g_df, c_df)
    else:  # azt1d
        patients = PATIENTS_AZT1D[:5]  # First 5 for ablation
        opt_features = OPTIMIZATION_FEATURES_AZT1D
        fast_features = FAST_FEATURES
        features_to_remove = FEATURES_TO_REMOVE_AZT1D + [f'glucose_{h}' for h in PREDICTION_HORIZONS] + ['patient_id']
        
        # Build training data (same as azt1d.py)
        all_patients = []
        for patient in patients:
            file_path = f"{AZT1D_DATA_PATH}/Subject {patient}/Subject {patient}.csv"
            df = pd.read_csv(file_path)
            df['patient'] = patient
            df['datetime'] = pd.to_datetime(df[AZT1D_COLUMNS['datetime']])
            df['glucose'] = df[AZT1D_COLUMNS['glucose']].fillna(0)
            df['carbohydrates'] = df[AZT1D_COLUMNS['carbohydrates']].fillna(0)
            df['insulin'] = df[AZT1D_COLUMNS['insulin']].fillna(0)
            df['correction'] = df[AZT1D_COLUMNS['correction']].fillna(0)
            df['hour'] = df['datetime'].dt.hour
            df['time'] = df['hour'] + df['datetime'].dt.minute / 60
            df = df[['patient', 'datetime', 'glucose', 'carbohydrates', 'insulin', 'correction', 'hour', 'time']].copy()
            df = df.sort_values('datetime').reset_index(drop=True)
            for horizon in PREDICTION_HORIZONS:
                df[f'glucose_{horizon}'] = df['glucose'].shift(-horizon) - df['glucose']
            df['glucose_change'] = df['glucose'] - df['glucose'].shift(1)
            df['glucose_change_projected'] = df['glucose_change'].rolling(6, min_periods=6).apply(lambda window: get_projected_value(window, 6))
            df['glucose_projected'] = df['glucose'].rolling(6, min_periods=6).apply(lambda window: get_projected_value(window, 6))
            df = df.dropna(subset=[f'glucose_24'])
            all_patients.append(df)

    results = []
    prediction_horizon = 24  # 120 minutes
    target_feature = f'glucose_{prediction_horizon}'
    
    # Build PH-aligned datasets per patient (same as main scripts)
    frames_bezier = []
    for p in patients:
        if dataset == 'd1namo':
        g_df, c_df = patient_to_data[p]
        else:
            df = next(df for df in all_patients if int(df['patient'].iloc[0]) == p)
            g_df = df.copy()
            c_df = g_df[['datetime', 'carbohydrates', 'insulin', 'correction']].copy()
        
        # Individual optimization for this patient (same as main scripts)
        train_data = []
        if dataset == 'd1namo':
            train_days = g_df['datetime'].dt.day.unique()[:3]
            g_train = g_df[g_df['datetime'].dt.day.isin(train_days)]
            c_train = c_df[c_df['datetime'].dt.day.isin(train_days)]
            train_data.append((g_train, c_train))
        else:
            first_14_dates = sorted(g_df['datetime'].dt.normalize().unique())[:14]
            g_train = g_df[g_df['datetime'].dt.normalize().isin(first_14_dates)].copy()
            c_train = g_train[['datetime', 'carbohydrates', 'insulin', 'correction']].copy()
            train_data.append((g_train, c_train))
        
        # Load existing individual patient parameters (no need to reoptimize)
        try:
            param_file = f'results/bezier_params/{dataset}_all_patient_bezier_params.json'
            with open(param_file, 'r') as f:
                all_patient_params = json.load(f)
            params = all_patient_params[f'patient_{p}']
        except Exception as e:
            print(f"Failed to load parameters for patient {p}: {e}")
            continue
        
        # Apply feature engineering with individual parameters
        d3 = add_temporal_features(params, opt_features, g_df, c_df, prediction_horizon)
        d3['patient_id'] = f"patient_{p}"
        frames_bezier.append(d3)

    all_bezier = pd.concat(frames_bezier, ignore_index=True)

    # Evaluate with stepwise retraining (same as main scripts)
    print("Training and predicting with stepwise retraining")
    for p in patients:
        # Get test data (same as main scripts)
        mask3 = all_bezier['patient_id'] == f"patient_{p}"
        
        if dataset == 'd1namo':
            test_days = all_bezier[mask3]['datetime'].dt.day.unique()
            test3 = all_bezier[mask3 & (all_bezier['datetime'].dt.day >= test_days[3])]
        else:
            all_days = sorted(all_bezier[mask3]['datetime'].dt.day.unique())
            test_days = all_days[14:]  # Test on all days after first 14
            test3 = all_bezier[mask3 & (all_bezier['datetime'].dt.day.isin(test_days))]
        
        for start_idx in range(0, len(test3), STEP_SIZE):
            end_idx = min(start_idx + STEP_SIZE, len(test3))
            batch3 = test3.iloc[start_idx:end_idx]
            
            if len(batch3) == 0:
                continue

            # Training data (same as main scripts)
            if dataset == 'd1namo':
                Xdf3 = pd.concat([
                    all_bezier[mask3 & (all_bezier['datetime'] < batch3['datetime'].min())],
                    all_bezier[~mask3],
                ])
                weights_train = [CURRENT_PATIENT_WEIGHT if Xdf3['patient_id'].iloc[idx] == f"patient_{p}" else 1 for idx in range(len(Xdf3))]
                weights_val = weights_train  # Same weights for validation
            else:
                Xdf3 = all_bezier[mask3 & (all_bezier['datetime'] < batch3['datetime'].min())]
                if len(Xdf3) == 0:
                    continue
                weights_train = [1 for _ in range(len(Xdf3))]  # All same patient, so all weight 1
                weights_val = weights_train

            indices = train_test_split(range(len(Xdf3)), test_size=VALIDATION_SIZE, random_state=RANDOM_SEED)
            weights_train_split = [weights_train[idx] for idx in indices[0]]
            weights_val_split = [weights_val[idx] for idx in indices[1]]

            # Approach 3: Glucose + Bezier features (WITHOUT monotonic constraints)
            feats3 = Xdf3.columns.difference(features_to_remove)
            rmse3 = train_and_predict(Xdf3, indices[0], indices[1], target_feature, feats3, batch3, weights_train_split, weights_val_split, use_monotone=False)
            results.append(rmse3)
    
    mean_rmse = np.mean(results) if results else float('nan')
    std_rmse = np.std(results) if results else float('nan')
    print(f"{dataset.upper()} Individual Optimization: {mean_rmse:.2f} ± {std_rmse:.2f}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results_df = pd.DataFrame(results, columns=['RMSE'])
    results_df.to_csv(result_file, index=False)
    
    return mean_rmse, std_rmse

def evaluate_with_monotonic_constraints(dataset='d1namo'):
    """Evaluate approach 3: Individual optimization WITH monotonic constraints (using loaded params)"""
    print(f"Evaluating {dataset.upper()}: Individual + Monotonic Constraints")
    
    # Check if results already exist
    result_file = f'results/ablation_individual_with_monotonic_{dataset}_ph24.csv'
    if os.path.exists(result_file):
        df = pd.read_csv(result_file)
        mean_rmse = df['RMSE'].mean()
        std_rmse = df['RMSE'].std()
        print(f"{dataset.upper()} Individual With Monotonic (Loaded): {mean_rmse:.2f} ± {std_rmse:.2f}")
        return mean_rmse, std_rmse
    
    # Load data and setup (same as main scripts)
    if dataset == 'd1namo':
        patients = PATIENTS_D1NAMO
        opt_features = OPTIMIZATION_FEATURES_D1NAMO
        features_to_remove = FEATURES_TO_REMOVE_D1NAMO + [f'glucose_{h}' for h in PREDICTION_HORIZONS] + ['patient_id']
        
        # Cache patient data (same as d1namo.py)
        patient_to_data = {}
        for patient in patients:
            g_df, c_df = get_d1namo_data(patient)
            patient_to_data[patient] = (g_df, c_df)
    else:  # azt1d
        patients = PATIENTS_AZT1D[:5]  # First 5 for ablation
        opt_features = OPTIMIZATION_FEATURES_AZT1D
        features_to_remove = FEATURES_TO_REMOVE_AZT1D + [f'glucose_{h}' for h in PREDICTION_HORIZONS] + ['patient_id']
        
        # Build training data (same as azt1d.py)
        all_patients = []
        for patient in patients:
            file_path = f"{AZT1D_DATA_PATH}/Subject {patient}/Subject {patient}.csv"
            df = pd.read_csv(file_path)
            df['patient'] = patient
            df['datetime'] = pd.to_datetime(df[AZT1D_COLUMNS['datetime']])
            df['glucose'] = df[AZT1D_COLUMNS['glucose']].fillna(0)
            df['carbohydrates'] = df[AZT1D_COLUMNS['carbohydrates']].fillna(0)
            df['insulin'] = df[AZT1D_COLUMNS['insulin']].fillna(0)
            df['correction'] = df[AZT1D_COLUMNS['correction']].fillna(0)
            df['hour'] = df['datetime'].dt.hour
            df['time'] = df['hour'] + df['datetime'].dt.minute / 60
            df = df[['patient', 'datetime', 'glucose', 'carbohydrates', 'insulin', 'correction', 'hour', 'time']].copy()
            df = df.sort_values('datetime').reset_index(drop=True)
            for horizon in PREDICTION_HORIZONS:
                df[f'glucose_{horizon}'] = df['glucose'].shift(-horizon) - df['glucose']
            df['glucose_change'] = df['glucose'] - df['glucose'].shift(1)
            df['glucose_change_projected'] = df['glucose_change'].rolling(6, min_periods=6).apply(lambda window: get_projected_value(window, 6))
            df['glucose_projected'] = df['glucose'].rolling(6, min_periods=6).apply(lambda window: get_projected_value(window, 6))
            df = df.dropna(subset=[f'glucose_24'])
            all_patients.append(df)

    results = []
    prediction_horizon = 24  # 120 minutes
    target_feature = f'glucose_{prediction_horizon}'

    # Build PH-aligned datasets per patient (same as main scripts)
    frames_bezier = []
    for p in patients:
        if dataset == 'd1namo':
        g_df, c_df = patient_to_data[p]
        else:
            df = next(df for df in all_patients if int(df['patient'].iloc[0]) == p)
            g_df = df.copy()
            c_df = g_df[['datetime', 'carbohydrates', 'insulin', 'correction']].copy()
        
        # Load existing individual patient parameters
        try:
            param_file = f'results/bezier_params/{dataset}_all_patient_bezier_params.json'
            with open(param_file, 'r') as f:
                all_patient_params = json.load(f)
            params = all_patient_params[f'patient_{p}']
        except Exception as e:
            print(f"Failed to load parameters for patient {p}: {e}")
            continue
        
        # Apply feature engineering with loaded individual parameters
        d3 = add_temporal_features(params, opt_features, g_df, c_df, prediction_horizon)
        d3['patient_id'] = f"patient_{p}"
        frames_bezier.append(d3)

    all_bezier = pd.concat(frames_bezier, ignore_index=True)

    # Evaluate with stepwise retraining (same as main scripts)
    print("Training and predicting with stepwise retraining")
    for p in patients:
        # Get test data (same as main scripts)
        print(f"Evaluating patient {p}")
        mask3 = all_bezier['patient_id'] == f"patient_{p}"
        
        if dataset == 'd1namo':
            test_days = all_bezier[mask3]['datetime'].dt.day.unique()
            test3 = all_bezier[mask3 & (all_bezier['datetime'].dt.day >= test_days[3])]
        else:
            all_days = sorted(all_bezier[mask3]['datetime'].dt.day.unique())
            test_days = all_days[14:]  # Test on all days after first 14
            test3 = all_bezier[mask3 & (all_bezier['datetime'].dt.day.isin(test_days))]
        
        for start_idx in range(0, len(test3), STEP_SIZE):
            end_idx = min(start_idx + STEP_SIZE, len(test3))
            batch3 = test3.iloc[start_idx:end_idx]
            
            if len(batch3) == 0:
                continue

            # Training data (same as main scripts)
            if dataset == 'd1namo':
                Xdf3 = pd.concat([
                    all_bezier[mask3 & (all_bezier['datetime'] < batch3['datetime'].min())],
                    all_bezier[~mask3],
                ])
                weights_train = [CURRENT_PATIENT_WEIGHT if Xdf3['patient_id'].iloc[idx] == f"patient_{p}" else 1 for idx in range(len(Xdf3))]
                weights_val = weights_train  # Same weights for validation
            else:
                Xdf3 = all_bezier[mask3 & (all_bezier['datetime'] < batch3['datetime'].min())]
                if len(Xdf3) == 0:
                    continue
                weights_train = [1 for _ in range(len(Xdf3))]  # All same patient, so all weight 1
                weights_val = weights_train

            indices = train_test_split(range(len(Xdf3)), test_size=VALIDATION_SIZE, random_state=RANDOM_SEED)
            weights_train_split = [weights_train[idx] for idx in indices[0]]
            weights_val_split = [weights_val[idx] for idx in indices[1]]

            # Approach 3: Glucose + Bezier features (WITH monotonic constraints)
            feats3 = Xdf3.columns.difference(features_to_remove)
            rmse3 = train_and_predict(Xdf3, indices[0], indices[1], target_feature, feats3, batch3, weights_train_split, weights_val_split, use_monotone=True)
            results.append(rmse3)
    
    mean_rmse = np.mean(results) if results else float('nan')
    std_rmse = np.std(results) if results else float('nan')
    print(f"{dataset.upper()} Individual + Monotonic: {mean_rmse:.2f} ± {std_rmse:.2f}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results_df = pd.DataFrame(results, columns=['RMSE'])
    results_df.to_csv(result_file, index=False)
    
    return mean_rmse, std_rmse

# Run the three approaches
approaches = [
    ("Generic", evaluate_generic_optimization),
    ("Individual + no monotonic constraints", evaluate_individual_optimization),
    ("Individual + monotonic constraints", evaluate_with_monotonic_constraints)
]

results = {'D1namo': [], 'AZT1D': []}

for name, approach_func in approaches:
    print(f"\n=== {name} ===")
    d1_mean, d1_std = approach_func('d1namo')
    azt_mean, azt_std = approach_func('azt1d')
    
    results['D1namo'].append((name, d1_mean, d1_std))
    results['AZT1D'].append((name, azt_mean, azt_std))

# Save results
results_df = pd.DataFrame({
    'Approach': [r[0] for r in results['D1namo']],
    'D1namo_RMSE': [r[1] for r in results['D1namo']],
    'D1namo_Std': [r[2] for r in results['D1namo']],
    'AZT1D_RMSE': [r[1] for r in results['AZT1D']],
    'AZT1D_Std': [r[2] for r in results['AZT1D']],
})

os.makedirs('results', exist_ok=True)
results_df.to_csv('results/ablation_study_ph24.csv', index=False)

# Print summary
print("\n=== SUMMARY (Prediction Horizon: 120 minutes) ===")
for i, (name, d1_mean, d1_std) in enumerate(results['D1namo']):
    azt_name, azt_mean, azt_std = results['AZT1D'][i]
    print(f"{name}: D1namo {d1_mean:.2f}±{d1_std:.2f}, AZT1D {azt_mean:.2f}±{azt_std:.2f}")

# Calculate improvements
print("\n=== IMPROVEMENT ANALYSIS ===")
for dataset in ['D1namo', 'AZT1D']:
    dataset_results = results[dataset]
    if len(dataset_results) >= 3:
        generic = dataset_results[0][1]  # Generic optimization
        individual_no_monotonic = dataset_results[1][1]  # Individual optimization without monotonic
        individual_with_monotonic = dataset_results[2][1]  # Individual optimization with monotonic
        
        if not np.isnan(generic) and not np.isnan(individual_with_monotonic):
            generic_improvement = ((individual_with_monotonic - generic) / individual_with_monotonic) * 100
            print(f"{dataset} Generic vs Individual+Monotonic: {generic_improvement:+.1f}%")
        
        if not np.isnan(individual_no_monotonic) and not np.isnan(individual_with_monotonic):
            monotonic_improvement = ((individual_with_monotonic - individual_no_monotonic) / individual_with_monotonic) * 100
            print(f"{dataset} Individual+NoMonotonic vs Individual+Monotonic: {monotonic_improvement:+.1f}%")
        
        if not np.isnan(generic) and not np.isnan(individual_no_monotonic):
            generic_vs_individual = ((individual_no_monotonic - generic) / individual_no_monotonic) * 100
            print(f"{dataset} Generic vs Individual+NoMonotonic: {generic_vs_individual:+.1f}%")

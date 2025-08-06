import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings
import json
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from params import *
from processing_functions import *

warnings.filterwarnings('ignore')

# Script-specific constants
validation_size = 0.2
random_seed = 42

def run_ablation_study():
    """Run ablation study with different configurations"""
    
    # Load optimized parameters from d1namo
    with open('../results/d1namo_bezier_params.json', 'r') as f:
        optimized_params = json.load(f)
    
    # Define domain knowledge parameters (reasonable estimates based on physiology)
    domain_params = {
        'simple_sugars': [0.0, 0.0, 0.5, 1.0, 1.0, 0.8, 2.0, 0.0],     # Fast absorption, peak ~30min
        'complex_sugars': [0.0, 0.0, 1.0, 0.6, 2.0, 0.9, 4.0, 0.0],    # Slower absorption, peak ~1h
        'proteins': [0.0, 0.0, 2.0, 0.3, 4.0, 0.6, 6.0, 0.0],          # Late effect, peak ~2-4h
        'fats': [0.0, 0.0, 1.5, 0.4, 3.0, 0.7, 5.0, 0.0],              # Delayed effect, peak ~1.5-3h
        'dietary_fibers': [0.0, 0.0, 2.0, 0.2, 4.0, 0.4, 6.0, 0.0],    # Slow, minimal effect
        'insulin': [0.0, 0.0, 0.25, 1.0, 0.75, 0.6, 2.0, 0.0]          # Fast acting insulin, peak ~15-45min
    }
    
    print(f"Running Ablation Study for PH={DEFAULT_PREDICTION_HORIZON} (60 minutes)")
    print("=" * 60)
    
    results = []
    
    # Configuration 1: Baseline (no macronutrient features)
    print("1. Running Baseline (no macronutrient features)...")
    baseline_rmse = run_configuration("Baseline", {}, use_weighting=False, use_macronutrients=False)
    results.append(["Baseline (no macronutrients)", baseline_rmse, "N/A", "No"])
    
    # Configuration 2: Domain knowledge parameters, no weighting
    print("2. Running Domain Knowledge Parameters (no weighting)...")
    domain_rmse = run_configuration("Domain Knowledge", domain_params, use_weighting=False, use_macronutrients=True)
    results.append(["Domain knowledge parameters", domain_rmse, "No", "No"])
    
    # Configuration 3: Optimized parameters, no weighting
    print("3. Running Optimized Parameters (no weighting)...")
    optimized_no_weight_rmse = run_configuration("Optimized (no weight)", optimized_params, use_weighting=False, use_macronutrients=True)
    results.append(["Optimized parameters", optimized_no_weight_rmse, "Yes", "No"])
    
    # Configuration 4: Optimized parameters with weighting (full model)
    print("4. Running Optimized Parameters with Weighting (full model)...")
    full_model_rmse = run_configuration("Full Model", optimized_params, use_weighting=True, use_macronutrients=True)
    results.append(["Optimized parameters + weighting", full_model_rmse, "Yes", "Yes"])
    
    # Create results dataframe
    results_df = pd.DataFrame(results, columns=['Configuration', 'RMSE (mg/dL)', 'Optimization', 'Patient Weighting'])
    
    # Calculate improvements
    results_df['Improvement vs Baseline (%)'] = ((baseline_rmse - results_df['RMSE (mg/dL)']) / baseline_rmse * 100).round(1)
    results_df['Improvement vs Baseline (%)'] = results_df['Improvement vs Baseline (%)'].apply(lambda x: f"{x:+.1f}" if x != 0 else "0.0")
    
    print("\nAblation Study Results:")
    print("=" * 80)
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv('../results/ablation_study_results.csv', index=False)
    
    return results_df

def run_configuration(name, params, use_weighting, use_macronutrients):
    """Run a single configuration and return average RMSE"""
    
    # Prepare data for all patients
    all_data_list = []
    for patient in PATIENTS_D1NAMO:
        glucose_data, combined_data = get_d1namo_data(patient)
        if use_macronutrients and params:
            patient_data = add_d1namo_features(params, OPTIMIZATION_FEATURES_D1NAMO, (glucose_data, combined_data))
        else:
            patient_data = glucose_data.copy()
        patient_data['patient_id'] = patient
        all_data_list.append(patient_data)
    
    all_data = pd.concat(all_data_list, ignore_index=True)
    
    # Run evaluation
    results_list = []
    target_feature = f'glucose_{DEFAULT_PREDICTION_HORIZON}'
    
    for patient in PATIENTS_D1NAMO:
        patient_mask = all_data['patient_id'] == patient
        days = all_data[patient_mask]['datetime'].dt.day.unique()[3:]  # Test days
        
        for test_day in days:
            day_hours = all_data[patient_mask & (all_data['datetime'].dt.day == test_day)]['hour'].unique()
            
            for hour in day_hours:
                test = all_data[patient_mask & (all_data['datetime'].dt.day == test_day) & (all_data['hour'] == hour)]
                if len(test) == 0:
                    continue
                    
                # Training data: patient data up to 6 hours before + all other patient data
                X = pd.concat([
                    all_data[patient_mask & (all_data['datetime'].shift(-6) < test['datetime'].min())],
                    all_data[~patient_mask]
                ])
                
                if len(X) == 0:
                    continue
                
                # Remove features and split
                available_features = X.columns.difference(FEATURES_TO_REMOVE_D1NAMO)
                train = X[available_features]
                
                indices = train_test_split(range(len(X)), test_size=validation_size, random_state=random_seed)
                
                if use_weighting:
                    # Apply 10:1 weighting for target patient
                    weights = [np.where(X['patient_id'].values[idx] == patient, 10, 1) for idx in indices]
                    weights_train, weights_val = weights[0], weights[1]
                else:
                    # No weighting
                    weights_train = np.ones(len(indices[0]))
                    weights_val = np.ones(len(indices[1]))
                
                X_train, y_train = train.values[indices[0]], X[target_feature].values[indices[0]]
                X_val, y_val = train.values[indices[1]], X[target_feature].values[indices[1]]
                
                # Train model
                model = lgb.train(
                    LGB_PARAMS,
                    lgb.Dataset(X_train, label=y_train, weight=weights_train),
                    valid_sets=[lgb.Dataset(X_val, label=y_val, weight=weights_val)]
                )
                
                # Predict
                test_features = test[available_features]
                predictions = model.predict(test_features.values)
                ground_truth = test[target_feature].values
                rmse = np.sqrt(mean_squared_error(ground_truth, predictions))
                results_list.append(rmse)
    
    avg_rmse = np.mean(results_list)
    print(f"   {name}: {avg_rmse:.2f} mg/dL (n={len(results_list)} predictions)")
    return avg_rmse

def create_latex_table(results_df):
    """Create LaTeX table for the manuscript"""
    
    latex_table = """\\begin{table}[htbp]
\\centering
\\caption{Ablation Study Results: Component Contribution Analysis}
\\label{tab:ablation_study}
\\begin{tabular}{lccccc}
\\toprule
\\textbf{Configuration} & \\textbf{RMSE} & \\textbf{Optimization} & \\textbf{Patient} & \\textbf{Improvement} \\\\
 & \\textbf{(mg/dL)} & & \\textbf{Weighting} & \\textbf{vs Baseline (\\%)} \\\\
\\midrule
"""
    
    for _, row in results_df.iterrows():
        config = row['Configuration'].replace('_', '\\_')
        rmse = f"{row['RMSE (mg/dL)']:.2f}"
        opt = row['Optimization']
        weight = row['Patient Weighting']
        improvement = row['Improvement vs Baseline (%)']
        
        latex_table += f"{config} & {rmse} & {opt} & {weight} & {improvement} \\\\\n"
    
    latex_table += """\\bottomrule
\\end{tabular}
\\end{table}"""
    
    # Save to file
    with open('../manuscript/tables/ablation_study.tex', 'w') as f:
        f.write(latex_table)
    
    print(f"\nLaTeX table saved to ../manuscript/tables/ablation_study.tex")
    return latex_table

if __name__ == "__main__":
    # Run ablation study
    results_df = run_ablation_study()
    
    # Create LaTeX table
    latex_table = create_latex_table(results_df)
    
    print("\nAblation study completed!")
    print("Results saved to:")
    print("- ../results/ablation_study_results.csv")
    print("- ../manuscript/tables/ablation_study.tex")
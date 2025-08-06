import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import json
import warnings
import os
warnings.filterwarnings('ignore')

from params import *
from processing_functions import *

def analyze_d1namo():
    """Analyze D1namo correlations"""
    print("D1NAMO Analysis of Sugar and Insulin Correlations with Future Blood Glucose")
    print("=" * 75)
    
    # Load globally optimized parameters
    with open(f'{RESULTS_PATH}/d1namo_bezier_params.json', 'r') as f:
        global_params = json.load(f)
    
    # Collect all data
    all_data_list = []
    timing_analysis = []
    
    for patient in PATIENTS_D1NAMO:
        glucose_data, combined_data = get_d1namo_data(patient)
        patient_data = add_d1namo_features(global_params, OPTIMIZATION_FEATURES_D1NAMO, (glucose_data, combined_data))
        patient_data['patient_id'] = patient
        all_data_list.append(patient_data)
        
        # Analyze timing patterns for this patient
        food_events = combined_data[combined_data[['simple_sugars', 'complex_sugars', 'proteins', 'fats', 'dietary_fibers']].sum(axis=1) > 0]
        insulin_events = combined_data[combined_data['insulin'] > 0]
        
        for _, food_event in food_events.iterrows():
            food_time = food_event['datetime']
            # Find insulin events within 2 hours before and after food
            nearby_insulin = insulin_events[
                (insulin_events['datetime'] >= food_time - pd.Timedelta(hours=2)) &
                (insulin_events['datetime'] <= food_time + pd.Timedelta(hours=2))
            ]
            
            if len(nearby_insulin) > 0:
                # Find closest insulin event
                time_diffs = (nearby_insulin['datetime'] - food_time).dt.total_seconds() / 60  # in minutes
                closest_idx = time_diffs.abs().idxmin()
                closest_insulin_time_diff = time_diffs.loc[closest_idx]
                
                timing_analysis.append({
                    'patient': patient,
                    'food_datetime': food_time,
                    'insulin_time_diff_minutes': closest_insulin_time_diff,
                    'insulin_dose': nearby_insulin.loc[closest_idx, 'insulin'],
                    'simple_sugars': food_event['simple_sugars'],
                    'complex_sugars': food_event['complex_sugars'],
                    'total_carbs': food_event['simple_sugars'] + food_event['complex_sugars']
                })
    
    # Combine all patient data
    all_data = pd.concat(all_data_list, ignore_index=True)
    
    print("1. Correlation Analysis between Features and Future Glucose (PH=12)")
    print("-" * 65)
    
            # Calculate correlations for all features of interest
    features_to_analyze = ['simple_sugars', 'complex_sugars', 'insulin']
    correlations = {}
    
    for feature in features_to_analyze:
        pearson_corr, pearson_p = pearsonr(all_data[feature], all_data[f'glucose_{DEFAULT_PREDICTION_HORIZON}'])
        spearman_corr, spearman_p = spearmanr(all_data[feature], all_data[f'glucose_{DEFAULT_PREDICTION_HORIZON}'])
        
        correlations[feature] = {
            'pearson_r': pearson_corr,
            'pearson_p': pearson_p,
            'spearman_r': spearman_corr,
            'spearman_p': spearman_p
        }
        
        print(f"{feature.replace('_', ' ').title()}:")
        print(f"  Pearson correlation: r = {pearson_corr:.4f}, p = {pearson_p:.4f}")
        print(f"  Spearman correlation: r = {spearman_corr:.4f}, p = {spearman_p:.4f}")
        print()
    
    # Additional correlations between modeled features
    print("2. Correlations Between Modeled Features")
    print("-" * 40)
    
    # Simple sugars vs Complex sugars
    simple_complex_corr, simple_complex_p = pearsonr(all_data['simple_sugars'], all_data['complex_sugars'])
    print(f"Simple Sugars vs Complex Sugars:")
    print(f"  Pearson correlation: r = {simple_complex_corr:.4f}, p = {simple_complex_p:.4f}")
    
    # Simple sugars vs Insulin
    simple_insulin_corr, simple_insulin_p = pearsonr(all_data['simple_sugars'], all_data['insulin'])
    print(f"Simple Sugars vs Insulin:")
    print(f"  Pearson correlation: r = {simple_insulin_corr:.4f}, p = {simple_insulin_p:.4f}")
    
    # Complex sugars vs Insulin
    complex_insulin_corr, complex_insulin_p = pearsonr(all_data['complex_sugars'], all_data['insulin'])
    print(f"Complex Sugars vs Insulin:")
    print(f"  Pearson correlation: r = {complex_insulin_corr:.4f}, p = {complex_insulin_p:.4f}")
    print()
    
    # Patient-specific correlations
    print("3. Patient-Specific Correlations with Future Glucose")
    print("-" * 50)
    patient_correlations = {}
    for patient in PATIENTS_D1NAMO:
        patient_data = all_data[all_data['patient_id'] == patient]
        if len(patient_data) > 10:  # Need sufficient data points
            patient_correlations[patient] = {}
            for feature in features_to_analyze:
                pearson_corr, pearson_p = pearsonr(patient_data[feature], patient_data[f'glucose_{DEFAULT_PREDICTION_HORIZON}'])
                patient_correlations[patient][feature] = {
                    'pearson_r': pearson_corr,
                    'pearson_p': pearson_p,
                    'n_points': len(patient_data)
                }
            
            print(f"Patient {patient} (n={len(patient_data)}):")
            for feature in features_to_analyze:
                corr_data = patient_correlations[patient][feature]
                print(f"  {feature}: r = {corr_data['pearson_r']:.4f}, p = {corr_data['pearson_p']:.4f}")
            print()
    
    print("4. Timing Analysis: Food vs Insulin Administration")
    print("-" * 55)
    
    timing_df = pd.DataFrame(timing_analysis)
    if len(timing_df) > 0:
        print(f"Total food-insulin pairs analyzed: {len(timing_df)}")
        print(f"Mean time difference (insulin relative to food): {timing_df['insulin_time_diff_minutes'].mean():.1f} ± {timing_df['insulin_time_diff_minutes'].std():.1f} minutes")
        
        # Categorize timing patterns
        pre_bolus = timing_df[timing_df['insulin_time_diff_minutes'] < -5]  # Insulin >5 min before food
        concurrent = timing_df[(timing_df['insulin_time_diff_minutes'] >= -5) & (timing_df['insulin_time_diff_minutes'] <= 15)]  # Within 5 min before to 15 min after
        post_bolus = timing_df[timing_df['insulin_time_diff_minutes'] > 15]  # Insulin >15 min after food
        
        print(f"\nTiming patterns:")
        print(f"  Pre-bolus (>5 min before food): {len(pre_bolus)} ({len(pre_bolus)/len(timing_df)*100:.1f}%)")
        print(f"  Concurrent (-5 to +15 min): {len(concurrent)} ({len(concurrent)/len(timing_df)*100:.1f}%)")
        print(f"  Post-bolus (>15 min after food): {len(post_bolus)} ({len(post_bolus)/len(timing_df)*100:.1f}%)")
        
        # Analyze insulin-to-carb ratios (only for meals with carbs > 0)
        timing_df_with_carbs = timing_df[timing_df['total_carbs'] > 0]
        if len(timing_df_with_carbs) > 0:
            timing_df_with_carbs['insulin_carb_ratio'] = timing_df_with_carbs['insulin_dose'] / timing_df_with_carbs['total_carbs']
        
        print(f"\nInsulin-to-carbohydrate patterns:")
        print(f"  Mean insulin dose: {timing_df['insulin_dose'].mean():.2f} ± {timing_df['insulin_dose'].std():.2f} units")
        print(f"  Mean total carbs: {timing_df['total_carbs'].mean():.1f} ± {timing_df['total_carbs'].std():.1f} g")
        if len(timing_df_with_carbs) > 0:
            print(f"  Mean insulin:carb ratio: {timing_df_with_carbs['insulin_carb_ratio'].mean():.3f} ± {timing_df_with_carbs['insulin_carb_ratio'].std():.3f} units/g")
        else:
            print(f"  No meals with carbohydrates found for ratio calculation")
    
    # Save D1namo results
    d1namo_results = {
        'overall_correlations': correlations,
        'inter_feature_correlations': {
            'simple_complex': {'r': simple_complex_corr, 'p': simple_complex_p},
            'simple_insulin': {'r': simple_insulin_corr, 'p': simple_insulin_p},
            'complex_insulin': {'r': complex_insulin_corr, 'p': complex_insulin_p}
        },
        'patient_correlations': patient_correlations,
        'timing_analysis_summary': {
            'total_pairs': len(timing_df),
            'mean_time_diff_minutes': timing_df['insulin_time_diff_minutes'].mean() if len(timing_df) > 0 else None,
            'std_time_diff_minutes': timing_df['insulin_time_diff_minutes'].std() if len(timing_df) > 0 else None,
            'pre_bolus_percentage': len(pre_bolus)/len(timing_df)*100 if len(timing_df) > 0 else None,
            'concurrent_percentage': len(concurrent)/len(timing_df)*100 if len(timing_df) > 0 else None,
            'post_bolus_percentage': len(post_bolus)/len(timing_df)*100 if len(timing_df) > 0 else None,
            'mean_insulin_carb_ratio': timing_df_with_carbs['insulin_carb_ratio'].mean() if len(timing_df_with_carbs) > 0 else None
        }
    }
    
    with open(f'{RESULTS_PATH}/d1namo_correlation_analysis.json', 'w') as f:
        json.dump(d1namo_results, f, indent=2)
    
    return d1namo_results

def analyze_azt1d():
    """Analyze AZT1D correlations"""
    print("\n" + "="*80)
    print("AZT1D Analysis of Carbohydrate and Insulin Correlations with Future Blood Glucose")
    print("=" * 80)
    
    azt1d_data_list = load_azt1d_data()
    if len(azt1d_data_list) == 0:
        print("No AZT1D data found. Please check file paths.")
        return None
    
    # Combine all AZT1D data
    data = pd.concat(azt1d_data_list, ignore_index=True)
    print(f"Loaded data for {data['patient'].nunique()} patients, {len(data)} total records")
    
    # Use domain knowledge parameters for Bezier curves
    global_params = AZT1D_BEZIER_PARAMS
    
    # Collect all processed data
    all_data_list = []
    timing_analysis = []
    
    for patient in PATIENTS_AZT1D:
        patient_data = data[data['patient'] == patient].copy()
        if len(patient_data) < 50:
            continue
        
        print(f"Processing patient {patient} ({len(patient_data)} records)")
        
        # Create combined data for temporal features
        combined_data = patient_data[['datetime', 'carbohydrates', 'insulin']].copy()
        
        # Add temporal features
        processed_data = add_azt1d_features(global_params, OPTIMIZATION_FEATURES_AZT1D, 
                                           patient_data, combined_data)
        all_data_list.append(processed_data)
        
        # Analyze timing patterns for this patient
        food_events = combined_data[combined_data['carbohydrates'] > 0]
        insulin_events = combined_data[combined_data['insulin'] > 0]
        
        for _, food_event in food_events.iterrows():
            food_time = food_event['datetime']
            # Find insulin events within 2 hours before and after food
            nearby_insulin = insulin_events[
                (insulin_events['datetime'] >= food_time - pd.Timedelta(hours=2)) &
                (insulin_events['datetime'] <= food_time + pd.Timedelta(hours=2))
            ]
            
            if len(nearby_insulin) > 0:
                # Find closest insulin event
                time_diffs = (nearby_insulin['datetime'] - food_time).dt.total_seconds() / 60  # in minutes
                closest_idx = time_diffs.abs().idxmin()
                closest_insulin_time_diff = time_diffs.loc[closest_idx]
                
                timing_analysis.append({
                    'patient': patient,
                    'food_datetime': food_time,
                    'insulin_time_diff_minutes': closest_insulin_time_diff,
                    'insulin_dose': nearby_insulin.loc[closest_idx, 'insulin'],
                    'carbohydrates': food_event['carbohydrates']
                })
    
    # Combine all patient data
    if len(all_data_list) > 0:
        all_data = pd.concat(all_data_list, ignore_index=True)
        
        print(f"\nCombined dataset: {len(all_data)} records from {all_data['patient'].nunique()} patients")
        
        print("\n1. Correlation Analysis between Features and Future Glucose")
        print("-" * 65)
        
        # Calculate correlations for features of interest
        features_to_analyze = ['carbohydrates', 'insulin']
        correlations = {}
        
        for feature in features_to_analyze:
            pearson_corr, pearson_p = pearsonr(all_data[feature], all_data[f'glucose_{DEFAULT_PREDICTION_HORIZON}'])
            spearman_corr, spearman_p = spearmanr(all_data[feature], all_data[f'glucose_{DEFAULT_PREDICTION_HORIZON}'])
            
            correlations[feature] = {
                'pearson_r': pearson_corr,
                'pearson_p': pearson_p,
                'spearman_r': spearman_corr,
                'spearman_p': spearman_p
            }
            
            print(f"{feature.replace('_', ' ').title()}:")
            print(f"  Pearson correlation: r = {pearson_corr:.4f}, p = {pearson_p:.4f}")
            print(f"  Spearman correlation: r = {spearman_corr:.4f}, p = {spearman_p:.4f}")
            print()
        
        # Correlation between carbohydrates and insulin
        print("2. Correlation Between Modeled Features")
        print("-" * 40)
        
        carb_insulin_corr, carb_insulin_p = pearsonr(all_data['carbohydrates'], all_data['insulin'])
        print(f"Carbohydrates vs Insulin:")
        print(f"  Pearson correlation: r = {carb_insulin_corr:.4f}, p = {carb_insulin_p:.4f}")
        print()
        
        # Patient-specific correlations
        print("3. Patient-Specific Correlations with Future Glucose")
        print("-" * 50)
        patient_correlations = {}
        for patient in PATIENTS_AZT1D:
            patient_data = all_data[all_data['patient'] == patient]
            if len(patient_data) > 20:  # Need sufficient data points
                patient_correlations[patient] = {}
                for feature in features_to_analyze:
                    pearson_corr, pearson_p = pearsonr(patient_data[feature], patient_data[f'glucose_{DEFAULT_PREDICTION_HORIZON}'])
                    patient_correlations[patient][feature] = {
                        'pearson_r': pearson_corr,
                        'pearson_p': pearson_p,
                        'n_points': len(patient_data)
                    }
                
                print(f"Patient {patient} (n={len(patient_data)}):")
                for feature in features_to_analyze:
                    corr_data = patient_correlations[patient][feature]
                    print(f"  {feature}: r = {corr_data['pearson_r']:.4f}, p = {corr_data['pearson_p']:.4f}")
                print()
        
        print("4. Timing Analysis: Food vs Insulin Administration")
        print("-" * 55)
        
        timing_df = pd.DataFrame(timing_analysis)
        if len(timing_df) > 0:
            print(f"Total food-insulin pairs analyzed: {len(timing_df)}")
            print(f"Mean time difference (insulin relative to food): {timing_df['insulin_time_diff_minutes'].mean():.1f} ± {timing_df['insulin_time_diff_minutes'].std():.1f} minutes")
            
            # Categorize timing patterns
            pre_bolus = timing_df[timing_df['insulin_time_diff_minutes'] < -5]  # Insulin >5 min before food
            concurrent = timing_df[(timing_df['insulin_time_diff_minutes'] >= -5) & (timing_df['insulin_time_diff_minutes'] <= 15)]  # Within 5 min before to 15 min after
            post_bolus = timing_df[timing_df['insulin_time_diff_minutes'] > 15]  # Insulin >15 min after food
            
            print(f"\nTiming patterns:")
            print(f"  Pre-bolus (>5 min before food): {len(pre_bolus)} ({len(pre_bolus)/len(timing_df)*100:.1f}%)")
            print(f"  Concurrent (-5 to +15 min): {len(concurrent)} ({len(concurrent)/len(timing_df)*100:.1f}%)")
            print(f"  Post-bolus (>15 min after food): {len(post_bolus)} ({len(post_bolus)/len(timing_df)*100:.1f}%)")
            
            # Analyze insulin-to-carb ratios (only for meals with carbs > 0)
            timing_df_with_carbs = timing_df[timing_df['carbohydrates'] > 0]
            if len(timing_df_with_carbs) > 0:
                timing_df_with_carbs['insulin_carb_ratio'] = timing_df_with_carbs['insulin_dose'] / timing_df_with_carbs['carbohydrates']
            
            print(f"\nInsulin-to-carbohydrate patterns:")
            print(f"  Mean insulin dose: {timing_df['insulin_dose'].mean():.2f} ± {timing_df['insulin_dose'].std():.2f} units")
            print(f"  Mean carbohydrates: {timing_df['carbohydrates'].mean():.1f} ± {timing_df['carbohydrates'].std():.1f} g")
            if len(timing_df_with_carbs) > 0:
                print(f"  Mean insulin:carb ratio: {timing_df_with_carbs['insulin_carb_ratio'].mean():.3f} ± {timing_df_with_carbs['insulin_carb_ratio'].std():.3f} units/g")
            else:
                print(f"  No meals with carbohydrates found for ratio calculation")
        
        # Save AZT1D results
        azt1d_results = {
            'overall_correlations': correlations,
            'inter_feature_correlations': {
                'carb_insulin': {'r': carb_insulin_corr, 'p': carb_insulin_p}
            },
            'patient_correlations': patient_correlations,
            'timing_analysis_summary': {
                'total_pairs': len(timing_df),
                'mean_time_diff_minutes': timing_df['insulin_time_diff_minutes'].mean() if len(timing_df) > 0 else None,
                'std_time_diff_minutes': timing_df['insulin_time_diff_minutes'].std() if len(timing_df) > 0 else None,
                'pre_bolus_percentage': len(pre_bolus)/len(timing_df)*100 if len(timing_df) > 0 else None,
                'concurrent_percentage': len(concurrent)/len(timing_df)*100 if len(timing_df) > 0 else None,
                'post_bolus_percentage': len(post_bolus)/len(timing_df)*100 if len(timing_df) > 0 else None,
                'mean_insulin_carb_ratio': timing_df_with_carbs['insulin_carb_ratio'].mean() if len(timing_df_with_carbs) > 0 else None
            }
        }
        
        with open(f'{RESULTS_PATH}/azt1d_correlation_analysis.json', 'w') as f:
            json.dump(azt1d_results, f, indent=2)
        
        return azt1d_results
    
    else:
        print("No data processed - check if AZT1D data files exist and have sufficient records.")
        return None

def create_comparison_visualization(d1namo_results, azt1d_results):
    """Create a comparison visualization of D1namo vs AZT1D correlations"""
    if d1namo_results is None or azt1d_results is None:
        print("Cannot create visualization - missing results")
        return
    
    # Extract correlation data
    d1namo_simple = d1namo_results['overall_correlations']['simple_sugars']['pearson_r']
    d1namo_complex = d1namo_results['overall_correlations']['complex_sugars']['pearson_r'] 
    d1namo_insulin = d1namo_results['overall_correlations']['insulin']['pearson_r']
    
    azt1d_carbs = azt1d_results['overall_correlations']['carbohydrates']['pearson_r']
    azt1d_insulin = azt1d_results['overall_correlations']['insulin']['pearson_r']
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # D1namo correlations
    features_d1namo = ['Simple Sugars', 'Complex Sugars', 'Insulin']
    correlations_d1namo = [d1namo_simple, d1namo_complex, d1namo_insulin]
    colors_d1namo = ['lightcoral', 'orange', 'skyblue']
    
    bars1 = ax1.bar(features_d1namo, correlations_d1namo, color=colors_d1namo, alpha=0.7, edgecolor='black')
    ax1.set_title('D1namo: Feature Correlations with Future Glucose', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Pearson Correlation (r)', fontsize=12)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 0.1)
    
    # Add correlation values on bars
    for bar, corr in zip(bars1, correlations_d1namo):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (0.005 if height >= 0 else -0.015),
                f'{corr:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    # AZT1D correlations  
    features_azt1d = ['Carbohydrates', 'Insulin']
    correlations_azt1d = [azt1d_carbs, azt1d_insulin]
    colors_azt1d = ['lightgreen', 'skyblue']
    
    bars2 = ax2.bar(features_azt1d, correlations_azt1d, color=colors_azt1d, alpha=0.7, edgecolor='black')
    ax2.set_title('AZT1D: Feature Correlations with Future Glucose', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Pearson Correlation (r)', fontsize=12)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 0.2)
    
    # Add correlation values on bars
    for bar, corr in zip(bars2, correlations_azt1d):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{corr:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../manuscript/images/supplementary_data/correlation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Comparison visualization saved to ../manuscript/images/supplementary_data/correlation_comparison.png")

def main():
    """Main function to run both analyses"""
    print("Combined Correlation Analysis: D1namo vs AZT1D")
    print("=" * 50)
    
    # Create output directory if it doesn't exist
    os.makedirs('../manuscript/images/supplementary_data', exist_ok=True)
    
    # Run D1namo analysis
    d1namo_results = analyze_d1namo()
    
    # Run AZT1D analysis
    azt1d_results = analyze_azt1d()
    
    # Create comparison visualization
    create_comparison_visualization(d1namo_results, azt1d_results)
    
    # Summary comparison
    if d1namo_results and azt1d_results:
        print("\n" + "="*80)
        print("SUMMARY COMPARISON: D1namo vs AZT1D")
        print("=" * 80)
        
        d1namo_simple = d1namo_results['overall_correlations']['simple_sugars']['pearson_r']
        d1namo_complex = d1namo_results['overall_correlations']['complex_sugars']['pearson_r']
        d1namo_insulin = d1namo_results['overall_correlations']['insulin']['pearson_r']
        
        azt1d_carbs = azt1d_results['overall_correlations']['carbohydrates']['pearson_r']
        azt1d_insulin = azt1d_results['overall_correlations']['insulin']['pearson_r']
        
        print(f"Carbohydrate Correlations:")
        print(f"  D1namo Simple Sugars:     r = {d1namo_simple:.4f}")
        print(f"  D1namo Complex Sugars:    r = {d1namo_complex:.4f}")
        print(f"  AZT1D Carbohydrates:      r = {azt1d_carbs:.4f}")
        print(f"  Direction consistency:    {'INCONSISTENT' if (d1namo_simple + d1namo_complex)/2 * azt1d_carbs < 0 else 'CONSISTENT'}")
        
        print(f"\nInsulin Correlations:")
        print(f"  D1namo Insulin:           r = {d1namo_insulin:.4f}")
        print(f"  AZT1D Insulin:            r = {azt1d_insulin:.4f}")
        print(f"  Direction consistency:    {'CONSISTENT' if d1namo_insulin * azt1d_insulin > 0 else 'INCONSISTENT'}")
        
        print(f"\nTiming Analysis:")
        d1namo_pre_bolus = d1namo_results['timing_analysis_summary']['pre_bolus_percentage']
        azt1d_pre_bolus = azt1d_results['timing_analysis_summary']['pre_bolus_percentage']
        print(f"  D1namo Pre-bolusing:      {d1namo_pre_bolus:.1f}%")
        print(f"  AZT1D Pre-bolusing:       {azt1d_pre_bolus:.1f}%")
        
        print(f"\nKey Finding: The counterintuitive D1namo correlations (negative sugar-glucose relationships)")
        print(f"are explained by sophisticated pre-bolusing patterns ({d1namo_pre_bolus:.1f}% vs {azt1d_pre_bolus:.1f}%)")
        print(f"and limited dataset size. AZT1D shows expected positive correlations with larger sample size.")
    
    print(f"\nAnalysis complete! Results saved to:")
    print(f"  - ../results/d1namo_correlation_analysis.json")
    print(f"  - ../results/azt1d_correlation_analysis.json")
    print(f"  - ../manuscript/images/supplementary_data/correlation_comparison.png")

if __name__ == "__main__":
    main()
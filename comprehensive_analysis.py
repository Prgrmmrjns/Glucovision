import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, f_oneway
import warnings
import os
warnings.filterwarnings('ignore')

def load_and_analyze_data():
    """Load and analyze the comparison data for both datasets"""
    
    # Load data
    d1namo_df = pd.read_csv('results/d1namo_comparison.csv')
    azt_df = pd.read_csv('results/azt1d_comparison.csv') if os.path.exists('results/azt1d_comparison.csv') else None
    
    # Extract approaches
    approaches = ['Glucose+Insulin', 'LastMeal', 'Bezier']
    
    results = {}
    
    # Analyze D1namo
    results['d1namo'] = analyze_dataset(d1namo_df, 'D1namo', approaches)
    
    # Analyze AZT1D if available
    if azt_df is not None:
        results['azt1d'] = analyze_dataset(azt_df, 'AZT1D', approaches)
    
    return results

def analyze_dataset(df, dataset_name, approaches):
    """Analyze a single dataset"""
    
    # Group by merge keys to handle duplicates
    merge_keys = ['Prediction Horizon', 'Patient', 'Day', 'Hour']
    df_grouped = df.groupby(merge_keys + ['Approach'])['RMSE'].mean().reset_index()
    
    # Extract each approach
    approach_data = {}
    for approach in approaches:
        approach_data[approach] = df_grouped[df_grouped['Approach'] == approach]
    
    # Calculate overall statistics
    overall_stats = {}
    for approach in approaches:
        overall_stats[approach] = {
            'mean': approach_data[approach]['RMSE'].mean(),
            'std': approach_data[approach]['RMSE'].std(),
            'count': len(approach_data[approach])
        }
    
    # Calculate horizon-specific statistics
    horizons = sorted(df_grouped['Prediction Horizon'].unique())
    horizon_stats = {}
    
    for horizon in horizons:
        horizon_stats[horizon] = {}
        for approach in approaches:
            h_data = approach_data[approach][approach_data[approach]['Prediction Horizon'] == horizon]
            horizon_stats[horizon][approach] = {
                'mean': h_data['RMSE'].mean(),
                'std': h_data['RMSE'].std(),
                'count': len(h_data)
            }
    
    # Calculate patient-specific statistics
    patients = sorted(df_grouped['Patient'].unique())
    patient_stats = {}
    
    for patient in patients:
        patient_stats[patient] = {}
        for approach in approaches:
            p_data = approach_data[approach][approach_data[approach]['Patient'] == patient]
            patient_stats[patient][approach] = {
                'mean': p_data['RMSE'].mean(),
                'std': p_data['RMSE'].std(),
                'count': len(p_data)
            }
    
    # Statistical comparisons
    comparisons = {}
    
    # Overall comparisons
    for i, approach1 in enumerate(approaches):
        for approach2 in approaches[i+1:]:
            key = f"{approach1}_vs_{approach2}"
            data1 = approach_data[approach1]['RMSE'].values
            data2 = approach_data[approach2]['RMSE'].values
            
            # Ensure we have paired data
            if len(data1) == len(data2):
                t_stat, p_val = ttest_rel(data1, data2)
                improvement = ((data2.mean() - data1.mean()) / data2.mean()) * 100
                comparisons[key] = {
                    't_stat': t_stat,
                    'p_value': p_val,
                    'improvement_pct': improvement,
                    'better_approach': approach1 if data1.mean() < data2.mean() else approach2
                }
    
    # Horizon-specific comparisons
    horizon_comparisons = {}
    for horizon in horizons:
        horizon_comparisons[horizon] = {}
        for i, approach1 in enumerate(approaches):
            for approach2 in approaches[i+1:]:
                key = f"{approach1}_vs_{approach2}"
                data1 = approach_data[approach1][approach_data[approach1]['Prediction Horizon'] == horizon]['RMSE'].values
                data2 = approach_data[approach2][approach_data[approach2]['Prediction Horizon'] == horizon]['RMSE'].values
                
                if len(data1) > 0 and len(data2) > 0:
                    t_stat, p_val = ttest_rel(data1, data2)
                    improvement = ((data2.mean() - data1.mean()) / data2.mean()) * 100
                    horizon_comparisons[horizon][key] = {
                        't_stat': t_stat,
                        'p_value': p_val,
                        'improvement_pct': improvement,
                        'better_approach': approach1 if data1.mean() < data2.mean() else approach2
                    }
    
    return {
        'overall_stats': overall_stats,
        'horizon_stats': horizon_stats,
        'patient_stats': patient_stats,
        'comparisons': comparisons,
        'horizon_comparisons': horizon_comparisons,
        'approach_data': approach_data,
        'horizons': horizons,
        'patients': patients
    }

def generate_main_comparison_table(results):
    """Generate the main comparison table for the manuscript"""
    
    horizons = [6, 9, 12, 18, 24]
    horizon_names = {6: "30 min", 9: "45 min", 12: "60 min", 18: "90 min", 24: "120 min"}
    
    table_content = ""
    
    for horizon in horizons:
        row_parts = []
        
        # D1namo data
        if 'd1namo' in results:
            d1_stats = results['d1namo']['horizon_stats'].get(horizon, {})
            
            # Glucose+Insulin vs Bezier comparison
            glucose_insulin = d1_stats.get('Glucose+Insulin', {'mean': np.nan, 'std': np.nan})
            bezier = d1_stats.get('Bezier', {'mean': np.nan, 'std': np.nan})
            
            glucose_insulin_str = f"{glucose_insulin['mean']:.2f} ± {glucose_insulin['std']:.2f}" if not np.isnan(glucose_insulin['mean']) else "N/A"
            bezier_str = f"{bezier['mean']:.2f} ± {bezier['std']:.2f}" if not np.isnan(bezier['mean']) else "N/A"
            
            # Get p-value for comparison
            comparison_key = "Glucose+Insulin_vs_Bezier"
            p_val = results['d1namo']['horizon_comparisons'].get(horizon, {}).get(comparison_key, {}).get('p_value', np.nan)
            p_str = f"{p_val:.3f}" if not np.isnan(p_val) else "N/A"
            
            # Apply bold formatting for significant differences
            if not np.isnan(p_val) and p_val < 0.05:
                if bezier['mean'] < glucose_insulin['mean']:
                    bezier_str = f"\\textbf{{{bezier_str}}}"
                else:
                    glucose_insulin_str = f"\\textbf{{{glucose_insulin_str}}}"
            
            row_parts.extend([glucose_insulin_str, bezier_str, p_str])
        else:
            row_parts.extend(["N/A", "N/A", "N/A"])
        
        # AZT1D data
        if 'azt1d' in results:
            azt_stats = results['azt1d']['horizon_stats'].get(horizon, {})
            
            glucose_insulin = azt_stats.get('Glucose+Insulin', {'mean': np.nan, 'std': np.nan})
            bezier = azt_stats.get('Bezier', {'mean': np.nan, 'std': np.nan})
            
            glucose_insulin_str = f"{glucose_insulin['mean']:.2f} ± {glucose_insulin['std']:.2f}" if not np.isnan(glucose_insulin['mean']) else "N/A"
            bezier_str = f"{bezier['mean']:.2f} ± {bezier['std']:.2f}" if not np.isnan(bezier['mean']) else "N/A"
            
            comparison_key = "Glucose+Insulin_vs_Bezier"
            p_val = results['azt1d']['horizon_comparisons'].get(horizon, {}).get(comparison_key, {}).get('p_value', np.nan)
            p_str = f"{p_val:.3f}" if not np.isnan(p_val) else "N/A"
            
            if not np.isnan(p_val) and p_val < 0.05:
                if bezier['mean'] < glucose_insulin['mean']:
                    bezier_str = f"\\textbf{{{bezier_str}}}"
                else:
                    glucose_insulin_str = f"\\textbf{{{glucose_insulin_str}}}"
            
            row_parts.extend([glucose_insulin_str, bezier_str, p_str])
        else:
            row_parts.extend(["N/A", "N/A", "N/A"])
        
        h_name = horizon_names.get(horizon, f"{horizon*5} min")
        table_content += f"\\rowcolor{{gray!5}} {h_name} & {' & '.join(row_parts)} \\\\\n"
    
    latex_table = f"""\\begin{{table}}[ht]
\\centering
\\caption{{Comparison of three approaches: Glucose+Insulin (baseline), LastMeal (cumulative features), and Bezier (temporal features) across D1namo and AZT1D datasets. Values show RMSE $\\pm$ standard deviation in mg/dL, with paired t-test p-values comparing Glucose+Insulin vs Bezier. Bold indicates statistically significant better performance (p<0.05).}}
\\label{{tab:three_approach_comparison}}
\\renewcommand{{\\arraystretch}}{{1.2}}
\\setlength{{\\tabcolsep}}{{2pt}}
\\resizebox{{\\textwidth}}{{!}}{{%
{{\\scriptsize
\\begin{{tabular}}{{|l|cc>{{\\columncolor{{blue!10}}\\scriptsize}}c|cc>{{\\columncolor{{blue!10}}\\scriptsize}}c|}}
\\hline
\\rowcolor{{gray!25}} \\multirow{{2}}{{*}}[1ex]{{\\textbf{{Prediction Horizon}}}} & \\multicolumn{{3}}{{c|}}{{\\cellcolor{{gray!25}}\\textbf{{D1namo}}}} & \\multicolumn{{3}}{{c|}}{{\\cellcolor{{gray!25}}\\textbf{{AZT1D}}}} \\\\
\\rowcolor{{gray!25}} & \\cellcolor{{orange!15}}{{\\textbf{{Glucose+Insulin}}}} & \\cellcolor{{green!15}}{{\\textbf{{Bezier}}}} & \\cellcolor{{blue!15}}{{\\textbf{{p}}}} & \\cellcolor{{orange!15}}{{\\textbf{{Glucose+Insulin}}}} & \\cellcolor{{green!15}}{{\\textbf{{Bezier}}}} & \\cellcolor{{blue!15}}{{\\textbf{{p}}}} \\\\
\\hline
{table_content}
\\hline
\\end{{tabular}}
}}
}}
\\end{{table}}"""
    
    return latex_table

def generate_patient_individual_table(results, dataset_name):
    """Generate patient individual table for a specific dataset"""
    
    if dataset_name not in results:
        return ""
    
    dataset_results = results[dataset_name]
    patients = dataset_results['patients']
    horizons = dataset_results['horizons']
    horizon_names = {6: "30 min", 9: "45 min", 12: "60 min", 18: "90 min", 24: "120 min"}
    
    table_content = ""
    
    for patient in patients:
        row_parts = [f"\\rowcolor{{gray!5}} {{\\tiny {patient}}}"]
        
        for horizon in horizons:
            patient_horizon_data = dataset_results['patient_stats'].get(patient, {})
            
            glucose_insulin = patient_horizon_data.get('Glucose+Insulin', {'mean': np.nan, 'std': np.nan})
            bezier = patient_horizon_data.get('Bezier', {'mean': np.nan, 'std': np.nan})
            
            glucose_insulin_str = f"{glucose_insulin['mean']:.1f}±{glucose_insulin['std']:.1f}" if not np.isnan(glucose_insulin['mean']) else "N/A"
            bezier_str = f"{bezier['mean']:.1f}±{bezier['std']:.1f}" if not np.isnan(bezier['mean']) else "N/A"
            
            # Calculate improvement percentage
            if not np.isnan(glucose_insulin['mean']) and not np.isnan(bezier['mean']) and glucose_insulin['mean'] > 0:
                improvement = ((glucose_insulin['mean'] - bezier['mean']) / glucose_insulin['mean']) * 100
                improvement_str = f"({improvement:+.1f}%)"
            else:
                improvement_str = ""
            
            row_parts.append(f"& \\cellcolor{{orange!10}}{{\\tiny {glucose_insulin_str}}} & \\cellcolor{{green!10}}{{\\tiny {bezier_str}}} & \\cellcolor{{blue!10}}{{\\tiny {improvement_str}}} ")
        
        table_content += " ".join(row_parts) + "\\\\\n"
    
    # Overall row
    overall_row = ["\\hline\n\\rowcolor{blue!10} \\textbf{All}"]
    
    for horizon in horizons:
        horizon_data = dataset_results['horizon_stats'].get(horizon, {})
        
        glucose_insulin = horizon_data.get('Glucose+Insulin', {'mean': np.nan, 'std': np.nan})
        bezier = horizon_data.get('Bezier', {'mean': np.nan, 'std': np.nan})
        
        glucose_insulin_str = f"{glucose_insulin['mean']:.1f}±{glucose_insulin['std']:.1f}" if not np.isnan(glucose_insulin['mean']) else "N/A"
        bezier_str = f"{bezier['mean']:.1f}±{bezier['std']:.1f}" if not np.isnan(bezier['mean']) else "N/A"
        
        if not np.isnan(glucose_insulin['mean']) and not np.isnan(bezier['mean']) and glucose_insulin['mean'] > 0:
            improvement = ((glucose_insulin['mean'] - bezier['mean']) / glucose_insulin['mean']) * 100
            improvement_str = f"({improvement:+.1f}%)"
        else:
            improvement_str = ""
        
        overall_row.append(f"& \\cellcolor{{orange!10}}{{\\tiny {glucose_insulin_str}}} & \\cellcolor{{green!10}}{{\\tiny {bezier_str}}} & \\cellcolor{{blue!10}}{{\\tiny {improvement_str}}} ")
    
    table_content += " ".join(overall_row) + "\\\\\n"
    
    latex_table = f"""\\begin{{table}}[ht]
\\centering
\\caption{{Patient-level RMSE results by prediction horizon for {dataset_name}. Values show RMSE $\\pm$ std (mg/dL) with improvement percentages comparing Glucose+Insulin vs Bezier approaches.}}
\\label{{tab:patient_individual_{dataset_name.lower()}}}
{{\\tiny
\\setlength{{\\tabcolsep}}{{2pt}}
\\begin{{tabular}}{{|p{{0.4cm}}|p{{0.8cm}}p{{0.8cm}}>{{\\columncolor{{blue!10}}\\tiny}}p{{0.3cm}}|p{{0.8cm}}p{{0.8cm}}>{{\\columncolor{{blue!10}}\\tiny}}p{{0.3cm}}|p{{0.8cm}}p{{0.8cm}}>{{\\columncolor{{blue!10}}\\tiny}}p{{0.3cm}}|p{{0.8cm}}p{{0.8cm}}>{{\\columncolor{{blue!10}}\\tiny}}p{{0.3cm}}|p{{0.8cm}}p{{0.8cm}}>{{\\columncolor{{blue!10}}\\tiny}}p{{0.3cm}}|}}
\\hline
\\rowcolor{{gray!25}} {{\\tiny\\textbf{{Pat.}}}} & \\multicolumn{{3}}{{c|}}{{{horizon_names.get(6, '30 min')}}} & \\multicolumn{{3}}{{c|}}{{{horizon_names.get(9, '45 min')}}} & \\multicolumn{{3}}{{c|}}{{{horizon_names.get(12, '60 min')}}} & \\multicolumn{{3}}{{c|}}{{{horizon_names.get(18, '90 min')}}} & \\multicolumn{{3}}{{c|}}{{{horizon_names.get(24, '120 min')}}} \\\\
\\rowcolor{{gray!25}} & \\cellcolor{{orange!15}}{{\\tiny\\textbf{{Glucose+Insulin}}}} & \\cellcolor{{green!15}}{{\\tiny\\textbf{{Bezier}}}} & \\cellcolor{{blue!15}}{{\\tiny\\textbf{{Imp.}}}} & \\cellcolor{{orange!15}}{{\\tiny\\textbf{{Glucose+Insulin}}}} & \\cellcolor{{green!15}}{{\\tiny\\textbf{{Bezier}}}} & \\cellcolor{{blue!15}}{{\\tiny\\textbf{{Imp.}}}} & \\cellcolor{{orange!15}}{{\\tiny\\textbf{{Glucose+Insulin}}}} & \\cellcolor{{green!15}}{{\\tiny\\textbf{{Bezier}}}} & \\cellcolor{{blue!15}}{{\\tiny\\textbf{{Imp.}}}} & \\cellcolor{{orange!15}}{{\\tiny\\textbf{{Glucose+Insulin}}}} & \\cellcolor{{green!15}}{{\\tiny\\textbf{{Bezier}}}} & \\cellcolor{{blue!15}}{{\\tiny\\textbf{{Imp.}}}} & \\cellcolor{{orange!15}}{{\\tiny\\textbf{{Glucose+Insulin}}}} & \\cellcolor{{green!15}}{{\\tiny\\textbf{{Bezier}}}} & \\cellcolor{{blue!15}}{{\\tiny\\textbf{{Imp.}}}} \\\\
\\hline
{table_content}
\\hline
\\end{{tabular}}
}}
\\end{{table}}"""
    
    return latex_table

def generate_key_findings(results):
    """Generate key findings for the Results section"""
    
    findings = []
    
    # Overall performance comparison
    if 'd1namo' in results:
        d1_results = results['d1namo']
        glucose_insulin_overall = d1_results['overall_stats']['Glucose+Insulin']
        bezier_overall = d1_results['overall_stats']['Bezier']
        lastmeal_overall = d1_results['overall_stats']['LastMeal']
        
        findings.append(f"D1namo overall performance: Glucose+Insulin {glucose_insulin_overall['mean']:.2f}±{glucose_insulin_overall['std']:.2f} mg/dL, LastMeal {lastmeal_overall['mean']:.2f}±{lastmeal_overall['std']:.2f} mg/dL, Bezier {bezier_overall['mean']:.2f}±{bezier_overall['std']:.2f} mg/dL")
        
        # Best approach
        approaches = ['Glucose+Insulin', 'LastMeal', 'Bezier']
        best_approach = min(approaches, key=lambda x: d1_results['overall_stats'][x]['mean'])
        findings.append(f"Best performing approach on D1namo: {best_approach} ({d1_results['overall_stats'][best_approach]['mean']:.2f} mg/dL)")
    
    if 'azt1d' in results:
        azt_results = results['azt1d']
        glucose_insulin_overall = azt_results['overall_stats']['Glucose+Insulin']
        bezier_overall = azt_results['overall_stats']['Bezier']
        lastmeal_overall = azt_results['overall_stats']['LastMeal']
        
        findings.append(f"AZT1D overall performance: Glucose+Insulin {glucose_insulin_overall['mean']:.2f}±{glucose_insulin_overall['std']:.2f} mg/dL, LastMeal {lastmeal_overall['mean']:.2f}±{lastmeal_overall['std']:.2f} mg/dL, Bezier {bezier_overall['mean']:.2f}±{bezier_overall['std']:.2f} mg/dL")
        
        best_approach = min(approaches, key=lambda x: azt_results['overall_stats'][x]['mean'])
        findings.append(f"Best performing approach on AZT1D: {best_approach} ({azt_results['overall_stats'][best_approach]['mean']:.2f} mg/dL)")
    
    # Horizon-specific findings
    for dataset_name in results:
        dataset_results = results[dataset_name]
        horizons = dataset_results['horizons']
        
        for horizon in horizons:
            horizon_data = dataset_results['horizon_stats'].get(horizon, {})
            if horizon_data:
                best_approach = min(horizon_data.keys(), key=lambda x: horizon_data[x]['mean'])
                findings.append(f"{dataset_name} {horizon*5}min: Best approach {best_approach} ({horizon_data[best_approach]['mean']:.2f} mg/dL)")
    
    # Statistical significance findings
    for dataset_name in results:
        dataset_results = results[dataset_name]
        comparisons = dataset_results['comparisons']
        
        for comparison_key, comparison_data in comparisons.items():
            if comparison_data['p_value'] < 0.05:
                findings.append(f"{dataset_name} {comparison_key}: {comparison_data['better_approach']} significantly better (p={comparison_data['p_value']:.3f}, improvement={comparison_data['improvement_pct']:.1f}%)")
    
    # Patient variability findings
    for dataset_name in results:
        dataset_results = results[dataset_name]
        patients = dataset_results['patients']
        
        # Find best and worst performing patients for each approach
        for approach in ['Glucose+Insulin', 'LastMeal', 'Bezier']:
            patient_performance = []
            for patient in patients:
                patient_data = dataset_results['patient_stats'].get(patient, {}).get(approach, {})
                if 'mean' in patient_data and not np.isnan(patient_data['mean']):
                    patient_performance.append((patient, patient_data['mean']))
            
            if patient_performance:
                best_patient = min(patient_performance, key=lambda x: x[1])
                worst_patient = max(patient_performance, key=lambda x: x[1])
                findings.append(f"{dataset_name} {approach}: Best patient {best_patient[0]} ({best_patient[1]:.2f} mg/dL), worst patient {worst_patient[0]} ({worst_patient[1]:.2f} mg/dL)")
    
    return findings

def main():
    """Main analysis function"""
    
    print("Loading and analyzing data...")
    results = load_and_analyze_data()
    
    print("Generating main comparison table...")
    main_table = generate_main_comparison_table(results)
    
    print("Generating patient individual tables...")
    d1namo_table = generate_patient_individual_table(results, 'd1namo')
    azt1d_table = generate_patient_individual_table(results, 'azt1d')
    
    print("Generating key findings...")
    findings = generate_key_findings(results)
    
    # Save tables
    os.makedirs('manuscript/tables', exist_ok=True)
    
    with open('manuscript/tables/comparison_table.tex', 'w') as f:
        f.write(main_table)
    
    with open('manuscript/tables/patient_individual_d1namo.tex', 'w') as f:
        f.write(d1namo_table)
    
    with open('manuscript/tables/patient_individual_azt1d.tex', 'w') as f:
        f.write(azt1d_table)
    
    # Save findings
    with open('results/key_findings.txt', 'w') as f:
        f.write('\n'.join(findings))
    
    print("Analysis complete!")
    print("\nKey findings:")
    for finding in findings[:10]:  # Show first 10 findings
        print(f"- {finding}")
    
    return results, main_table, d1namo_table, azt1d_table, findings

if __name__ == "__main__":
    main()

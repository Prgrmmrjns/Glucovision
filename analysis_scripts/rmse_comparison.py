import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
import warnings
import os
warnings.filterwarnings('ignore')

def generate_detailed_analysis():
    """Generate detailed analysis of RMSE differences by prediction horizon and patient"""
    d1namo_df = pd.read_csv('../results/d1namo_predictions.csv')
    baseline_df = pd.read_csv('../results/baseline_predictions.csv')
    
    # Merge the data
    merge_keys = ['Prediction Horizon', 'Patient', 'Day', 'Hour']
    merged_df = pd.merge(d1namo_df, baseline_df, on=merge_keys, suffixes=('_d1namo', '_baseline'))
    
    patients = sorted(merged_df['Patient'].unique())
    horizons = sorted(merged_df['Prediction Horizon'].unique())
    
    analysis_text = []
    analysis_text.append("DETAILED RMSE ANALYSIS: MACRONUTRIENT-INFORMED vs BASELINE MODEL")
    analysis_text.append("=" * 80)
    analysis_text.append("")
    
    # Overall statistics
    overall_macro = merged_df['RMSE_d1namo'].mean()
    overall_base = merged_df['RMSE_baseline'].mean()
    overall_improvement = ((overall_base - overall_macro) / overall_base) * 100
    
    analysis_text.append("OVERALL PERFORMANCE:")
    analysis_text.append(f"Macronutrient-informed model: {overall_macro:.2f} ± {merged_df['RMSE_d1namo'].std():.2f} mg/dL")
    analysis_text.append(f"Baseline model: {overall_base:.2f} ± {merged_df['RMSE_baseline'].std():.2f} mg/dL")
    analysis_text.append(f"Overall improvement: {overall_improvement:.1f}%")
    analysis_text.append("")
    
    # Analysis by prediction horizon
    analysis_text.append("ANALYSIS BY PREDICTION HORIZON:")
    analysis_text.append("-" * 40)
    
    horizon_names = {6: "30 min", 9: "45 min", 12: "60 min", 18: "90 min", 24: "120 min"}
    significant_horizons = []
    
    for horizon in horizons:
        h_data = merged_df[merged_df['Prediction Horizon'] == horizon]
        d1n_rmse = h_data['RMSE_d1namo'].mean()
        base_rmse = h_data['RMSE_baseline'].mean()
        d1n_std = h_data['RMSE_d1namo'].std()
        base_std = h_data['RMSE_baseline'].std()
        t_stat, p_val = ttest_rel(h_data['RMSE_d1namo'], h_data['RMSE_baseline'])
        improvement = ((base_rmse - d1n_rmse) / base_rmse) * 100
        
        analysis_text.append(f"{horizon_names[horizon]}:")
        analysis_text.append(f"  Macro: {d1n_rmse:.2f} ± {d1n_std:.2f} mg/dL")
        analysis_text.append(f"  Base:  {base_rmse:.2f} ± {base_std:.2f} mg/dL")
        analysis_text.append(f"  Improvement: {improvement:.1f}% (p={p_val:.3f})")
        
        if p_val < 0.05:
            significant_horizons.append(horizon_names[horizon])
            analysis_text.append(f"  *** STATISTICALLY SIGNIFICANT ***")
        analysis_text.append("")
    
    analysis_text.append(f"Statistically significant improvements at: {', '.join(significant_horizons)}")
    analysis_text.append("")
    
    # Analysis by patient
    analysis_text.append("ANALYSIS BY PATIENT:")
    analysis_text.append("-" * 25)
    
    patient_improvements = []
    best_patients = []
    worst_patients = []
    
    for patient in patients:
        patient_data = merged_df[merged_df['Patient'] == patient]
        d1n_rmse = patient_data['RMSE_d1namo'].mean()
        base_rmse = patient_data['RMSE_baseline'].mean()
        improvement = ((base_rmse - d1n_rmse) / base_rmse) * 100
        t_stat, p_val = ttest_rel(patient_data['RMSE_d1namo'], patient_data['RMSE_baseline'])
        
        analysis_text.append(f"Patient {patient}:")
        analysis_text.append(f"  Macro: {d1n_rmse:.2f} ± {patient_data['RMSE_d1namo'].std():.2f} mg/dL")
        analysis_text.append(f"  Base:  {base_rmse:.2f} ± {patient_data['RMSE_baseline'].std():.2f} mg/dL")
        analysis_text.append(f"  Improvement: {improvement:.1f}% (p={p_val:.3f})")
        
        patient_improvements.append((patient, improvement))
        
        if improvement > 5:
            best_patients.append(f"Patient {patient} ({improvement:.1f}%)")
        elif improvement < -2:
            worst_patients.append(f"Patient {patient} ({improvement:.1f}%)")
        
        analysis_text.append("")
    
    # Patient-specific significant improvements by horizon
    analysis_text.append("PATIENT-SPECIFIC SIGNIFICANT IMPROVEMENTS BY HORIZON:")
    analysis_text.append("-" * 55)
    
    for horizon in horizons:
        significant_patients = []
        for patient in patients:
            patient_horizon_data = merged_df[(merged_df['Patient'] == patient) & 
                                           (merged_df['Prediction Horizon'] == horizon)]
            if len(patient_horizon_data) > 0:
                t_stat, p_val = ttest_rel(patient_horizon_data['RMSE_d1namo'], 
                                        patient_horizon_data['RMSE_baseline'])
                d1n_rmse = patient_horizon_data['RMSE_d1namo'].mean()
                base_rmse = patient_horizon_data['RMSE_baseline'].mean()
                
                if p_val < 0.05 and d1n_rmse < base_rmse:
                    improvement = ((base_rmse - d1n_rmse) / base_rmse) * 100
                    significant_patients.append(f"Patient {patient} ({improvement:.1f}%)")
        
        if significant_patients:
            analysis_text.append(f"{horizon_names[horizon]}: {', '.join(significant_patients)}")
    
    analysis_text.append("")
    
    # Summary insights
    analysis_text.append("KEY INSIGHTS:")
    analysis_text.append("-" * 15)
    
    # Find best and worst performing patients
    patient_improvements.sort(key=lambda x: x[1], reverse=True)
    best_patient = patient_improvements[0]
    worst_patient = patient_improvements[-1]
    
    analysis_text.append(f"• Best performing patient: Patient {best_patient[0]} ({best_patient[1]:.1f}% improvement)")
    analysis_text.append(f"• Worst performing patient: Patient {worst_patient[0]} ({worst_patient[1]:.1f}% change)")
    
    # Horizon trends
    horizon_improvements = []
    for horizon in horizons:
        h_data = merged_df[merged_df['Prediction Horizon'] == horizon]
        d1n_rmse = h_data['RMSE_d1namo'].mean()
        base_rmse = h_data['RMSE_baseline'].mean()
        improvement = ((base_rmse - d1n_rmse) / base_rmse) * 100
        horizon_improvements.append((horizon, improvement))
    
    horizon_improvements.sort(key=lambda x: x[1], reverse=True)
    best_horizon = horizon_improvements[0]
    
    analysis_text.append(f"• Best prediction horizon: {horizon_names[best_horizon[0]]} ({best_horizon[1]:.1f}% improvement)")
    
    # Performance variability
    patient_stds = []
    for patient in patients:
        patient_data = merged_df[merged_df['Patient'] == patient]
        patient_stds.append(patient_data['RMSE_d1namo'].std())
    
    analysis_text.append(f"• Patient with most variable performance: Patient {patients[np.argmax(patient_stds)]} (std: {max(patient_stds):.1f} mg/dL)")
    analysis_text.append(f"• Patient with most consistent performance: Patient {patients[np.argmin(patient_stds)]} (std: {min(patient_stds):.1f} mg/dL)")
    
    return '\n'.join(analysis_text)

def generate_latex_table():
    """Generate LaTeX table and save to manuscript/tables/"""
    d1namo_df = pd.read_csv('../results/d1namo_predictions.csv')
    baseline_df = pd.read_csv('../results/baseline_predictions.csv')
    
    # Merge the data
    merge_keys = ['Prediction Horizon', 'Patient', 'Day', 'Hour']
    merged_df = pd.merge(d1namo_df, baseline_df, on=merge_keys, suffixes=('_d1namo', '_baseline'))
    
    patients = sorted(merged_df['Patient'].unique())
    horizons = sorted(merged_df['Prediction Horizon'].unique())
    
    # Horizon data across all patients
    horizon_data = {}
    for horizon in horizons:
        h_data = merged_df[merged_df['Prediction Horizon'] == horizon]
        d1n_rmse = h_data['RMSE_d1namo'].mean()
        base_rmse = h_data['RMSE_baseline'].mean()
        d1n_std = h_data['RMSE_d1namo'].std()
        base_std = h_data['RMSE_baseline'].std()
        t_stat, p_val = ttest_rel(h_data['RMSE_d1namo'], h_data['RMSE_baseline'])
        horizon_data[int(horizon)] = {
            'd1namo': d1n_rmse,
            'baseline': base_rmse,
            'd1namo_std': d1n_std,
            'baseline_std': base_std,
            'p_value': p_val
        }
    
    # Generate LaTeX table
    latex_table = """\\begin{table}[ht]
\\centering
\\caption{Comparison of macronutrient-informed model (with macronutrient features) vs Baseline model (glucose and insulin features only) across patients and prediction horizons. Values show RMSE $\\pm$ standard deviation in mg/dL.}
\\label{tab:d1namo_baseline_comparison}
\\renewcommand{\\arraystretch}{1.2}
\\setlength{\\tabcolsep}{2pt}
\\resizebox{\\textwidth}{!}{%
{\\scriptsize
\\begin{tabular}{|p{0.5cm}|cc>{\\columncolor{blue!10}\\scriptsize}p{0.6cm}|cc>{\\columncolor{blue!10}\\scriptsize}p{0.6cm}|cc>{\\columncolor{blue!10}\\scriptsize}p{0.6cm}|cc>{\\columncolor{blue!10}\\scriptsize}p{0.6cm}|cc>{\\columncolor{blue!10}\\scriptsize}p{0.6cm}|}
\\hline
\\rowcolor{gray!25} \\multirow{2}{*}[1ex]{\\textbf{Pat.}} & \\multicolumn{3}{c|}{\\cellcolor{gray!25}\\textbf{30 min}} & \\multicolumn{3}{c|}{\\cellcolor{gray!25}\\textbf{45 min}} & \\multicolumn{3}{c|}{\\cellcolor{gray!25}\\textbf{60 min}} & \\multicolumn{3}{c|}{\\cellcolor{gray!25}\\textbf{90 min}} & \\multicolumn{3}{c|}{\\cellcolor{gray!25}\\textbf{120 min}} \\\\
\\rowcolor{gray!25} & \\cellcolor{green!15}{\\textbf{Macro}} & \\cellcolor{orange!15}{\\textbf{Base}} & \\cellcolor{blue!15}{\\textbf{p}} & \\cellcolor{green!15}{\\textbf{Macro}} & \\cellcolor{orange!15}{\\textbf{Base}} & \\cellcolor{blue!15}{\\textbf{p}} & \\cellcolor{green!15}{\\textbf{Macro}} & \\cellcolor{orange!15}{\\textbf{Base}} & \\cellcolor{blue!15}{\\textbf{p}} & \\cellcolor{green!15}{\\textbf{Macro}} & \\cellcolor{orange!15}{\\textbf{Base}} & \\cellcolor{blue!15}{\\textbf{p}} & \\cellcolor{green!15}{\\textbf{Macro}} & \\cellcolor{orange!15}{\\textbf{Base}} & \\cellcolor{blue!15}{\\textbf{p}} \\\\
\\hline"""

    # Add patient rows
    for patient in patients:
        row_color = "\\rowcolor{gray!5}" if int(patient) % 2 == 1 else "\\rowcolor{white}"
        latex_table += f"\n{row_color} {patient}"
        
        for horizon in [6, 9, 12, 18, 24]:
            patient_horizon_data = merged_df[(merged_df['Patient'] == patient) & (merged_df['Prediction Horizon'] == horizon)]
            
            if len(patient_horizon_data) > 0:
                d1n_rmse = patient_horizon_data['RMSE_d1namo'].mean()
                base_rmse = patient_horizon_data['RMSE_baseline'].mean()
                d1n_std = patient_horizon_data['RMSE_d1namo'].std()
                base_std = patient_horizon_data['RMSE_baseline'].std()
                t_stat, p_val = ttest_rel(patient_horizon_data['RMSE_d1namo'], patient_horizon_data['RMSE_baseline'])
                
                d1n_better = d1n_rmse < base_rmse
                p_sig = p_val < 0.05
                
                d1n_str = f"{d1n_rmse:.2f}±{d1n_std:.2f}"
                base_str = f"{base_rmse:.2f}±{base_std:.2f}"
                p_str = f"{p_val:.2f}"
                
                if d1n_better:
                    d1n_str = f"\\textbf{{{d1n_str}}}"
                else:
                    base_str = f"\\textbf{{{base_str}}}"
                    
                if p_sig:
                    p_str = f"\\textbf{{{p_str}}}"
            else:
                d1n_str = base_str = p_str = "N/A"
            
            latex_table += f" & \\cellcolor{{green!10}}{{{d1n_str}}} & \\cellcolor{{orange!10}}{{{base_str}}} & \\cellcolor{{blue!10}}{{{p_str}}}"
        
        latex_table += " \\\\"

    # Add overall row
    latex_table += """
\\hline
\\rowcolor{blue!10} \\textbf{All}"""

    for horizon in [6, 9, 12, 18, 24]:
        h_data = horizon_data[horizon]
        d1n = h_data['d1namo']
        base = h_data['baseline']
        d1n_std = h_data['d1namo_std']
        base_std = h_data['baseline_std']
        p_val = h_data['p_value']
        
        d1n_better = d1n < base
        p_sig = p_val < 0.05
        
        d1n_str = f"{d1n:.2f}±{d1n_std:.2f}"
        base_str = f"{base:.2f}±{base_std:.2f}"
        p_str = f"{p_val:.2f}"
        
        if d1n_better:
            d1n_str = f"\\textbf{{{d1n_str}}}"
        else:
            base_str = f"\\textbf{{{base_str}}}"
            
        if p_sig:
            p_str = f"\\textbf{{{p_str}}}"
            
        latex_table += f" & \\cellcolor{{green!10}}{{{d1n_str}}} & \\cellcolor{{orange!10}}{{{base_str}}} & \\cellcolor{{blue!10}}{{{p_str}}}"
    
    latex_table += """ \\\\
\\hline
\\end{tabular}
}
}
\\end{table}"""

    # Create directories if they don't exist
    os.makedirs('../manuscript/tables', exist_ok=True)
    os.makedirs('../results', exist_ok=True)
    
    # Save LaTeX table
    with open('../manuscript/tables/comparison_table.tex', 'w') as f:
        f.write(latex_table)
    
    # Generate and save analysis
    analysis = generate_detailed_analysis()
    with open('../results/rmse_analysis.txt', 'w') as f:
        f.write(analysis)

if __name__ == "__main__":
    generate_latex_table()

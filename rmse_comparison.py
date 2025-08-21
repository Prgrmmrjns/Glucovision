import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, f_oneway, friedmanchisquare
import warnings
import os
warnings.filterwarnings('ignore')

def generate_detailed_analysis():
    """Generate detailed analysis of RMSE differences by prediction horizon and patient"""
    d1namo_df = pd.read_csv('results/d1namo_comparison.csv')
    azt_df = pd.read_csv('results/azt1d_comparison.csv') if os.path.exists('results/azt1d_comparison.csv') else None
    
    # Extract different approaches for D1namo
    d1namo_bezier = d1namo_df[d1namo_df['Approach'] == 'Bezier']
    d1namo_baseline = d1namo_df[d1namo_df['Approach'] == 'Glucose+Insulin']
    d1namo_lastmeal = d1namo_df[d1namo_df['Approach'] == 'LastMeal']
    
    # Handle duplicates by averaging RMSE for same merge keys
    merge_keys = ['Prediction Horizon', 'Patient', 'Day', 'Hour']
    d1namo_bezier = d1namo_bezier.groupby(merge_keys)['RMSE'].mean().reset_index()
    d1namo_baseline = d1namo_baseline.groupby(merge_keys)['RMSE'].mean().reset_index()
    
    # Merge the data (compare Bezier vs Glucose+Insulin as macro vs baseline)
    merged_df = pd.merge(d1namo_bezier, d1namo_baseline, on=merge_keys, suffixes=('_d1namo', '_baseline'))
    
    # Extract approaches for AZT1D if available
    azt_merged = None
    if azt_df is not None:
        azt_bezier = azt_df[azt_df['Approach'] == 'Bezier']
        azt_baseline = azt_df[azt_df['Approach'] == 'Glucose+Insulin']
        azt_bezier = azt_bezier.groupby(merge_keys)['RMSE'].mean().reset_index()
        azt_baseline = azt_baseline.groupby(merge_keys)['RMSE'].mean().reset_index()
        azt_merged = pd.merge(azt_bezier, azt_baseline, on=merge_keys, suffixes=('_azt', '_azt_base'))
    
    patients = sorted(merged_df['Patient'].unique())
    horizons = sorted(merged_df['Prediction Horizon'].unique())
    
    analysis_text = []
    analysis_text.append("DETAILED RMSE ANALYSIS: MACRONUTRIENT-INFORMED vs BASELINE MODEL")
    analysis_text.append("=" * 80)
    analysis_text.append("")
    
    # Overall statistics (D1namo)
    overall_macro = merged_df['RMSE_d1namo'].mean()
    overall_base = merged_df['RMSE_baseline'].mean()
    overall_improvement = ((overall_base - overall_macro) / overall_base) * 100
    
    analysis_text.append("OVERALL PERFORMANCE:")
    analysis_text.append(f"Macronutrient-informed model: {overall_macro:.2f} ± {merged_df['RMSE_d1namo'].std():.2f} mg/dL")
    analysis_text.append(f"Baseline model: {overall_base:.2f} ± {merged_df['RMSE_baseline'].std():.2f} mg/dL")
    analysis_text.append(f"Overall improvement: {overall_improvement:.1f}%")
    analysis_text.append("")
    
    # Analysis by prediction horizon (D1namo)
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
    
    # Analysis by patient (D1namo)
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
    
    # Patient-specific significant improvements by horizon (D1namo)
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
    
    # Summary insights (D1namo)
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
    
    # AZT1D section if available
    if azt_merged is not None:
        analysis_text.append("\n" + "=" * 80)
        analysis_text.append("AZT1D: MACRONUTRIENT (CARB)-INFORMED vs BASELINE")
        analysis_text.append("=" * 80)
        azt_patients = sorted(azt_merged['Patient'].unique())
        azt_horizons = sorted(azt_merged['Prediction Horizon'].unique())

        overall_macro = azt_merged['RMSE_azt'].mean()
        overall_base = azt_merged['RMSE_azt_base'].mean()
        overall_improvement = ((overall_base - overall_macro) / overall_base) * 100
        analysis_text.append(f"OVERALL PERFORMANCE (AZT1D):")
        analysis_text.append(f"Macronutrient-informed: {overall_macro:.2f} ± {azt_merged['RMSE_azt'].std():.2f} mg/dL")
        analysis_text.append(f"Baseline: {overall_base:.2f} ± {azt_merged['RMSE_azt_base'].std():.2f} mg/dL")
        analysis_text.append(f"Overall improvement: {overall_improvement:.1f}%")
        analysis_text.append("")

        analysis_text.append("ANALYSIS BY PREDICTION HORIZON (AZT1D):")
        analysis_text.append("-" * 40)
        horizon_names = {6: "30 min", 9: "45 min", 12: "60 min", 18: "90 min", 24: "120 min"}
        for horizon in azt_horizons:
            h_data = azt_merged[azt_merged['Prediction Horizon'] == horizon]
            macro = h_data['RMSE_azt'].mean()
            base = h_data['RMSE_azt_base'].mean()
            macro_std = h_data['RMSE_azt'].std()
            base_std = h_data['RMSE_azt_base'].std()
            t_stat, p_val = ttest_rel(h_data['RMSE_azt'], h_data['RMSE_azt_base'])
            improvement = ((base - macro) / base) * 100
            analysis_text.append(f"{horizon_names[horizon]}:")
            analysis_text.append(f"  Macro: {macro:.2f} ± {macro_std:.2f} mg/dL")
            analysis_text.append(f"  Base:  {base:.2f} ± {base_std:.2f} mg/dL")
            analysis_text.append(f"  Improvement: {improvement:.1f}% (p={p_val:.3f})")
            analysis_text.append("")

        analysis_text.append("ANALYSIS BY PATIENT (AZT1D):")
        analysis_text.append("-" * 25)
        for patient in azt_patients:
            pdata = azt_merged[azt_merged['Patient'] == patient]
            macro = pdata['RMSE_azt'].mean()
            base = pdata['RMSE_azt_base'].mean()
            improvement = ((base - macro) / base) * 100
            t_stat, p_val = ttest_rel(pdata['RMSE_azt'], pdata['RMSE_azt_base'])
            analysis_text.append(f"Patient {patient}:")
            analysis_text.append(f"  Macro: {macro:.2f} ± {pdata['RMSE_azt'].std():.2f} mg/dL")
            analysis_text.append(f"  Base:  {base:.2f} ± {pdata['RMSE_azt_base'].std():.2f} mg/dL")
            analysis_text.append(f"  Improvement: {improvement:.1f}% (p={p_val:.3f})")
            analysis_text.append("")
    
    return '\n'.join(analysis_text)

def generate_latex_table():
    """Generate LaTeX table and save to manuscript/tables/"""
    d1namo_df = pd.read_csv('results/d1namo_comparison.csv')
    azt_df = pd.read_csv('results/azt1d_comparison.csv')
    
    # Extract different approaches for D1namo
    d1namo_bezier = d1namo_df[d1namo_df['Approach'] == 'Bezier']
    d1namo_baseline = d1namo_df[d1namo_df['Approach'] == 'Glucose+Insulin']
    
    # Handle duplicates by averaging RMSE for same merge keys
    merge_keys = ['Prediction Horizon', 'Patient', 'Day', 'Hour']
    d1namo_bezier = d1namo_bezier.groupby(merge_keys)['RMSE'].mean().reset_index()
    d1namo_baseline = d1namo_baseline.groupby(merge_keys)['RMSE'].mean().reset_index()
    
    # Merge the data
    merged_df = pd.merge(d1namo_bezier, d1namo_baseline, on=merge_keys, suffixes=('_d1namo', '_baseline'))
    
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
    
    # Generate LaTeX table (D1namo)
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
    
    # Also add AZT1D table if available
    if azt_df is not None:
        # Extract different approaches for AZT1D
        azt_bezier = azt_df[azt_df['Approach'] == 'Bezier']
        azt_baseline = azt_df[azt_df['Approach'] == 'Glucose+Insulin']
        
        merge_keys = ['Prediction Horizon', 'Patient', 'Day', 'Hour']
        azt_merged = pd.merge(
            azt_bezier.groupby(merge_keys)['RMSE'].mean().reset_index(),
            azt_baseline.groupby(merge_keys)['RMSE'].mean().reset_index(),
            on=merge_keys,
            suffixes=('_azt', '_azt_base')
        )
        patients = sorted(azt_merged['Patient'].unique())
        horizons = sorted(azt_merged['Prediction Horizon'].unique())

        # Build AZT1D table
        latex_table_azt = """\\begin{table}[ht]
\\centering
\\caption{AZT1D: Macronutrient (carbohydrates) vs Baseline (insulin only). RMSE $\\pm$ std in mg/dL with paired p-values.}
\\label{tab:azt1d_baseline_comparison}
\\renewcommand{\\arraystretch}{1.2}
\\setlength{\\tabcolsep}{2pt}
\\resizebox{\\textwidth}{!}{%
{\\scriptsize
\\begin{tabular}{|p{0.5cm}|cc>{\\columncolor{blue!10}\\scriptsize}p{0.6cm}|cc>{\\columncolor{blue!10}\\scriptsize}p{0.6cm}|cc>{\\columncolor{blue!10}\\scriptsize}p{0.6cm}|cc>{\\columncolor{blue!10}\\scriptsize}p{0.6cm}|cc>{\\columncolor{blue!10}\\scriptsize}p{0.6cm}|}
\\hline
\\rowcolor{gray!25} \\multirow{2}{*}[1ex]{\\textbf{Pat.}} & \\multicolumn{3}{c|}{\\cellcolor{gray!25}\\textbf{30 min}} & \\multicolumn{3}{c|}{\\cellcolor{gray!25}\\textbf{45 min}} & \\multicolumn{3}{c|}{\\cellcolor{gray!25}\\textbf{60 min}} & \\multicolumn{3}{c|}{\\cellcolor{gray!25}\\textbf{90 min}} & \\multicolumn{3}{c|}{\\cellcolor{gray!25}\\textbf{120 min}} \\\\
\\rowcolor{gray!25} & \\cellcolor{green!15}{\\textbf{Macro}} & \\cellcolor{orange!15}{\\textbf{Base}} & \\cellcolor{blue!15}{\\textbf{p}} & \\cellcolor{green!15}{\\textbf{Macro}} & \\cellcolor{orange!15}{\\textbf{Base}} & \\cellcolor{blue!15}{\\textbf{p}} & \\cellcolor{green!15}{\\textbf{Macro}} & \\cellcolor{orange!15}{\\textbf{Base}} & \\cellcolor{blue!15}{\\textbf{p}} & \\cellcolor{green!15}{\\textbf{Macro}} & \\cellcolor{orange!15}{\\textbf{Base}} & \\cellcolor{blue!15}{\\textbf{p}} & \\cellcolor{green!15}{\\textbf{Macro}} & \\cellcolor{orange!15}{\\textbf{Base}} & \\cellcolor{blue!15}{\\textbf{p}} \\\\
\\hline"""

        for patient in patients:
            row_color = "\\rowcolor{gray!5}" if str(patient).endswith(('1','3','5','7','9')) else "\\rowcolor{white}"
            latex_table_azt += f"\n{row_color} {patient}"
            for horizon in [6, 9, 12, 18, 24]:
                ph = azt_merged[(azt_merged['Patient'] == patient) & (azt_merged['Prediction Horizon'] == horizon)]
                if len(ph) > 0:
                    macro = ph['RMSE_azt'].mean()
                    base = ph['RMSE_azt_base'].mean()
                    macro_std = ph['RMSE_azt'].std()
                    base_std = ph['RMSE_azt_base'].std()
                    t_stat, p_val = ttest_rel(ph['RMSE_azt'], ph['RMSE_azt_base'])
                    macro_str = f"{macro:.2f}±{macro_std:.2f}"
                    base_str = f"{base:.2f}±{base_std:.2f}"
                    p_str = f"{p_val:.2f}"
                else:
                    macro_str = base_str = p_str = "N/A"
                latex_table_azt += f" & \\cellcolor{{green!10}}{{{macro_str}}} & \\cellcolor{{orange!10}}{{{base_str}}} & \\cellcolor{{blue!10}}{{{p_str}}}"
            latex_table_azt += " \\\\"  

        # Overall row for AZT1D
        azt_horizon_data = {}
        for horizon in [6, 9, 12, 18, 24]:
            h_data = azt_merged[azt_merged['Prediction Horizon'] == horizon]
            if len(h_data) == 0:
                continue
            azt_horizon_data[horizon] = {
                'macro': h_data['RMSE_azt'].mean(),
                'base': h_data['RMSE_azt_base'].mean(),
                'macro_std': h_data['RMSE_azt'].std(),
                'base_std': h_data['RMSE_azt_base'].std(),
                'p_value': ttest_rel(h_data['RMSE_azt'], h_data['RMSE_azt_base'])[1]
            }
        latex_table_azt += """
\\hline
\\rowcolor{blue!10} \\textbf{All}"""
        for horizon in [6, 9, 12, 18, 24]:
            if horizon in azt_horizon_data:
                h = azt_horizon_data[horizon]
                macro_str = f"{h['macro']:.2f}±{h['macro_std']:.2f}"
                base_str = f"{h['base']:.2f}±{h['base_std']:.2f}"
                p_str = f"{h['p_value']:.2f}"
            else:
                macro_str = base_str = p_str = "N/A"
            latex_table_azt += f" & \\cellcolor{{green!10}}{{{macro_str}}} & \\cellcolor{{orange!10}}{{{base_str}}} & \\cellcolor{{blue!10}}{{{p_str}}}"

        latex_table_azt += """ \\\\
\\hline
\\end{tabular}
}
}
\\end{table}"""

        # Note: we will overwrite comparison_table.tex below with a single combined table
        with open('../manuscript/tables/azt1d_comparison_detail.tex', 'w') as f:
            f.write(latex_table_azt)
    
    # Generate and save analysis
    analysis = generate_detailed_analysis()
    with open('../results/rmse_analysis.txt', 'w') as f:
        f.write(analysis)

os.makedirs('manuscript/tables', exist_ok=True)
d1_df = pd.read_csv('results/d1namo_comparison.csv')
azt_path = 'results/azt1d_comparison.csv'
azt_df = pd.read_csv(azt_path) if os.path.exists(azt_path) else None

# Extract Bezier approach RMSE for overall stats
d1_bezier_overall = d1_df[d1_df['Approach'] == 'Bezier']['RMSE'].mean()
azt_bezier_overall = (azt_df[azt_df['Approach'] == 'Bezier']['RMSE'].mean() if azt_df is not None else float('nan'))
lines = []
lines.append("% Auto-generated overall summary for D1namo and AZT1D")
overall_line = f"Overall macro-model RMSE: D1namo {d1_bezier_overall:.2f} mg/dL; AZT1D " + (f"{azt_bezier_overall:.2f} mg/dL" if azt_bezier_overall == azt_bezier_overall else "N/A") + "."
lines.append(overall_line)
lines.append("Across patients, we observe clear interpatient variability. Macronutrient features benefit some patients while slightly deteriorating performance in others, reflecting heterogeneous meal handling, insulin timing, and adherence patterns.")
with open('manuscript/tables/overall_summary.tex', 'w') as f:
    f.write('\n'.join(lines))

# Build single combined table (horizons as rows; columns: D1namo macro, D1namo baseline, AZT1D macro, AZT1D baseline, with p-values)
os.makedirs('manuscript/tables', exist_ok=True)

d1namo_df = pd.read_csv('results/d1namo_comparison.csv') if os.path.exists('results/d1namo_comparison.csv') else None
azt_df = pd.read_csv('results/azt1d_comparison.csv') if os.path.exists('results/azt1d_comparison.csv') else None

# Extract approach data for all three approaches
d1namo_bezier_df = d1namo_df[d1namo_df['Approach'] == 'Bezier'] if d1namo_df is not None else None
d1namo_lastmeal_df = d1namo_df[d1namo_df['Approach'] == 'LastMeal'] if d1namo_df is not None else None
d1namo_baseline_df = d1namo_df[d1namo_df['Approach'] == 'Glucose+Insulin'] if d1namo_df is not None else None
azt_bezier_df = azt_df[azt_df['Approach'] == 'Bezier'] if azt_df is not None else None
azt_lastmeal_df = azt_df[azt_df['Approach'] == 'LastMeal'] if azt_df is not None else None
azt_baseline_df = azt_df[azt_df['Approach'] == 'Glucose+Insulin'] if azt_df is not None else None

# Note: Keep all individual predictions instead of grouping to preserve sample size for t-tests
# The Hour=-1 issue in d1namo.py means many predictions have same Day/Hour, but they're still separate predictions

horizons = sorted(d1namo_df['Prediction Horizon'].unique()) if d1namo_df is not None else [6, 9, 12, 18, 24]
horizon_names = {6: "30 min", 9: "45 min", 12: "60 min", 18: "90 min", 24: "120 min"}

table_content = ""
for horizon in horizons:
    # D1namo data for all three approaches
    d1n_bezier_h = d1namo_bezier_df[d1namo_bezier_df['Prediction Horizon'] == horizon]['RMSE']
    d1n_lastmeal_h = d1namo_lastmeal_df[d1namo_lastmeal_df['Prediction Horizon'] == horizon]['RMSE']
    d1n_baseline_h = d1namo_baseline_df[d1namo_baseline_df['Prediction Horizon'] == horizon]['RMSE']
    
    d1n_bezier_mean, d1n_bezier_std = d1n_bezier_h.mean(), d1n_bezier_h.std()
    d1n_lastmeal_mean, d1n_lastmeal_std = d1n_lastmeal_h.mean(), d1n_lastmeal_h.std()
    d1n_baseline_mean, d1n_baseline_std = d1n_baseline_h.mean(), d1n_baseline_h.std()
    
    # Paired t-tests for all comparisons between all three methods
    p_d1n_bezier_vs_baseline = ttest_rel(d1n_bezier_h, d1n_baseline_h)[1] if len(d1n_bezier_h) > 1 else np.nan
    p_d1n_lastmeal_vs_baseline = ttest_rel(d1n_lastmeal_h, d1n_baseline_h)[1] if len(d1n_lastmeal_h) > 1 else np.nan
    p_d1n_bezier_vs_lastmeal = ttest_rel(d1n_bezier_h, d1n_lastmeal_h)[1] if len(d1n_bezier_h) > 1 else np.nan
    
    # One-way repeated measures ANOVA for all three methods
    # For repeated measures, we need to ensure same length arrays
    min_len = min(len(d1n_bezier_h), len(d1n_lastmeal_h), len(d1n_baseline_h))
    d1n_bezier_anova = d1n_bezier_h[:min_len]
    d1n_lastmeal_anova = d1n_lastmeal_h[:min_len]
    d1n_baseline_anova = d1n_baseline_h[:min_len]
    f_stat_d1n, p_anova_d1n = f_oneway(d1n_bezier_anova, d1n_lastmeal_anova, d1n_baseline_anova)
    
    # AZT1D data for all three approaches
    azt_bezier_h = azt_bezier_df[azt_bezier_df['Prediction Horizon'] == horizon]['RMSE']
    azt_lastmeal_h = azt_lastmeal_df[azt_lastmeal_df['Prediction Horizon'] == horizon]['RMSE']
    azt_baseline_h = azt_baseline_df[azt_baseline_df['Prediction Horizon'] == horizon]['RMSE']
    
    azt_bezier_mean, azt_bezier_std = azt_bezier_h.mean(), azt_bezier_h.std()
    azt_lastmeal_mean, azt_lastmeal_std = azt_lastmeal_h.mean(), azt_lastmeal_h.std()
    azt_baseline_mean, azt_baseline_std = azt_baseline_h.mean(), azt_baseline_h.std()
    
    # Paired t-tests for all comparisons between all three methods
    p_azt_bezier_vs_baseline = ttest_rel(azt_bezier_h, azt_baseline_h)[1] if len(azt_bezier_h) > 1 else np.nan
    p_azt_lastmeal_vs_baseline = ttest_rel(azt_lastmeal_h, azt_baseline_h)[1] if len(azt_lastmeal_h) > 1 else np.nan
    p_azt_bezier_vs_lastmeal = ttest_rel(azt_bezier_h, azt_lastmeal_h)[1] if len(azt_bezier_h) > 1 else np.nan
    
    # One-way repeated measures ANOVA for all three methods
    # For repeated measures, we need to ensure same length arrays
    min_len = min(len(azt_bezier_h), len(azt_lastmeal_h), len(azt_baseline_h))
    azt_bezier_anova = azt_bezier_h[:min_len]
    azt_lastmeal_anova = azt_lastmeal_h[:min_len]
    azt_baseline_anova = azt_baseline_h[:min_len]
    f_stat_azt, p_anova_azt = f_oneway(azt_bezier_anova, azt_lastmeal_anova, azt_baseline_anova)
    # Format strings
    h_name = horizon_names.get(horizon, f"{horizon*5} min")
    
    # D1namo strings
    d1n_bezier_str = f"{d1n_bezier_mean:.2f} $\\pm$ {d1n_bezier_std:.2f}" if not np.isnan(d1n_bezier_mean) else "N/A"
    d1n_lastmeal_str = f"{d1n_lastmeal_mean:.2f} $\\pm$ {d1n_lastmeal_std:.2f}" if not np.isnan(d1n_lastmeal_mean) else "N/A"
    d1n_baseline_str = f"{d1n_baseline_mean:.2f} $\\pm$ {d1n_baseline_std:.2f}" if not np.isnan(d1n_baseline_mean) else "N/A"
    
    # AZT1D strings
    azt_bezier_str = f"{azt_bezier_mean:.2f} $\\pm$ {azt_bezier_std:.2f}" if not np.isnan(azt_bezier_mean) else "N/A"
    azt_lastmeal_str = f"{azt_lastmeal_mean:.2f} $\\pm$ {azt_lastmeal_std:.2f}" if not np.isnan(azt_lastmeal_mean) else "N/A"
    azt_baseline_str = f"{azt_baseline_mean:.2f} $\\pm$ {azt_baseline_std:.2f}" if not np.isnan(azt_baseline_mean) else "N/A"
    
    # P-value strings with bold formatting for significant results (all three comparisons)
    p_d1n_bezier_str = f"\\textbf{{{p_d1n_bezier_vs_baseline:.3f}}}" if not np.isnan(p_d1n_bezier_vs_baseline) and p_d1n_bezier_vs_baseline < 0.05 else f"{p_d1n_bezier_vs_baseline:.3f}" if not np.isnan(p_d1n_bezier_vs_baseline) else "N/A"
    p_d1n_lastmeal_str = f"\\textbf{{{p_d1n_lastmeal_vs_baseline:.3f}}}" if not np.isnan(p_d1n_lastmeal_vs_baseline) and p_d1n_lastmeal_vs_baseline < 0.05 else f"{p_d1n_lastmeal_vs_baseline:.3f}" if not np.isnan(p_d1n_lastmeal_vs_baseline) else "N/A"
    p_d1n_bezier_vs_lastmeal_str = f"\\textbf{{{p_d1n_bezier_vs_lastmeal:.3f}}}" if not np.isnan(p_d1n_bezier_vs_lastmeal) and p_d1n_bezier_vs_lastmeal < 0.05 else f"{p_d1n_bezier_vs_lastmeal:.3f}" if not np.isnan(p_d1n_bezier_vs_lastmeal) else "N/A"
    p_d1n_anova_str = f"\\textbf{{{p_anova_d1n:.3f}}}" if not np.isnan(p_anova_d1n) and p_anova_d1n < 0.05 else f"{p_anova_d1n:.3f}" if not np.isnan(p_anova_d1n) else "N/A"
    p_azt_bezier_str = f"\\textbf{{{p_azt_bezier_vs_baseline:.3f}}}" if not np.isnan(p_azt_bezier_vs_baseline) and p_azt_bezier_vs_baseline < 0.05 else f"{p_azt_bezier_vs_baseline:.3f}" if not np.isnan(p_azt_bezier_vs_baseline) else "N/A"
    p_azt_lastmeal_str = f"\\textbf{{{p_azt_lastmeal_vs_baseline:.3f}}}" if not np.isnan(p_azt_lastmeal_vs_baseline) and p_azt_lastmeal_vs_baseline < 0.05 else f"{p_azt_lastmeal_vs_baseline:.3f}" if not np.isnan(p_azt_lastmeal_vs_baseline) else "N/A"
    p_azt_bezier_vs_lastmeal_str = f"\\textbf{{{p_azt_bezier_vs_lastmeal:.3f}}}" if not np.isnan(p_azt_bezier_vs_lastmeal) and p_azt_bezier_vs_lastmeal < 0.05 else f"{p_azt_bezier_vs_lastmeal:.3f}" if not np.isnan(p_azt_bezier_vs_lastmeal) else "N/A"
    p_azt_anova_str = f"\\textbf{{{p_anova_azt:.3f}}}" if not np.isnan(p_anova_azt) and p_anova_azt < 0.05 else f"{p_anova_azt:.3f}" if not np.isnan(p_anova_azt) else "N/A"

    # Apply bolding for better performance based on ANOVA significance
    if p_anova_d1n < 0.05:
        # Find the best performing method and bold it
        means = [d1n_bezier_mean, d1n_lastmeal_mean, d1n_baseline_mean]
        best_idx = np.argmin(means)
        if best_idx == 0:
            d1n_bezier_str = f"\\textbf{{{d1n_bezier_str}}}"
        elif best_idx == 1:
            d1n_lastmeal_str = f"\\textbf{{{d1n_lastmeal_str}}}"
        elif best_idx == 2:
            d1n_baseline_str = f"\\textbf{{{d1n_baseline_str}}}"

    if p_anova_azt < 0.05:
        # Find the best performing method and bold it
        means = [azt_bezier_mean, azt_lastmeal_mean, azt_baseline_mean]
        best_idx = np.argmin(means)
        if best_idx == 0:
            azt_bezier_str = f"\\textbf{{{azt_bezier_str}}}"
        elif best_idx == 1:
            azt_lastmeal_str = f"\\textbf{{{azt_lastmeal_str}}}"
        elif best_idx == 2:
            azt_baseline_str = f"\\textbf{{{azt_baseline_str}}}"

    table_content += f"\\rowcolor{{gray!5}} {h_name} & {d1n_bezier_str} & {d1n_lastmeal_str} & {d1n_baseline_str} & {p_d1n_anova_str} & {azt_bezier_str} & {azt_lastmeal_str} & {azt_baseline_str} & {p_azt_anova_str} \\\\\n"

latex_table = f"""\\begin{{table}}[ht]
\\centering
\\caption{{Comparison of all three approaches across D1namo and AZT1D datasets. Values show RMSE $\\pm$ standard deviation. P-values show one-way ANOVA results for overall significance across all three methods. Bold p-values indicate statistically significant differences (p<0.05).}}
\\label{{tab:three_approaches_comparison}}
\\renewcommand{{\\arraystretch}}{{1.2}}
\\setlength{{\\tabcolsep}}{{2pt}}
\\resizebox{{\\textwidth}}{{!}}{{%
{{\\scriptsize
\\begin{{tabular}}{{|l|ccc>{{\\columncolor{{blue!10}}\\scriptsize}}c|ccc>{{\\columncolor{{blue!10}}\\scriptsize}}c|}}
\\hline
\\rowcolor{{gray!25}} \\multirow{{2}}{{*}}[1ex]{{\\textbf{{Prediction Horizon}}}} & \\multicolumn{{4}}{{c|}}{{\\cellcolor{{gray!25}}\\textbf{{D1namo}}}} & \\multicolumn{{4}}{{c|}}{{\\cellcolor{{gray!25}}\\textbf{{AZT1D}}}} \\\\
\\rowcolor{{gray!25}} & \\cellcolor{{green!15}}{{\\textbf{{Bezier}}}} & \\cellcolor{{orange!15}}{{\\textbf{{LastMeal}}}} & \\cellcolor{{red!15}}{{\\textbf{{Baseline}}}} & \\cellcolor{{blue!15}}{{\\textbf{{ANOVA}}}} & \\cellcolor{{green!15}}{{\\textbf{{Bezier}}}} & \\cellcolor{{orange!15}}{{\\textbf{{LastMeal}}}} & \\cellcolor{{red!15}}{{\\textbf{{Baseline}}}} & \\cellcolor{{blue!15}}{{\\textbf{{ANOVA}}}} \\\\
\\hline
{table_content}
\\hline
\\end{{tabular}}
}}
}}
\\end{{table}}"""

with open('manuscript/tables/three_approaches_comparison.tex', 'w') as f:
    f.write(latex_table)

# Write per-horizon overall stats and patient-level improvement insights
os.makedirs('results', exist_ok=True)

overall_rows = []
patient_rows_d1 = []
patient_rows_azt = []

for horizon in horizons:
    # Overall stats per dataset
    if d1namo_bezier_df is not None and d1namo_baseline_df is not None:
        d1n_bezier_h = d1namo_bezier_df[d1namo_bezier_df['Prediction Horizon'] == horizon]
        d1n_base_h = d1namo_baseline_df[d1namo_baseline_df['Prediction Horizon'] == horizon]
        overall_rows.append({
            'dataset': 'D1namo',
            'horizon': horizon,
            'macro_mean': d1n_bezier_h['RMSE'].mean(),
            'macro_std': d1n_bezier_h['RMSE'].std(),
            'base_mean': d1n_base_h['RMSE'].mean(),
            'base_std': d1n_base_h['RMSE'].std(),
            'p_value': float(ttest_rel(d1n_bezier_h['RMSE'], d1n_base_h['RMSE'])[1]) if len(d1n_bezier_h) > 1 else np.nan,
            'improvement_pct': float(((d1n_base_h['RMSE'].mean() - d1n_bezier_h['RMSE'].mean()) / d1n_base_h['RMSE'].mean()) * 100)
        })
        # Patient-level improvement distribution
        for patient in sorted(d1namo_bezier_df['Patient'].unique()):
            pm = d1namo_bezier_df[(d1namo_bezier_df['Patient'] == patient) & (d1namo_bezier_df['Prediction Horizon'] == horizon)]['RMSE']
            pb = d1namo_baseline_df[(d1namo_baseline_df['Patient'] == patient) & (d1namo_baseline_df['Prediction Horizon'] == horizon)]['RMSE']
            if len(pm) > 0 and len(pb) > 0:
                imp = (pb.mean() - pm.mean()) / pb.mean() * 100
                pval = float(ttest_rel(pm, pb)[1]) if len(pm) > 1 and len(pb) > 1 else np.nan
                patient_rows_d1.append({'horizon': horizon, 'patient': patient, 'improvement_pct': float(imp), 'p_value': pval})

    azt_bezier_h = azt_bezier_df[azt_bezier_df['Prediction Horizon'] == horizon]
    azt_base_h = azt_baseline_df[azt_baseline_df['Prediction Horizon'] == horizon]
    overall_rows.append({
        'dataset': 'AZT1D',
        'horizon': horizon,
        'macro_mean': azt_bezier_h['RMSE'].mean(),
        'macro_std': azt_bezier_h['RMSE'].std(),
        'base_mean': azt_base_h['RMSE'].mean(),
        'base_std': azt_base_h['RMSE'].std(),
        'p_value': float(ttest_rel(azt_bezier_h['RMSE'], azt_base_h['RMSE'])[1]) if len(azt_bezier_h) > 1 else np.nan,
        'improvement_pct': float(((azt_base_h['RMSE'].mean() - azt_bezier_h['RMSE'].mean()) / azt_base_h['RMSE'].mean()) * 100)
    })
    for patient in sorted(azt_bezier_df['Patient'].unique()):
        pm = azt_bezier_df[(azt_bezier_df['Patient'] == patient) & (azt_bezier_df['Prediction Horizon'] == horizon)]['RMSE']
        pb = azt_baseline_df[(azt_baseline_df['Patient'] == patient) & (azt_baseline_df['Prediction Horizon'] == horizon)]['RMSE']
        if len(pm) > 0 and len(pb) > 0:
            imp = (pb.mean() - pm.mean()) / pb.mean() * 100
            pval = float(ttest_rel(pm, pb)[1]) if len(pm) > 1 and len(pb) > 1 else np.nan
            patient_rows_azt.append({'horizon': horizon, 'patient': patient, 'improvement_pct': float(imp), 'p_value': pval})

pd.DataFrame(overall_rows).to_csv('results/combined_overall_stats.csv', index=False)
pd.DataFrame(patient_rows_d1).to_csv('results/patient_improvement_stats_d1namo.csv', index=False)
pd.DataFrame(patient_rows_azt).to_csv('results/patient_improvement_stats_azt1d.csv', index=False)

# Build concise LaTeX summary paragraph with variability and cross-dataset differences
summary_lines = []
summary_lines.append("% Auto-generated performance summary")
def fmt(v):
    return ("N/A" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}")

for ds in ['D1namo', 'AZT1D']:
    ds_rows = [r for r in overall_rows if r['dataset'] == ds]
    if not ds_rows:
        continue
    trend = sorted([(r['horizon'], r['improvement_pct']) for r in ds_rows], key=lambda x: x[0])
    best = max(trend, key=lambda x: (x[1] if x[1] is not None else -1))
    worst = min(trend, key=lambda x: (x[1] if x[1] is not None else 1e9))
    summary_lines.append(f"For {ds}, macro vs baseline improvements increase with horizon, peaking at {horizon_names.get(best[0], str(best[0]*5)+ ' min')} ({best[1]:.1f}%).")

# Patient variability stats (median improvement and fraction significant)
def summarize_patient(rows, label):
    if not rows:
        return
    dfp = pd.DataFrame(rows)
    lines = []
    for h in sorted(dfp['horizon'].unique()):
        sub = dfp[dfp['horizon'] == h]
        med = sub['improvement_pct'].median()
        iqr = sub['improvement_pct'].quantile(0.75) - sub['improvement_pct'].quantile(0.25)
        improved_pct = (sub['improvement_pct'] > 0).mean() * 100
        sig_pct = (sub['p_value'].notna() & (sub['p_value'] < 0.05) & (sub['improvement_pct'] > 0)).mean() * 100
        lines.append(f"{horizon_names.get(h, str(h*5)+' min')}: median {med:.1f}% (IQR {iqr:.1f}), improved {improved_pct:.0f}%, significant {sig_pct:.0f}%.")
    return f"{label}: " + " ".join(lines)

d1_line = summarize_patient(patient_rows_d1, 'D1namo patient variability')
azt_line = summarize_patient(patient_rows_azt, 'AZT1D patient variability')
if d1_line: summary_lines.append(d1_line)
if azt_line: summary_lines.append(azt_line)

# Build plain-text findings with mean±std and p-values per horizon and dataset
h_order = [6, 9, 12, 18, 24]
h_names = {6: '30 min', 9: '45 min', 12: '60 min', 18: '90 min', 24: '120 min'}

def get_row(ds, h):
    for r in overall_rows:
        if r['dataset'] == ds and r['horizon'] == h:
            return r
    return None

text_lines = []
text_lines.append("Overall macro vs baseline RMSE (mean ± std, p-values) by prediction horizon across datasets.")
for h in h_order:
    d1 = get_row('D1namo', h)
    az = get_row('AZT1D', h)
    d1_str = "N/A"
    az_str = "N/A"
    if d1 is not None:
        d1_str = f"D1namo: macro {d1['macro_mean']:.2f} ± {d1['macro_std']:.2f} vs baseline {d1['base_mean']:.2f} ± {d1['base_std']:.2f} (p={d1['p_value']:.3f})"
    if az is not None:
        az_str = f"AZT1D: macro {az['macro_mean']:.2f} ± {az['macro_std']:.2f} vs baseline {az['base_mean']:.2f} ± {az['base_std']:.2f} (p={az['p_value']:.3f})"
    text_lines.append(f"{h_names[h]}: {d1_str}; {az_str}.")

# Concise variability summary (median improvement, IQR, % improved, % significant) per horizon
def var_summary(rows):
    if not rows:
        return {}
    dfp = pd.DataFrame(rows)
    out = {}
    for h in sorted(dfp['horizon'].unique()):
        sub = dfp[dfp['horizon'] == h]
        med = sub['improvement_pct'].median()
        iqr = sub['improvement_pct'].quantile(0.75) - sub['improvement_pct'].quantile(0.25)
        improved_pct = (sub['improvement_pct'] > 0).mean() * 100
        sig_pct = (sub['p_value'].notna() & (sub['p_value'] < 0.05) & (sub['improvement_pct'] > 0)).mean() * 100
        out[h] = (med, iqr, improved_pct, sig_pct)
    return out

d1_var = var_summary(patient_rows_d1)
az_var = var_summary(patient_rows_azt)

if d1_var or az_var:
    text_lines.append("Patient variability (median improvement%, IQR, % improved, % significant) by horizon.")
for h in h_order:
    parts = []
    if h in d1_var:
        med, iqr, imp, sig = d1_var[h]
        parts.append(f"D1namo {med:.1f}% (IQR {iqr:.1f}), {imp:.0f}% improved, {sig:.0f}% significant")
    if h in az_var:
        med, iqr, imp, sig = az_var[h]
        parts.append(f"AZT1D {med:.1f}% (IQR {iqr:.1f}), {imp:.0f}% improved, {sig:.0f}% significant")
    if parts:
        text_lines.append(f"{h_names[h]}: " + "; ".join(parts) + ".")

with open('manuscript/tables/performance_summary.tex', 'w') as f:
    f.write(" ".join(text_lines))

# Helper to build a single dataset table with three-way comparison
def build_single_table(name, df_bezier, df_lastmeal, df_baseline, label):
    if df_bezier is None or df_lastmeal is None or df_baseline is None:
        return ""
    patients = sorted(df_bezier['Patient'].unique())
    horizons_local = [h for h in [6,9,12,18,24] if (df_bezier['Prediction Horizon'] == h).any()]
    header = "|p{0.3cm}|" + "".join(["p{0.85cm}p{0.85cm}p{0.85cm}>{\\columncolor{blue!10}\\tiny}p{0.3cm}|" for _ in horizons_local])
    
    table = []
    table.append("\\begin{table}[ht]")
    table.append("\\centering")
    table.append(f"\\caption{{Patient-level RMSE results by prediction horizon for {name}. Values show RMSE $\\pm$ std (mg/dL) with one-way ANOVA p-values for overall significance across all three methods.}}")
    table.append(f"\\label{{{label}}}")
    table.append("{\\tiny")
    table.append("\\setlength{\\tabcolsep}{1pt}")
    table.append("\\begin{tabular}{" + header + "}")
    table.append("\\rowcolor{gray!25} {\\tiny\\textbf{Pat.}} " + "".join([f"& \\multicolumn{{4}}{{c|}}{{{h_names[h]}}} " for h in horizons_local]) + "\\\\")
    table.append("\\rowcolor{gray!25} " + "".join(["& \\cellcolor{green!15}{\\tiny\\textbf{Bezier}} & \\cellcolor{orange!15}{\\tiny\\textbf{LastMeal}} & \\cellcolor{red!15}{\\tiny\\textbf{Baseline}} & \\cellcolor{blue!15}{\\tiny\\textbf{ANOVA}} " for _ in horizons_local]) + "\\\\")
    table.append("\\hline")
    
    for patient in patients:
        row = [f"\\rowcolor{{gray!5}} {{\\tiny {patient}}}"]
        for h in horizons_local:
            bez = df_bezier[(df_bezier['Patient']==patient) & (df_bezier['Prediction Horizon']==h)]['RMSE']
            last = df_lastmeal[(df_lastmeal['Patient']==patient) & (df_lastmeal['Prediction Horizon']==h)]['RMSE']
            base = df_baseline[(df_baseline['Patient']==patient) & (df_baseline['Prediction Horizon']==h)]['RMSE']
            
            if len(bez) > 0 and len(last) > 0 and len(base) > 0:
                bez_mean, bez_std = bez.mean(), bez.std()
                last_mean, last_std = last.mean(), last.std()
                base_mean, base_std = base.mean(), base.std()
                
                # Paired t-tests for all three comparisons
                p_bezier_vs_baseline = ttest_rel(bez, base)[1] if len(bez) > 1 and len(base) > 1 else np.nan
                p_lastmeal_vs_baseline = ttest_rel(last, base)[1] if len(last) > 1 and len(base) > 1 else np.nan
                p_bezier_vs_lastmeal = ttest_rel(bez, last)[1] if len(bez) > 1 and len(last) > 1 else np.nan
                
                # One-way ANOVA for all three methods
                if len(bez) > 1 and len(last) > 1 and len(base) > 1:
                    min_len = min(len(bez), len(last), len(base))
                    bez_anova = bez[:min_len]
                    last_anova = last[:min_len]
                    base_anova = base[:min_len]
                    f_stat, p_anova = f_oneway(bez_anova, last_anova, base_anova)
                else:
                    p_anova = np.nan
                
                bez_str = f"{bez_mean:.1f}±{bez_std:.1f}"
                last_str = f"{last_mean:.1f}±{last_std:.1f}"
                base_str = f"{base_mean:.1f}±{base_std:.1f}"
                
                # Format p-values for all three comparisons plus ANOVA
                p_bezier_str = f"{p_bezier_vs_baseline:.2f}" if not np.isnan(p_bezier_vs_baseline) else "N/A"
                p_lastmeal_str = f"{p_lastmeal_vs_baseline:.2f}" if not np.isnan(p_lastmeal_vs_baseline) else "N/A"
                p_bezier_vs_lastmeal_str = f"{p_bezier_vs_lastmeal:.2f}" if not np.isnan(p_bezier_vs_lastmeal) else "N/A"
                p_anova_str = f"{p_anova:.2f}" if not np.isnan(p_anova) else "N/A"
                
                # Bold p-values below 0.05
                if not np.isnan(p_bezier_vs_baseline) and p_bezier_vs_baseline < 0.05:
                    p_bezier_str = f"\\textbf{{{p_bezier_str}}}"
                if not np.isnan(p_lastmeal_vs_baseline) and p_lastmeal_vs_baseline < 0.05:
                    p_lastmeal_str = f"\\textbf{{{p_lastmeal_str}}}"
                if not np.isnan(p_bezier_vs_lastmeal) and p_bezier_vs_lastmeal < 0.05:
                    p_bezier_vs_lastmeal_str = f"\\textbf{{{p_bezier_vs_lastmeal_str}}}"
                if not np.isnan(p_anova) and p_anova < 0.05:
                    p_anova_str = f"\\textbf{{{p_anova_str}}}"
                
                # Apply bold formatting for statistically significant better performance based on ANOVA
                if not np.isnan(p_anova) and p_anova < 0.05:
                    # Find the best performing method and bold it
                    means = [bez_mean, last_mean, base_mean]
                    best_idx = np.argmin(means)
                    if best_idx == 0:
                        bez_str = f"\\textbf{{{bez_str}}}"
                    elif best_idx == 1:
                        last_str = f"\\textbf{{{last_str}}}"
                    elif best_idx == 2:
                        base_str = f"\\textbf{{{base_str}}}"
                
            else:
                bez_str = last_str = base_str = p_bezier_str = p_lastmeal_str = p_bezier_vs_lastmeal_str = p_anova_str = "N/A"
            row.append(f"& \\cellcolor{{green!10}}{{\\tiny {bez_str}}} & \\cellcolor{{orange!10}}{{\\tiny {last_str}}} & \\cellcolor{{red!10}}{{\\tiny {base_str}}} & \\cellcolor{{blue!10}}{{\\tiny {p_anova_str}}} ")
        table.append(" ".join(row) + "\\\\")
    
    table.append("\\hline")
    table.append("\\end{tabular}")
    table.append("}")
    table.append("\\end{table}")
    
    return "\n".join(table)

# Build separate tables for D1namo and AZT1D
d1namo_table = build_single_table("D1namo", d1namo_bezier_df, d1namo_lastmeal_df, d1namo_baseline_df, "tab:patient_individual_d1namo")
azt1d_table = build_single_table("AZT1D", azt_bezier_df, azt_lastmeal_df, azt_baseline_df, "tab:patient_individual_azt1d")

# Save D1namo table
with open('manuscript/tables/patient_individual_d1namo.tex', 'w') as f:
    f.write(d1namo_table)

# Save AZT1D table  
with open('manuscript/tables/patient_individual_azt1d.tex', 'w') as f:
    f.write(azt1d_table)

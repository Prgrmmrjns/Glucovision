import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from params import *
from processing_functions import *

# Load Bézier parameters
with open('results/bezier_params/d1namo_all_patient_bezier_params.json', 'r') as f:
    all_patient_params = json.load(f)

# Calculate global parameters as average across all patients for visualization
global_params = {}
features = ['simple_sugars', 'complex_sugars', 'proteins', 'fats', 'dietary_fibers', 'insulin']

for feature in features:
    feature_params = []
    for patient_key in all_patient_params.keys():
        if feature in all_patient_params[patient_key]:
            feature_params.append(all_patient_params[patient_key][feature])
    
    if feature_params:
        # Average the parameters across all patients
        global_params[feature] = np.mean(feature_params, axis=0).tolist()

# Load data from patient 001
patient = '001'
glucose_data, combined_data = get_d1namo_data(patient)

# Add macronutrient features using Bézier curves
patient_data = add_temporal_features(global_params, OPTIMIZATION_FEATURES_D1NAMO, glucose_data, combined_data, prediction_horizon=0)

# Create figure with custom width ratios (1:2 for 1/3:2/3)
fig = plt.figure(figsize=(16, 6))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 3], hspace=0.3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

# Colors and names for consistency
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
feature_names = ['Simple Sugars', 'Complex Sugars', 'Proteins', 'Fats', 'Dietary Fibers', 'Insulin']

# Subplot A: Bézier Curves (1/3 width)
ax1.text(-0.18, 0.95, 'A', transform=ax1.transAxes, fontsize=16, va='top', fontweight='bold')

max_x = 0
for i, (feature, params) in enumerate(global_params.items()):
    points = np.array(params).reshape(-1, 2)
    curve = bezier_curve(points)
    max_x = max(max_x, curve[:, 0].max())
    
    ax1.plot(curve[:, 0], curve[:, 1], color=colors[i], linewidth=3, 
             label=feature_names[i], alpha=0.8)

ax1.set_xlabel('Time After Consumption (hours)', fontsize=12)
ax1.set_ylabel('Influence Weight', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-0.1, max_x + 0.3)
ax1.set_ylim(-0.05, 1.05)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Subplot B: Feature modeling over time (2/3 width)
ax2.text(-0.13, 0.95, 'B', transform=ax2.transAxes, fontsize=16, va='top', fontweight='bold')

# Select a representative time period (first 3 days)
start_time = patient_data['datetime'].min()
end_time = start_time + pd.Timedelta(days=3)
plot_data = patient_data[(patient_data['datetime'] >= start_time) & (patient_data['datetime'] <= end_time)].copy()

# Plot glucose on primary y-axis
line_glucose = ax2.plot(plot_data['datetime'], plot_data['glucose_12'], 'k-', linewidth=3, 
                       label='Glucose', alpha=0.9)
ax2.set_ylabel('Change in Glucose (mg/dL) over 60 minutes', fontsize=12)
ax2.set_xlabel('Time', fontsize=12)
ax2.grid(True, alpha=0.3)

# Create secondary y-axis for macronutrient features
ax2_twin = ax2.twinx()

lines_features = []
for i, feature in enumerate(OPTIMIZATION_FEATURES_D1NAMO):
    if feature in plot_data.columns:
        line = ax2_twin.plot(plot_data['datetime'], plot_data[feature], color=colors[i], 
                           linewidth=2, label=feature_names[i], alpha=0.7)
        lines_features.extend(line)

ax2_twin.set_ylabel('Feature Values (Modeled Influence)', fontsize=12)

# Add food/insulin event markers
for idx, row in combined_data.iterrows():
    if start_time <= row['datetime'] <= end_time:
        has_food = any(row[f] > 0 for f in ['simple_sugars', 'complex_sugars', 'proteins', 'fats', 'dietary_fibers'])
        has_insulin = row.get('insulin', 0) > 0 or row.get('slow_insulin', 0) > 0 or row.get('fast_insulin', 0) > 0
        if has_food:
            ax2.axvline(x=row['datetime'], color='red', linestyle='--', alpha=0.5, linewidth=1)
        if has_insulin:
            ax2.axvline(x=row['datetime'], color='blue', linestyle=':', alpha=0.7, linewidth=1)

# Create shared legend at bottom center
legend_elements = ([Line2D([0], [0], color=colors[i], linewidth=2, label=feature_names[i]) for i in range(len(feature_names))] +
                  [Line2D([0], [0], color='black', linewidth=3, label='Glucose')] +
                  [Line2D([0], [0], color='red', linestyle='--', alpha=0.5, label='Food Events')] +
                  [Line2D([0], [0], color='blue', linestyle=':', alpha=0.7, label='Insulin Events')])

fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
           ncol=4, fontsize=10, frameon=False)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # Make room for legend
plt.savefig('manuscript/images/results/combined_metabolic_visualization.eps', dpi=300, bbox_inches='tight')
plt.close()

# Extract peak times for analysis
peak_times = {}
for feature, params in global_params.items():
    points = np.array(params).reshape(-1, 2)
    curve = bezier_curve(points)
    peak_idx = np.argmax(curve[:, 1])
    peak_times[feature] = curve[peak_idx, 0]

print("\nPeak times for each feature:")
for feature, peak_time in peak_times.items():
    print(f"{feature}: {peak_time:.2f} hours")

# ==========================
# Patient metabolism insights
# ==========================

# Helper utilities
def _params_to_curve_points(param_list, num=128):
    pts = np.array(param_list, dtype=float).reshape(-1, 2)
    return bezier_curve(pts, num=num)

def _peak_time_and_height(curve):
    idx = int(np.argmax(curve[:, 1]))
    return float(curve[idx, 0]), float(curve[idx, 1])

def _end_time_x3(param_list):
    pts = np.array(param_list, dtype=float).reshape(-1, 2)
    return float(pts[-1, 0])

def _summarize_patient_beziers(json_path, feature_keys):
    with open(json_path, 'r') as f:
        per_patient = json.load(f)
    rows = []
    for patient_key, fmap in per_patient.items():
        patient = patient_key.split('_')[-1]
        for feature in feature_keys:
            if feature not in fmap:
                continue
            params = fmap[feature]
            curve = _params_to_curve_points(params, num=128)
            t_peak, h_peak = _peak_time_and_height(curve)
            t_end = _end_time_x3(params)
            rows.append({
                'patient': str(patient),
                'feature': feature,
                'peak_time_h': t_peak,
                'peak_height': h_peak,
                'duration_h': t_end
            })
    return pd.DataFrame(rows)

def _patient_improvement_at_60(results_csv):
    df = pd.read_csv(results_csv)
    df = df.groupby(['Prediction Horizon', 'Patient', 'Approach'])['RMSE'].mean().reset_index()
    df60 = df[df['Prediction Horizon'] == 12].copy()
    df60['Patient'] = df60['Patient'].astype(str)
    base = df60[df60['Approach'] == 'Glucose+Insulin'][['Patient', 'RMSE']].rename(columns={'RMSE': 'rmse_base'})
    bez = df60[df60['Approach'] == 'Bezier'][['Patient', 'RMSE']].rename(columns={'RMSE': 'rmse_bezier'})
    merged = pd.merge(base, bez, on='Patient', how='inner')
    if len(merged) == 0:
        # try zero-filled merge as fallback
        df60['Patient_z'] = df60['Patient'].str.zfill(3)
        base = df60[df60['Approach'] == 'Glucose+Insulin'][['Patient_z', 'RMSE']].rename(columns={'RMSE': 'rmse_base'})
        bez = df60[df60['Approach'] == 'Bezier'][['Patient_z', 'RMSE']].rename(columns={'RMSE': 'rmse_bezier'})
        merged = pd.merge(base, bez, on='Patient_z', how='inner').rename(columns={'Patient_z': 'Patient'})
    merged['improvement'] = merged['rmse_base'] - merged['rmse_bezier']
    merged['improvement_pct'] = 100.0 * merged['improvement'] / merged['rmse_base']
    return merged

def _compute_correlations(bezier_df, improvements_df, feature_map):
    from scipy.stats import pearsonr
    corrs = []
    for name, fkey in feature_map.items():
        sub = bezier_df[bezier_df['feature'] == fkey].copy()
        sub['Patient'] = sub['patient'].astype(str)
        merged = pd.merge(sub, improvements_df, on='Patient', how='inner')
        if len(merged) >= 3:
            for metric in ['peak_time_h', 'peak_height', 'duration_h']:
                try:
                    r, p = pearsonr(merged[metric].values, merged['improvement'].values)
                    corrs.append({'feature': name, 'metric': metric, 'r': r, 'p': p, 'n': len(merged)})
                except Exception:
                    pass
    return pd.DataFrame(corrs)

# Compute and save insights for D1namo and AZT1D
try:
    d1_features = ['simple_sugars', 'complex_sugars', 'proteins', 'fats', 'dietary_fibers', 'insulin']
    azt_features = ['carbohydrates', 'insulin']

    d1_bez = _summarize_patient_beziers('results/bezier_params/d1namo_all_patient_bezier_params.json', d1_features)
    azt_bez = _summarize_patient_beziers('results/bezier_params/azt1d_all_patient_bezier_params.json', azt_features)

    d1_imp = _patient_improvement_at_60('results/d1namo_comparison.csv')
    azt_imp = _patient_improvement_at_60('results/azt1d_comparison.csv')

    d1_corr = _compute_correlations(d1_bez, d1_imp, {
        'Simple sugars': 'simple_sugars',
        'Complex sugars': 'complex_sugars',
        'Proteins': 'proteins',
        'Fats': 'fats',
        'Dietary fibers': 'dietary_fibers',
        'Insulin': 'insulin'
    })
    azt_corr = _compute_correlations(azt_bez, azt_imp, {
        'Carbohydrates': 'carbohydrates',
        'Insulin': 'insulin'
    })

    # Alignment: insulin peak time minus sugar/carbohydrate peak time
    def _alignment_df(bez_df, sugar_key):
        pivot = bez_df.pivot_table(index='patient', columns='feature', values='peak_time_h', aggfunc='mean')
        if 'insulin' in pivot.columns and sugar_key in pivot.columns:
            out = pivot[['insulin', sugar_key]].dropna().copy()
            out['alignment_h'] = out['insulin'] - out[sugar_key]
            out = out.reset_index().rename(columns={'patient': 'Patient'})
            out['Patient'] = out['Patient'].astype(str)
            return out
        return pd.DataFrame()

    d1_align = _alignment_df(d1_bez, 'simple_sugars')
    azt_align = _alignment_df(azt_bez, 'carbohydrates')

    from scipy.stats import pearsonr
    d1_align_res = None
    azt_align_res = None
    if len(d1_align):
        d1_m = pd.merge(d1_align, d1_imp, on='Patient', how='inner')
        if len(d1_m) >= 3:
            d1_align_res = pearsonr(d1_m['alignment_h'].values, d1_m['improvement'].values) + (len(d1_m),)
    if len(azt_align):
        a_m = pd.merge(azt_align, azt_imp, on='Patient', how='inner')
        if len(a_m) >= 3:
            azt_align_res = pearsonr(a_m['alignment_h'].values, a_m['improvement'].values) + (len(a_m),)

    # Summaries
    d1_summary = d1_bez.groupby('feature').agg(
        peak_time_h_mean=('peak_time_h', 'mean'), peak_time_h_std=('peak_time_h', 'std'),
        duration_h_mean=('duration_h', 'mean'), duration_h_std=('duration_h', 'std'),
        peak_height_mean=('peak_height', 'mean'), peak_height_std=('peak_height', 'std')
    ).reset_index()
    azt_summary = azt_bez.groupby('feature').agg(
        peak_time_h_mean=('peak_time_h', 'mean'), peak_time_h_std=('peak_time_h', 'std'),
        duration_h_mean=('duration_h', 'mean'), duration_h_std=('duration_h', 'std'),
        peak_height_mean=('peak_height', 'mean'), peak_height_std=('peak_height', 'std')
    ).reset_index()

    # Save outputs
    d1_corr.to_csv('results/d1namo_bezier_correlations.csv', index=False)
    azt_corr.to_csv('results/azt1d_bezier_correlations.csv', index=False)
    d1_summary.to_csv('results/d1namo_bezier_temporal_summary.csv', index=False)
    azt_summary.to_csv('results/azt1d_bezier_temporal_summary.csv', index=False)

    # Human-readable insights
    with open('results/new_insights_from_combined.txt', 'w') as f:
        f.write('D1namo temporal summary (mean±std):\n')
        if not d1_summary.empty:
            for _, row in d1_summary.iterrows():
                f.write(f"- {row['feature']}: peak {row['peak_time_h_mean']:.2f}±{row['peak_time_h_std']:.2f} h; duration {row['duration_h_mean']:.2f}±{row['duration_h_std']:.2f} h; peak height {row['peak_height_mean']:.2f}±{row['peak_height_std']:.2f}\n")
        f.write('\nAZT1D temporal summary (mean±std):\n')
        if not azt_summary.empty:
            for _, row in azt_summary.iterrows():
                f.write(f"- {row['feature']}: peak {row['peak_time_h_mean']:.2f}±{row['peak_time_h_std']:.2f} h; duration {row['duration_h_mean']:.2f}±{row['duration_h_std']:.2f} h; peak height {row['peak_height_mean']:.2f}±{row['peak_height_std']:.2f}\n")
        f.write('\nD1namo correlations (improvement vs temporal metrics):\n')
        if not d1_corr.empty and set(['feature','metric']).issubset(d1_corr.columns):
            for _, r in d1_corr.sort_values(['feature', 'metric']).iterrows():
                f.write(f"- {r['feature']} {r['metric']}: r={r['r']:.2f}, p={r['p']:.3f}, n={int(r['n'])}\n")
        f.write('\nAZT1D correlations (improvement vs temporal metrics):\n')
        if not azt_corr.empty and set(['feature','metric']).issubset(azt_corr.columns):
            for _, r in azt_corr.sort_values(['feature', 'metric']).iterrows():
                f.write(f"- {r['feature']} {r['metric']}: r={r['r']:.2f}, p={r['p']:.3f}, n={int(r['n'])}\n")
        if d1_align_res:
            f.write(f"\nD1namo alignment (insulin - simple sugars) vs improvement: r={d1_align_res[0]:.2f}, p={d1_align_res[1]:.3f}, n={d1_align_res[2]}\n")
        if azt_align_res:
            f.write(f"AZT1D alignment (insulin - carbohydrates) vs improvement: r={azt_align_res[0]:.2f}, p={azt_align_res[1]:.3f}, n={azt_align_res[2]}\n")

    # LaTeX-ready paragraph
    def _find_row(df, key):
        m = df[df['feature'] == key]
        return None if m.empty else m.iloc[0]

    para_lines = []
    d1_ss = _find_row(d1_summary, 'simple_sugars')
    d1_cs = _find_row(d1_summary, 'complex_sugars')
    d1_pr = _find_row(d1_summary, 'proteins')
    d1_ft = _find_row(d1_summary, 'fats')
    d1_df = _find_row(d1_summary, 'dietary_fibers')
    d1_in = _find_row(d1_summary, 'insulin')
    if all(x is not None for x in [d1_ss, d1_cs, d1_pr, d1_ft, d1_df, d1_in]):
        para_lines.append(
            f"Across D1namo patients, simple sugars peak at {d1_ss['peak_time_h_mean']:.2f}±{d1_ss['peak_time_h_std']:.2f} h, complex sugars at {d1_cs['peak_time_h_mean']:.2f}±{d1_cs['peak_time_h_std']:.2f} h, fats at {d1_ft['peak_time_h_mean']:.2f}±{d1_ft['peak_time_h_std']:.2f} h, proteins at {d1_pr['peak_time_h_mean']:.2f}±{d1_pr['peak_time_h_std']:.2f} h, dietary fibers at {d1_df['peak_time_h_mean']:.2f}±{d1_df['peak_time_h_std']:.2f} h, and insulin at {d1_in['peak_time_h_mean']:.2f}±{d1_in['peak_time_h_std']:.2f} h."
        )
    azt_cb = _find_row(azt_summary, 'carbohydrates')
    azt_in = _find_row(azt_summary, 'insulin')
    if azt_cb is not None and azt_in is not None:
        para_lines.append(
            f"On AZT1D, carbohydrates peak at {azt_cb['peak_time_h_mean']:.2f}±{azt_cb['peak_time_h_std']:.2f} h and insulin at {azt_in['peak_time_h_mean']:.2f}±{azt_in['peak_time_h_std']:.2f} h."
        )
    if not d1_corr.empty and 'p' in d1_corr.columns:
        sig_d1 = d1_corr[d1_corr['p'] < 0.05].sort_values('p').head(2)
        for _, r in sig_d1.iterrows():
            para_lines.append(
                f"In D1namo, improvement at 60 min correlates with {r['feature']} {r['metric'].replace('_',' ')} (r={r['r']:.2f}, p={r['p']:.3f})."
            )
    if not azt_corr.empty and 'p' in azt_corr.columns:
        sig_azt = azt_corr[azt_corr['p'] < 0.05].sort_values('p').head(2)
        for _, r in sig_azt.iterrows():
            para_lines.append(
                f"In AZT1D, improvement at 60 min correlates with {r['feature']} {r['metric'].replace('_',' ')} (r={r['r']:.2f}, p={r['p']:.3f})."
            )
    if d1_align_res and d1_align_res[1] < 0.05:
        para_lines.append(
            f"Alignment between insulin and simple sugar peaks (insulin earlier than sugars; negative alignment) is associated with larger improvements on D1namo (r={d1_align_res[0]:.2f}, p={d1_align_res[1]:.3f})."
        )
    if azt_align_res and azt_align_res[1] < 0.05:
        para_lines.append(
            f"On AZT1D, insulin–carbohydrate peak alignment shows an association with improvement (r={azt_align_res[0]:.2f}, p={azt_align_res[1]:.3f})."
        )
    with open('results/new_insights_latex_from_combined.txt', 'w') as f:
        f.write(' '.join(para_lines) + '\n')
except Exception as e:
    print(f"[Insights] Skipped due to error: {e}")

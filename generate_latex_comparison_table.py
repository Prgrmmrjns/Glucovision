import pandas as pd
import os

# Create the latex_tables directory if it doesn't exist
os.makedirs('latex_tables', exist_ok=True)

# Load the raw data files
df_baseline = pd.read_csv('results/evaluation_metrics_baseline.csv')
df_mlm = pd.read_csv('results/evaluation_metrics.csv')

# Rename 'dietitian' to 'Dietitian' and 'pixtral-large-latest' to 'mLLM'
df_baseline['Approach'] = df_baseline['Approach'].replace('dietitian', 'Dietitian')
df_mlm['Approach'] = df_mlm['Approach'].replace('pixtral-large-latest', 'mLLM')

# Combine the data
combined_df = pd.concat([df_baseline, df_mlm])

# Convert Patient to string
combined_df['Patient'] = combined_df['Patient'].astype(str)

# Ensure unique patient identifiers are properly padded
combined_df['Patient'] = combined_df['Patient'].apply(lambda x: x.zfill(3))

# Calculate mean and std RMSE for each approach, prediction horizon, and patient
pivot_mean = combined_df.groupby(['Patient', 'Approach', 'Prediction Horizon'])['RMSE'].mean().unstack(level=[1, 2])
pivot_std = combined_df.groupby(['Patient', 'Approach', 'Prediction Horizon'])['RMSE'].std().unstack(level=[1, 2])

# Calculate overall mean and std for each approach and prediction horizon
overall_mean = combined_df.groupby(['Approach', 'Prediction Horizon'])['RMSE'].mean().reset_index()
overall_std = combined_df.groupby(['Approach', 'Prediction Horizon'])['RMSE'].std().reset_index()

# Create dictionaries for easier access
avg_values = {}
std_values = {}
for approach in ['mLLM', 'Dietitian']:
    for horizon in [6, 9, 12, 18, 24]:
        # Filter data for this approach and horizon
        filtered_mean = overall_mean[(overall_mean['Approach'] == approach) & 
                          (overall_mean['Prediction Horizon'] == horizon)]
        filtered_std = overall_std[(overall_std['Approach'] == approach) & 
                          (overall_std['Prediction Horizon'] == horizon)]
        
        # Check if data exists for this combination
        if not filtered_mean.empty and not filtered_std.empty:
            avg_values[(approach, horizon)] = filtered_mean['RMSE'].values[0]
            std_values[(approach, horizon)] = filtered_std['RMSE'].values[0]
        else:
            avg_values[(approach, horizon)] = float('nan')
            std_values[(approach, horizon)] = float('nan')

# Format the LaTeX table with nice styling
styled_latex_template = r"""\begin{table}[ht]
\centering
\caption{RMSE by Patient, Approach, and Prediction Horizon (mg/dL)}
\label{tab:performance_table}
\renewcommand{\arraystretch}{1.3}
\setlength{\tabcolsep}{6pt}
\resizebox{\textwidth}{!}{%%
\begin{tabular}{|l|%s|}
\hline
\rowcolor{gray!25} \multirow{2}{*}[1ex]{\textbf{Patient}} & %s \\
\rowcolor{gray!25} & %s \\
\hline
%s
\hline
\rowcolor{blue!10} \textbf{Average} & %s \\
\hline
\end{tabular}
}
\end{table}
"""

# Create column headers
approaches = ['mLLM', 'Dietitian']
horizons = [6, 9, 12, 18, 24]

# Format headers for LaTeX
column_format = "|".join(["cc" for _ in horizons])
header_parts = []
for h in horizons:
    header_parts.append(r"\multicolumn{2}{c|}{\cellcolor{gray!25}\textbf{Horizon " + str(h*5) + " Min" + "}}")
header_row = " & ".join(header_parts)
subheader_parts = []
for _ in horizons:
    subheader_parts.append(r"\cellcolor{green!10}{\textbf{mLLM}} & \cellcolor{orange!10}{\textbf{Dietitian}}")
subheader_row = " & ".join(subheader_parts)

# Format data rows with background colors and bold for better values
rows = []
for patient in sorted(pivot_mean.index):
    row_data = []
    for horizon in horizons:
        # Skip if data for this patient-horizon-approach combination doesn't exist
        if ('mLLM', horizon) not in pivot_mean.loc[patient] or ('Dietitian', horizon) not in pivot_mean.loc[patient]:
            row_data.append(r"\cellcolor{green!10}{---} & \cellcolor{orange!10}{---}")
            continue
            
        # Get RMSE and std for both approaches at this horizon
        with_mean = pivot_mean.loc[patient, ('mLLM', horizon)]
        without_mean = pivot_mean.loc[patient, ('Dietitian', horizon)]
        with_std = pivot_std.loc[patient, ('mLLM', horizon)]
        without_std = pivot_std.loc[patient, ('Dietitian', horizon)]
        
        # Format with 2 decimal places, include std dev, and bold the better value
        if with_mean < without_mean:
            cell_with = fr"\cellcolor{{green!10}}{{\textbf{{{with_mean:.2f} $\pm$ {with_std:.2f}}}}}"
            cell_without = fr"\cellcolor{{orange!10}}{{{without_mean:.2f} $\pm$ {without_std:.2f}}}"
        else:
            cell_with = fr"\cellcolor{{green!10}}{{{with_mean:.2f} $\pm$ {with_std:.2f}}}"
            cell_without = fr"\cellcolor{{orange!10}}{{\textbf{{{without_mean:.2f} $\pm$ {without_std:.2f}}}}}"
        row_data.append(f"{cell_with} & {cell_without}")
    
    # Join the data for this patient with alternating row colors
    row_color = "gray!5" if len(rows) % 2 == 0 else "white"
    rows.append(fr"\rowcolor{{{row_color}}} {patient} & {' & '.join(row_data)} \\")

# Combine all rows
all_rows = "\n".join(rows)

# Format average row
avg_row_data = []
for horizon in horizons:
    # Skip if no data for this horizon and approach combination
    if (('mLLM', horizon) not in avg_values) or (('Dietitian', horizon) not in avg_values):
        avg_row_data.append(r"\cellcolor{green!10}{---} & \cellcolor{orange!10}{---}")
        continue
        
    # Get average RMSE and std for both approaches at this horizon
    with_mean = avg_values[('mLLM', horizon)]
    without_mean = avg_values[('Dietitian', horizon)]
    with_std = std_values[('mLLM', horizon)]
    without_std = std_values[('Dietitian', horizon)]
    
    # Format with 2 decimal places, include std dev, and bold the better value
    if with_mean < without_mean:
        cell_with = fr"\cellcolor{{green!10}}{{\textbf{{{with_mean:.2f} $\pm$ {with_std:.2f}}}}}"
        cell_without = fr"\cellcolor{{orange!10}}{{{without_mean:.2f} $\pm$ {without_std:.2f}}}"
    else:
        cell_with = fr"\cellcolor{{green!10}}{{{with_mean:.2f} $\pm$ {with_std:.2f}}}"
        cell_without = fr"\cellcolor{{orange!10}}{{\textbf{{{without_mean:.2f} $\pm$ {without_std:.2f}}}}}"
    avg_row_data.append(f"{cell_with} & {cell_without}")

avg_row = " & ".join(avg_row_data)

# Generate the full LaTeX table
styled_latex_table = styled_latex_template % (column_format, header_row, subheader_row, all_rows, avg_row)

# Write to file
with open('latex_tables/comparison_table.tex', 'w') as f:
    f.write(styled_latex_table)
    
print("LaTeX table has been generated at 'latex_tables/comparison_table.tex'")

# Calculate and display the comparison statistics
print("\nComparison Statistics (Dietitian vs mLLM):")
print("-" * 50)
print(f"{'Prediction Horizon':<20} {'Absolute Diff':<15} {'Percentage Diff':<15}")
print("-" * 50)

for horizon in horizons:
    if (('mLLM', horizon) in avg_values) and (('Dietitian', horizon) in avg_values):
        mlm_mean = avg_values[('mLLM', horizon)]
        dietitian_mean = avg_values[('Dietitian', horizon)]
        
        abs_diff = dietitian_mean - mlm_mean  # positive means mLLM is better
        pct_diff = (abs_diff / dietitian_mean) * 100  # percentage improvement
        
        # Highlight which approach is better
        better = "mLLM" if mlm_mean < dietitian_mean else "Dietitian"
        
        print(f"{horizon*5} Min{' ':<12} {abs_diff:+.2f}{' ':<8} {pct_diff:+.2f}%{' ':<5} (Better: {better})")

print("-" * 50) 
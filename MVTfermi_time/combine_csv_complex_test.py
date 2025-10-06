import pandas as pd
import os
from datetime import datetime
import ast

now = datetime.now().strftime("%y_%m_%d-%H_%M")

# List of input files
loc_dist = [
    "complex_10ms_ALL.csv",
    "complex_S_10ms_ALL.csv",
    "complex_S_3ms_ALL.csv",

]

# File paths
path = os.path.join(os.getcwd(), "01_ANALYSIS_RESULTS_Complex")
files = [os.path.join(path, loc) for loc in loc_dist]
output_file = os.path.join(path, f"complex_combined_test.csv")

# Columns to drop
drop_col_list = ['sim_det', 'analysis_det', 'base_det']

# Collect DataFrames
dfs = []
for f in files:
    if os.path.exists(f):
        df = pd.read_csv(f)

        # Optional: scale features
        for col in ['sigma_feature', 'sigma']:
            if col in df.columns:
                df[col] *= 1000

        dfs.append(df)
    else:
        print(f"Warning: File not found -> {f}")

# Combine and process
if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)

    # Convert analysis_det to list and compute features
    if 'analysis_det' in combined_df.columns:
        combined_df['analysis_det'] = combined_df['analysis_det'].apply(ast.literal_eval)
        combined_df['num_analysis_det'] = combined_df['analysis_det'].apply(len)


    # Drop unnecessary or empty-value columns
    combined_df = combined_df.drop(columns=drop_col_list, errors='ignore')

    cols_all_neg_999 = [col for col in combined_df.columns
                        if combined_df[col].nunique(dropna=True) == 1 and combined_df[col].iloc[0] == -999]
    cols_all_neg_99900 = [col for col in combined_df.columns
                          if combined_df[col].nunique(dropna=True) == 1 and combined_df[col].iloc[0] == -99900]

    combined_df = combined_df.drop(columns=cols_all_neg_999 + cols_all_neg_99900)

    # Save to file
    combined_df.to_csv(output_file, index=False)
    print(f"✅ Combined CSV saved to:\n{output_file}")
else:
    print("⚠️ No CSVs found to combine.")

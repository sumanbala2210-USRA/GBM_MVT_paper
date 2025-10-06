import pandas as pd
import os
from datetime import datetime
import ast

now = datetime.now().strftime("%y_%m_%d-%H_%M")

loc_dist = [
    "Norris_BW_amp_1.csv",
    "Norris_BW_amp_2.csv",
    "Norris_BW_amp_3.csv",
    "Norris_BW_amp_4.csv",
    "Norris_BW_amp_5.csv",
    "Norris_BW_amp_6.csv",
]

path = os.path.join(os.getcwd(), "01_ANALYSIS_RESULTS")

files = [os.path.join(path, loc) for loc in loc_dist]

output_file = os.path.join(path, f"Norris_BW_AMP_ALL.csv")
drop_col_list = ['sim_det', 'analysis_det', 'base_det'] 

# Collect DataFrames
dfs = []
for f in files:
    if os.path.exists(f):
        df = pd.read_csv(f)
        if "sigma" in df.columns:
            df["sigma"] = df["sigma"] * 1000
        if "width" in df.columns:
            df["width"] = df["width"] * 1000
        if "rise_time" in df.columns:
            df["rise_time"] = df["rise_time"] * 1000
        if "decay_time" in df.columns:
            df["decay_time"] = df["decay_time"] * 1000
        dfs.append(df)
    else:
        print(f"Warning: File not found -> {f}")

# Combine them
if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)

    # Convert analysis_det strings to lists
    combined_df['analysis_det'] = combined_df['analysis_det'].apply(ast.literal_eval)

    # Add column with number of detections
    combined_df['num_analysis_det'] = combined_df['analysis_det'].apply(len)

    combined_df = combined_df.drop(columns=drop_col_list, errors='ignore')

    cols_all_neg_999 = [col for col in combined_df.columns
                        if combined_df[col].nunique(dropna=True) == 1 and combined_df[col].iloc[0] == -999]
    cols_all_neg_99900 = [col for col in combined_df.columns
                        if combined_df[col].nunique(dropna=True) == 1 and combined_df[col].iloc[0] == -99900]
    

    combined_df = combined_df.drop(columns=cols_all_neg_999)
    combined_df = combined_df.drop(columns=cols_all_neg_99900)
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV saved to:\n{output_file}")
else:
    print("No CSVs found to combine.")

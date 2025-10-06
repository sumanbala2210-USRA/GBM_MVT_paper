import yaml
import numpy as np
from pathlib import Path
import argparse
import sys
from collections import defaultdict

# --- Helper functions for parsing and formatting ---

def format_value(value, precision=1):
    """Formats a number, handling None or NaN and lists."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return '---'
    if isinstance(value, list):
        return f"[{format_value(value[0])}, {format_value(value[1])}]"
    return f"{value:.{precision}f}"

def format_background_interval(intervals):
    """Formats the background interval list into a clear string."""
    if not isinstance(intervals, list) or len(intervals) != 2:
        return '---'
    part1 = format_value(intervals[0])
    part2 = format_value(intervals[1])
    return f"{part1}; {part2}"

def parse_yaml_data(filepath):
    """Reads a single YAML file and extracts/formats data."""
    try:
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"--- WARNING: Could not read/parse {filepath}: {e} ---", file=sys.stderr)
        return None

    mvt = data.get('mvt_summary', {})
    det_list = data.get('det_list', [])
    det_combo_str = ", ".join(det_list) if isinstance(det_list, list) else str(det_list)
    #if det_list == ['n2']: det_combo_str = "Single Best"
    #elif sorted(det_list) == ['n1', 'n2', 'na']: det_combo_str = "n2, na, n1"
    if len(det_list) == 12: det_combo_str = "All"
    #elif len(det_list) > 3: det_combo_str = f"{len(det_list)} Detectors"
    mvt_err = mvt.get('mvt_err_ms')
    if mvt_err ==0:
        mvt_err_str = "<"
    else:
        mvt_err_str = "\\pm " + format_value(mvt_err)

    return {
        'det_combo_str': det_combo_str,
        'source_interval': format_value(data.get('source_interval')),
        'background_intervals': format_background_interval(data.get('background_intervals')),
        'bin_width': str(round(data.get('bin_width_ms', np.nan), 3)),
        'single_mvt': f"${format_value(mvt.get('mvt_ms'))} {mvt_err_str}$",
        'median_mvt': f"${format_value(mvt.get('median_mvt_ms'))}_{{-{format_value(mvt.get('mvt_err_lower'))}}}^{{+{format_value(mvt.get('mvt_err_upper'))}}}$",
        'snr_mvt': format_value(mvt.get('SNR_mvt')),
        'T90': format_value(data.get('T90')),
        'successful_runs': round(mvt.get('successful_runs')/300*100, 1) if mvt.get('successful_runs') is not None else '---',
        'failed_runs': round(mvt.get('failed_runs')/300*100, 1) if mvt.get('failed_runs') is not None else '---'
    }

# --- Main LaTeX Generation Logic ---

def generate_latex_table(all_records, trigger_number):
    """Generates a complete, standalone LaTeX table for a single GRB."""
    if not all_records:
        return

    # Group data by their interval combinations
    grouped_data = defaultdict(list)
    for record in all_records:
        key = (record['source_interval'], record['background_intervals'])
        grouped_data[key].append(record)

    # --- Print the complete table structure ---
    print("\n" + "="*70)
    print(f"% COMPLETE LATEX TABLE FOR GRB {trigger_number}")
    print("% To combine GRBs, create one table environment and copy the data")
    print("% rows (from the first \\midrule to \\bottomrule) from each output.")
    print("="*70)
    
    print(f"\\begin{{table*}}[h!]")
    print(f"  \\centering")
    print(f"  \\caption{{Detailed MVT Results for GRB {trigger_number}.}}")
    print(f"  \\label{{tab:appendix_{trigger_number}}}")
    print(f"  \\resizebox{{\\textwidth}}{{!}}{{")
    print(f"  \\begin{{tabular}}{{l c c c c c c}}") # 7 columns
    print(f"    \\toprule")
    print(f"    Detector(s) & Bin Width & Single MVT & Median MVT & $\\text{{SNR}}_{{\\text{{MVT}}}}$ & Success & Failed \\\\")
    print(f"                & (ms)      & (ms)       & (ms)       &                             & Runs (%)    & Runs (%) \\\\")
    print(f"    \\midrule")

    # --- Generate the sub-header and data rows for each group ---
    sorted_groups = sorted(grouped_data.items(), key=lambda item: item[0])
    for (src_interval, bg_interval), records in sorted_groups:
        subheader = (f"    \\multicolumn{{7}}{{l}}{{\\textbf{{GRB {trigger_number}}} \\quad "
                     f"Source: {src_interval} s; \\quad Background: {bg_interval} s; \\quad T_{{90}}: {record['T90']} s}} \\\\")
        print(subheader)
        print("    \\cmidrule(r){1-7}")
        
        for record in records:
            row_parts = [
                record['det_combo_str'],
                record['bin_width'],
                record['single_mvt'],
                record['median_mvt'],
                record['snr_mvt'],
                str(record['successful_runs']),
                str(record['failed_runs'])
            ]
            print("    " + " & ".join(row_parts) + " \\\\")

    # --- Print the table closing statements ---
    print(f"    \\bottomrule")
    print(f"  \\end{{tabular}}")
    print(f"  }}") # Closes \resizebox
    print(f"\\end{{table*}}")

def main():
    parser = argparse.ArgumentParser(description="Generate a complete LaTeX table with sub-headers for a given GRB.")
    parser.add_argument("trigger_number", help="The trigger number of the GRB (e.g., 211211549).")
    args = parser.parse_args()
    
    trigger_number = args.trigger_number
    data_dir = Path(f"MVT_bn{trigger_number}")
    
    if not data_dir.is_dir():
        print(f"❌ Error: Directory not found at './{data_dir}/'", file=sys.stderr)
        sys.exit(1)
        
    file_pattern = f"config_MVT_{trigger_number}_*.yaml"
    #yaml_files = sorted(list(data_dir.glob(file_pattern)))
    yaml_files = sorted(list(data_dir.glob(file_pattern)), reverse=True)
    
    if not yaml_files:
        print(f"⚠️ Warning: No YAML files found for GRB {trigger_number}.", file=sys.stderr)
        sys.exit(0)
    
    print(f"✅ Found {len(yaml_files)} YAML files for GRB {trigger_number}. Processing...", file=sys.stderr)

    all_records = [parse_yaml_data(f) for f in yaml_files if f is not None]
    
    generate_latex_table(all_records, trigger_number)

if __name__ == '__main__':
    main()
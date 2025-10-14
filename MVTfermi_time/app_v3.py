import streamlit as st
import pandas as pd
from plot_from_config_v2 import plot_plotly #plot_matplotlib,
import numpy as np
import ast

st.set_page_config(layout="wide")
st.title("MVT Simulation Plotter")

uploaded_file = st.file_uploader("Choose a final summary CSV file")
drop_col_list = ['sim_det', 'analysis_det', 'base_det'] 

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    initial_rows = len(df)

    combined_df = df[~(df.drop(columns=['median_mvt_ms']) == -100).any(axis=1)].copy()

    try:
        combined_df['analysis_det'] = combined_df['analysis_det'].apply(ast.literal_eval)
        combined_df['num_analysis_det'] = combined_df['analysis_det'].apply(len)
    except Exception as e:
        st.warning(f"Could not process 'analysis_det' column: {e}")
        pass

    combined_df = combined_df.drop(columns=drop_col_list, errors='ignore')

    cols_all_neg_999 = [col for col in combined_df.columns
                        if combined_df[col].nunique(dropna=True) == 1 and combined_df[col].iloc[0] == -999]
    df = combined_df.drop(columns=cols_all_neg_999)

    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        st.info(f"Removed {dropped_rows} rows with invalid simulation runs.")
    
    st.write("Data Preview (after filtering):", df.head())
    columns = df.columns.tolist()

    st.sidebar.header("Plot Configuration")
    
    plot_backend = st.sidebar.radio("Plotting Backend", ["Interactive (Plotly)", "Static (Matplotlib)"])

    x_col = st.sidebar.selectbox("X-Axis:", columns, index=columns.index('peak_amplitude') if 'peak_amplitude' in columns else 0)
    y_col = st.sidebar.selectbox("Y-Axis:", columns, index=columns.index('median_mvt_ms') if 'median_mvt_ms' in columns else 1)
    group_col = st.sidebar.selectbox("Group By:", columns, index=columns.index('sigma') if 'sigma' in columns else 2)

    marker_col_options = [None] + columns
    marker_col = st.sidebar.selectbox("Marker Style By:", marker_col_options, index=0)

    
    st.sidebar.markdown("---")
    show_error_bars = st.sidebar.checkbox("Show Error Bars", value=True)
    show_limits = st.sidebar.checkbox("Show Lower Limits", value=True)
    use_log_x = st.sidebar.checkbox("Use Log Scale for X-axis")
    use_log_y = st.sidebar.checkbox("Use Log Scale for Y-axis")
    use_intensity_color = st.sidebar.checkbox("Color by Success Rate")


    st.sidebar.header("Axis Ranges (Optional)")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        x_min = st.number_input("X-axis Min", value=None, format="%g")
        y_min = st.number_input("Y-axis Min", value=None, format="%g")
    with col2:
        x_max = st.number_input("X-axis Max", value=None, format="%g")
        y_max = st.number_input("Y-axis Max", value=None, format="%g")
    
    plot_theme = st.sidebar.selectbox("Plot Theme", ["simple_white", "Contrast White", "Contrast Dark", "Seaborn-like"])
    
    st.sidebar.markdown("---") 
    plot_height = st.sidebar.number_input("Plot Height (pixels)", min_value=400, value=700, step=50)
    st.sidebar.header("Filters")
    filters = {}
    exclude_list = ['t_start', 't_stop', 'det', 'trigger_number', 'mvt_error_ms']
    filter_cols = [c for c in columns if c not in exclude_list]
    for col in filter_cols:
        unique_vals = sorted(df[col].unique().tolist())
        selected = st.sidebar.multiselect(f"Filter by {col}:", unique_vals, default=unique_vals)
        if selected != unique_vals:
            filters[col] = selected

    @st.cache_data
    def convert_df_to_csv(df_to_convert):
        return df_to_convert.to_csv(index=False).encode('utf-8')

    if st.button("Generate Plot"):
        filtered_df = df.copy()
        if filters:
            for col, selected_values in filters.items():
                if selected_values:
                    filtered_df = filtered_df[filtered_df[col].isin(selected_values)]

        if not filtered_df.empty:
            x_range = [x_min, x_max] if x_min is not None and x_max is not None else None
            y_range = [y_min, y_max] if y_min is not None and y_max is not None else None

            if "Interactive" in plot_backend:
                y_err_upper = 'mvt_err_upper' if 'mvt_err_upper' in df.columns else (
              'mvt_err_upper_ms' if 'mvt_err_upper_ms' in df.columns else None)

                y_err_lower = 'mvt_err_lower' if 'mvt_err_lower' in df.columns else (
              'mvt_err_lower_ms' if 'mvt_err_lower_ms' in df.columns else None)

                color_intensity_arg = 'success_percent' if use_intensity_color else None

                fig = plot_plotly(
                    df=df, x_axis_col=x_col, y_axis_col=y_col, group_by_col=group_col, 
                    marker_col=marker_col,
                    y_err_upper_col=y_err_upper,
                    y_err_lower_col=y_err_lower,
                    color_intensity_col=color_intensity_arg,
                    filters=filters,
                    show_lower_limits=show_limits,
                    show_error_bars=show_error_bars, 
                    use_log_x=use_log_x, use_log_y=use_log_y, x_range=x_range, y_range=y_range, 
                    plot_theme=plot_theme, plot_height=plot_height
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True, theme=None)

                    # <<< MODIFIED BLOCK: More robust logic for collecting download columns >>>
                    # Start with a list to be explicit
                    cols_for_download = [x_col, y_col, group_col]

                    # Add the marker column if it has been selected
                    if marker_col is not None:
                        cols_for_download.append(marker_col)

                    # Add error bar columns if they are used and exist
                    if show_error_bars:
                        if y_err_upper and y_err_upper in df.columns:
                            cols_for_download.append(y_err_upper)
                        if y_err_lower and y_err_lower in df.columns:
                            cols_for_download.append(y_err_lower)

                    # Add intensity color column if used and exists
                    if color_intensity_arg and color_intensity_arg in df.columns:
                        cols_for_download.append(color_intensity_arg)

                    # Remove duplicates while preserving order
                    final_cols = list(dict.fromkeys(cols_for_download))
                    
                    # Create the final dataframe for download
                    download_df = filtered_df[final_cols]
                    # <<< END OF MODIFIED BLOCK >>>

                    csv_data = convert_df_to_csv(download_df)
                    st.download_button(
                       label="📥 Download Plotted Data (CSV)",
                       data=csv_data,
                       file_name=f"plotted_data_{x_col}_vs_{y_col}.csv",
                       mime="text/csv",
                    )

                else:
                    st.warning("No data to plot after applying filters.")
        else:
            st.warning("No data to plot after applying filters.")
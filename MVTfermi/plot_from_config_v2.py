import pandas as pd
import yaml
import streamlit as st
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go



import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_plotly(
    df,
    x_axis_col,
    group_by_col,
    y_axis_col='median_mvt_ms',
    y_err_upper_col=None,
    y_err_lower_col=None,
    marker_col=None,
    # This argument will act as our "switch"
    color_intensity_col=None,
    filters=None,
    show_lower_limits=True,
    show_error_bars=True,
    use_log_x=False,
    use_log_y=False,
    x_range=None,
    y_range=None,
    plot_theme="Contrast White",
    plot_height=800
):
    """
    Creates a publication-quality plot that can operate in two modes:
    1. Default: Assigns a unique solid color to each group.
    2. Intensity Mode: Assigns color intensity based on a continuous variable.
    """
    error_col_for_filtering = y_err_lower_col or y_err_upper_col
    if not show_lower_limits and error_col_for_filtering and error_col_for_filtering in df.columns:
        plot_df = df[df[error_col_for_filtering] > 0].copy()
    else:
        plot_df = df.copy()

    if filters:
        for key, value in filters.items():
            plot_df = plot_df[plot_df[key].isin(value)]

    if plot_df.empty:
        return None
    
    # --- Check if we should use the advanced intensity coloring ---
    use_intensity_coloring = (
        color_intensity_col is not None and
        'successful_runs' in plot_df.columns and
        'total_sim' in plot_df.columns
    )

    # Prepare columns for either plotting mode
    plot_df[group_by_col] = plot_df[group_by_col].astype(str)
    if marker_col:
        plot_df[marker_col] = plot_df[marker_col].astype(str)
    
    # --- Set up the color and symbol arguments ---
    if use_intensity_coloring:
        plot_df['success_percent'] = (plot_df['successful_runs'] / plot_df['total_sim']) * 100
        color_arg = 'success_percent'
        symbol_arg = group_by_col
        # --- Use a colorblind-friendly continuous scale ---
        color_scale = 'Cividis'
    else:
        color_arg = group_by_col
        symbol_arg = marker_col
        # --- Use a colorblind-friendly discrete color palette ---
        color_scale = px.colors.qualitative.D3

    # ... (Lower limits and hover data logic is the same) ...
    reject_list = ['t_start', 't_stop', 'det', 'trigger_number', 'pulse']
    hover_cols = [col for col in plot_df.columns if col not in reject_list]
    error_y_arg = y_err_upper_col if show_error_bars and y_err_upper_col in plot_df.columns else None
    error_y_minus_arg = y_err_lower_col if show_error_bars and y_err_lower_col in plot_df.columns else None

    # 5. Create the figure using Plotly Express
    fig = px.scatter(
        plot_df,
        x=x_axis_col,
        y=y_axis_col,
        color=color_arg,
        symbol=symbol_arg,
        error_y=error_y_arg,
        error_y_minus=error_y_minus_arg,
        # --- Use the determined color scale ---
        color_continuous_scale=color_scale if use_intensity_coloring else None,
        color_discrete_sequence=color_scale if not use_intensity_coloring else None,
        log_x=use_log_x,
        log_y=use_log_y,
        labels={
            x_axis_col: x_axis_col.replace('_', ' ').title(),
            y_axis_col: y_axis_col.replace('_', ' ').title(),
            group_by_col: group_by_col.replace('_', ' ').title(),
            color_arg: "Success Rate (%)" if use_intensity_coloring else group_by_col.replace('_', ' ').title(),
            symbol_arg: symbol_arg.replace('_', ' ').title() if symbol_arg and not use_intensity_coloring else ""
        },
        template="plotly_white",
        hover_data=hover_cols,
    )

    # ... (Styling logic is the same) ...
    theme_styles = {
        "Contrast White": {'paper_bgcolor': "white", 'plot_bgcolor': "white", 'font_color': "black", 'gridcolor': '#D3D3D3', 'zerolinecolor': '#C0C0C0'},
        "Contrast Dark": {'paper_bgcolor': "#1E1E1E", 'plot_bgcolor': "#2E2E2E", 'font_color': "white", 'gridcolor': '#4A4A4A', 'zerolinecolor': '#7A7A7A'},
        "Seaborn-like": {'paper_bgcolor': "#F0F2F6", 'plot_bgcolor': "#F0F2F6", 'font_color': "black", 'gridcolor': 'white', 'zerolinecolor': 'white'}
    }
    font_color = "black"
    border_color = "black" 
    if plot_theme in theme_styles:
        style = theme_styles[plot_theme]
        font_color = style['font_color']
        if plot_theme == "Contrast Dark":
            border_color = "white"
        fig.update_layout(paper_bgcolor=style['paper_bgcolor'], plot_bgcolor=style['plot_bgcolor'], font_color=font_color)
        fig.update_xaxes(gridcolor=style['gridcolor'], zerolinecolor=style['zerolinecolor'])
        fig.update_yaxes(gridcolor=style['gridcolor'], zerolinecolor=style['zerolinecolor'])
    elif plot_theme == "simple_white":
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", font_color="black")
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', zerolinecolor='gray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', zerolinecolor='gray')

    fig.update_traces(marker=dict(size=12, line=dict(width=1.5, color=border_color)), error_y=dict(thickness=2.0))
    fig.update_layout(
        height=plot_height,
        title_x=0.5,
        xaxis_range=x_range,
        yaxis_range=y_range,
        showlegend=(not use_intensity_coloring),
        font=dict(
            family="Arial, sans-serif",
            size=18,
            color=font_color
        ),
        title_font_size=24,
        xaxis=dict(title_font_size=22),
        yaxis=dict(title_font_size=22),
        legend=dict(
            title_font_size=20,
            font_size=18,
            traceorder="normal"
        )
    )

    return fig
    
import plotly.express as px


#### Working version of plotly function with dynamic legend configuration







# This function is defined in your script but not used. It's safe to keep or remove.
def plot_dynamic_legend(
    df,
    x_axis_col,
    y_axis_col,
    group_by_col,
    color_col=None,
    symbol_col=None,
    hover_data=None,
    plot_height=800
):
    fig = px.scatter(
        df,
        x=x_axis_col,
        y=y_axis_col,
        color=color_col,
        symbol=symbol_col,
        hover_data=hover_data,
        height=plot_height
    )
    if group_by_col in df.columns:
        fig.for_each_trace(lambda t: t.update(name=t.name.split("=")[-1].strip(), legendgroup=t.name.split("=")[-1].strip()))
    return fig






def plot_plotly(
    df,
    x_axis_col,
    group_by_col,
    y_axis_col='median_mvt_ms',
    y_err_upper_col=None,
    y_err_lower_col=None,
    marker_col=None,
    color_intensity_col=None,
    filters=None,
    show_lower_limits=True,
    show_error_bars=True,
    use_log_x=False,
    use_log_y=False,
    x_range=None,
    y_range=None,
    plot_theme="Contrast White",
    plot_height=800
):
    error_col_for_filtering = y_err_lower_col or y_err_upper_col
    if not show_lower_limits and error_col_for_filtering and error_col_for_filtering in df.columns:
        plot_df = df[df[error_col_for_filtering] > 0].copy()
    else:
        plot_df = df.copy()

    if filters:
        for key, value in filters.items():
            plot_df = plot_df[plot_df[key].isin(value)]

    if plot_df.empty:
        return None
    # Add this line
    should_display_errors = show_error_bars and (y_axis_col == 'median_mvt_ms')
    
    use_intensity_coloring = (
        color_intensity_col is not None and
        'successful_runs' in plot_df.columns and
        'total_sim' in plot_df.columns
    )

    plot_df[group_by_col] = plot_df[group_by_col].astype(str)
    if marker_col:
        plot_df[marker_col] = plot_df[marker_col].astype(str)
    
    # --- LOGIC FOR SETTING UP ARGS & LABELS ---
    labels_dict = {
        x_axis_col: x_axis_col.replace('_', ' ').title(),
        y_axis_col: y_axis_col.replace('_', ' ').title(),
    }

    """ --- MODIFIED BLOCK: More robust logic for collecting legend columns --- 
    if use_intensity_coloring:
        plot_df['success_percent'] = (plot_df['successful_runs'] / plot_df['total_sim']) * 100
        color_arg = 'success_percent'
        color_scale = 'Cividis'

        if marker_col and marker_col in plot_df.columns:
            plot_df['legend_pair'] = plot_df[group_by_col].astype(str) + ', ' + plot_df[marker_col].astype(str)
            symbol_arg = 'legend_pair'
            legend_title = f"{group_by_col.replace('_', ' ').title()}, {marker_col.replace('_', ' ').title()}"
        else:
            symbol_arg = group_by_col
            legend_title = group_by_col.replace('_', ' ').title()
    """
    # Replace the old section with this new, updated one
    if use_intensity_coloring:
        plot_df['success_percent'] = (plot_df['successful_runs'] / plot_df['total_sim']) * 100
        color_arg = 'success_percent'
        color_scale = 'Cividis'

        if marker_col and marker_col in plot_df.columns:
            plot_df['legend_pair'] = plot_df[group_by_col].astype(str) + ', ' + plot_df[marker_col].astype(str)
            symbol_arg = 'legend_pair'
            legend_title = f"{group_by_col.replace('_', ' ').title()}, {marker_col.replace('_', ' ').title()}"

            # --- ADDED: Sort the legend pairs numerically ---
            unique_pairs = plot_df['legend_pair'].unique()
            # This sorts by the first value (group_by), then the second (marker_by)
            sorted_pairs = sorted(unique_pairs, key=lambda p: (float(p.split(',')[0]), float(p.split(',')[1])))
            # Convert column to an ordered category
            plot_df['legend_pair'] = pd.Categorical(plot_df['legend_pair'], categories=sorted_pairs, ordered=True)

        else:
            # Fallback if no marker_col is selected
            symbol_arg = group_by_col
            legend_title = group_by_col.replace('_', ' ').title()

            # --- ADDED: Sort the legend numerically here as well ---
            sorted_categories = sorted(plot_df[group_by_col].unique(), key=float)
            plot_df[group_by_col] = pd.Categorical(plot_df[group_by_col], categories=sorted_categories, ordered=True)
        # Set labels for success rate mode
        labels_dict[color_arg] = "Success Rate (%)"
        labels_dict[symbol_arg] = legend_title
            
    else:
        # Default behavior: Let Plotly create the combined legend automatically
        color_arg = group_by_col
        symbol_arg = marker_col
        color_scale = px.colors.qualitative.D3
        # We don't need to manually set titles here; Plotly handles it best.
        if color_arg:
             labels_dict[color_arg] = color_arg.replace("_", " ").title()
        if symbol_arg:
             labels_dict[symbol_arg] = symbol_arg.replace("_", " ").title()

    reject_list = ['t_start', 't_stop', 'det', 'trigger_number', 'pulse']
    hover_cols = [col for col in plot_df.columns if col not in reject_list]
    #error_y_arg = y_err_upper_col if show_error_bars and y_err_upper_col in plot_df.columns else None
    #error_y_minus_arg = y_err_lower_col if show_error_bars and y_err_lower_col in plot_df.columns else None
    # Replace them with these
    error_y_arg = y_err_upper_col if should_display_errors and y_err_upper_col in plot_df.columns else None
    error_y_minus_arg = y_err_lower_col if should_display_errors and y_err_lower_col in plot_df.columns else None

    fig = px.scatter(
        plot_df,
        x=x_axis_col,
        y=y_axis_col,
        color=color_arg,
        symbol=symbol_arg,
        error_y=error_y_arg,
        error_y_minus=error_y_minus_arg,
        color_continuous_scale=color_scale if use_intensity_coloring else None,
        color_discrete_sequence=color_scale if not use_intensity_coloring else None,
        log_x=use_log_x,
        log_y=use_log_y,
        labels=labels_dict, # Use the conditionally built dictionary
        template="plotly_white",
        hover_data=hover_cols,
    )
    
    # The rest of the function remains the same
    theme_styles = {
        "Contrast White": {'paper_bgcolor': "white", 'plot_bgcolor': "white", 'font_color': "black", 'gridcolor': '#D3D3D3', 'zerolinecolor': '#C0C0C0'},
        "Contrast Dark": {'paper_bgcolor': "#1E1E1E", 'plot_bgcolor': "#2E2E2E", 'font_color': "white", 'gridcolor': '#4A4A4A', 'zerolinecolor': '#7A7A7A'},
        "Seaborn-like": {'paper_bgcolor': "#F0F2F6", 'plot_bgcolor': "#F0F2F6", 'font_color': "black", 'gridcolor': 'white', 'zerolinecolor': 'white'}
    }
    font_color = "black"
    border_color = "black" 
    if plot_theme in theme_styles:
        style = theme_styles[plot_theme]
        font_color = style['font_color']
        if plot_theme == "Contrast Dark":
            border_color = "white"
        fig.update_layout(paper_bgcolor=style['paper_bgcolor'], plot_bgcolor=style['plot_bgcolor'], font_color=font_color)
        fig.update_xaxes(gridcolor=style['gridcolor'], zerolinecolor=style['zerolinecolor'])
        fig.update_yaxes(gridcolor=style['gridcolor'], zerolinecolor=style['zerolinecolor'])
    elif plot_theme == "simple_white":
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", font_color="black")
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', zerolinecolor='gray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', zerolinecolor='gray')

    fig.update_traces(marker=dict(size=12, line=dict(width=1.5, color=border_color)), error_y=dict(thickness=2.0))

    legend_config = dict(
        title_font_size=20,
        font_size=18,
        traceorder="normal"
    )
    if use_intensity_coloring:
        legend_config.update(dict(
            x=1.02, y=1, xanchor="left", yanchor="top"
        ))
    
    fig.update_layout(
        height=plot_height,
        title_x=0.5,
        xaxis_range=x_range,
        yaxis_range=y_range,
        legend=legend_config,
        font=dict(
            family="Arial, sans-serif",
            size=18,
            color=font_color
        ),
        title_font_size=24,
        xaxis=dict(title_font_size=22),
        yaxis=dict(title_font_size=22),
    )

    if use_intensity_coloring:
        fig.update_layout(
            coloraxis_colorbar=dict(
                title="Success Rate (%)",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            coloraxis_cmin=0,  # <-- CORRECTED: Set the minimum of the color scale
            coloraxis_cmax=100  #
        )

    return fig




def plot_plotly(
    df,
    x_axis_col,
    group_by_col,
    y_axis_col='median_mvt_ms',
    y_err_upper_col=None,
    y_err_lower_col=None,
    marker_col=None,
    color_intensity_col=None,
    filters=None,
    show_lower_limits=True,
    show_error_bars=True,
    use_log_x=False,
    use_log_y=False,
    x_range=None,
    y_range=None,
    plot_theme="Contrast White",
    plot_height=800
):
    error_col_for_filtering = y_err_lower_col or y_err_upper_col
    if not show_lower_limits and error_col_for_filtering and error_col_for_filtering in df.columns:
        plot_df = df[df[error_col_for_filtering] > 0].copy()
    else:
        plot_df = df.copy()

    if filters:
        for key, value in filters.items():
            plot_df = plot_df[plot_df[key].isin(value)]

    if plot_df.empty:
        return None
    
    should_display_errors = show_error_bars and (y_axis_col == 'median_mvt_ms')
    
    use_intensity_coloring = (
        color_intensity_col is not None and
        'successful_runs' in plot_df.columns and
        'total_sim' in plot_df.columns
    )

    plot_df[group_by_col] = plot_df[group_by_col].astype(str)
    if marker_col:
        plot_df[marker_col] = plot_df[marker_col].astype(str)
    
    labels_dict = {
        x_axis_col: x_axis_col.replace('_', ' ').title(),
        y_axis_col: y_axis_col.replace('_', ' ').title(),
    }
    
    # --- NEW: Build a dictionary for category sorting ---
    category_orders_dict = {}

    if use_intensity_coloring:
        plot_df['success_percent'] = (plot_df['successful_runs'] / plot_df['total_sim']) * 100
        color_arg = 'success_percent'
        color_scale = 'Cividis'

        if marker_col and marker_col in plot_df.columns:
            plot_df['legend_pair'] = plot_df[group_by_col].astype(str) + ', ' + plot_df[marker_col].astype(str)
            symbol_arg = 'legend_pair'
            legend_title = f"{group_by_col.replace('_', ' ').title()}, {marker_col.replace('_', ' ').title()}"
            
            # Create the sorted list for the category_orders dictionary
            unique_pairs = plot_df['legend_pair'].unique()
            sorted_pairs = sorted(unique_pairs, key=lambda p: (float(p.split(',')[0]), float(p.split(',')[1])))
            category_orders_dict[symbol_arg] = sorted_pairs
        else:
            symbol_arg = group_by_col
            legend_title = group_by_col.replace('_', ' ').title()
            sorted_categories = sorted(plot_df[group_by_col].unique(), key=float)
            category_orders_dict[symbol_arg] = sorted_categories
        
        labels_dict[color_arg] = "Success Rate (%)"
        labels_dict[symbol_arg] = legend_title
            
    else:
        color_arg = group_by_col
        symbol_arg = marker_col
        color_scale = px.colors.qualitative.D3
        
        # --- NEW: Also sort the categories for the default mode ---
        sorted_groups = sorted(plot_df[group_by_col].unique(), key=float)
        category_orders_dict[color_arg] = sorted_groups
        if symbol_arg and symbol_arg in plot_df.columns:
            # Check if marker column is numeric before sorting
            try:
                sorted_markers = sorted(plot_df[symbol_arg].unique(), key=float)
                category_orders_dict[symbol_arg] = sorted_markers
            except (ValueError, TypeError):
                # If marker column is not numeric, use default sorting
                pass

        if color_arg: labels_dict[color_arg] = color_arg.replace("_", " ").title()
        if symbol_arg: labels_dict[symbol_arg] = symbol_arg.replace("_", " ").title()

    reject_list = ['t_start', 't_stop', 'det', 'trigger_number', 'pulse']
    hover_cols = [col for col in plot_df.columns if col not in reject_list]
    error_y_arg = y_err_upper_col if should_display_errors and y_err_upper_col in plot_df.columns else None
    error_y_minus_arg = y_err_lower_col if should_display_errors and y_err_lower_col in plot_df.columns else None

    fig = px.scatter(
        plot_df,
        x=x_axis_col,
        y=y_axis_col,
        color=color_arg,
        symbol=symbol_arg,
        error_y=error_y_arg,
        error_y_minus=error_y_minus_arg,
        color_continuous_scale=color_scale if use_intensity_coloring else None,
        color_discrete_sequence=color_scale if not use_intensity_coloring else None,
        log_x=use_log_x,
        log_y=use_log_y,
        labels=labels_dict,
        category_orders=category_orders_dict, # <-- FINALLY: Pass the sorting dictionary here
        template="plotly_white",
        hover_data=hover_cols,
    )
    
    # The rest of the function remains the same
    theme_styles = {
        "Contrast White": {'paper_bgcolor': "white", 'plot_bgcolor': "white", 'font_color': "black", 'gridcolor': '#D3D3D3', 'zerolinecolor': '#C0C0C0'},
        "Contrast Dark": {'paper_bgcolor': "#1E1E1E", 'plot_bgcolor': "#2E2E2E", 'font_color': "white", 'gridcolor': '#4A4A4A', 'zerolinecolor': '#7A7A7A'},
        "Seaborn-like": {'paper_bgcolor': "#F0F2F6", 'plot_bgcolor': "#F0F2F6", 'font_color': "black", 'gridcolor': 'white', 'zerolinecolor': 'white'}
    }
    font_color = "black"
    border_color = "black" 
    if plot_theme in theme_styles:
        style = theme_styles[plot_theme]
        font_color = style['font_color']
        if plot_theme == "Contrast Dark":
            border_color = "white"
        fig.update_layout(paper_bgcolor=style['paper_bgcolor'], plot_bgcolor=style['plot_bgcolor'], font_color=font_color)
        fig.update_xaxes(gridcolor=style['gridcolor'], zerolinecolor=style['zerolinecolor'])
        fig.update_yaxes(gridcolor=style['gridcolor'], zerolinecolor=style['zerolinecolor'])
    elif plot_theme == "simple_white":
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", font_color="black")
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', zerolinecolor='gray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', zerolinecolor='gray')

    fig.update_traces(marker=dict(size=12, line=dict(width=1.5, color=border_color)), error_y=dict(thickness=2.0))

    legend_config = dict(title_font_size=20, font_size=18, traceorder="normal")
    if use_intensity_coloring:
        legend_config.update(dict(x=1.02, y=1, xanchor="left", yanchor="top"))
    
    fig.update_layout(
        height=plot_height, title_x=0.5,
        xaxis_range=x_range, yaxis_range=y_range,
        legend=legend_config,
        font=dict(family="Arial, sans-serif", size=18, color=font_color),
        title_font_size=24,
        xaxis=dict(title_font_size=22),
        yaxis=dict(title_font_size=22),
    )

    if use_intensity_coloring:
        fig.update_layout(
            coloraxis_colorbar=dict(title="Success Rate (%)", orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            coloraxis_cmin=0, coloraxis_cmax=100
        )

    return fig
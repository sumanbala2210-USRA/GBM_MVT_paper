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



def plot_plotly_old(
    df,
    x_axis_col,
    group_by_col,
    y_axis_col='median_mvt_ms',
    # <<< MODIFIED: Replaced y_err_col with upper and lower error columns >>>
    y_err_upper_col=None,
    y_err_lower_col=None,
    marker_col=None,
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
    Creates a publication-quality, advanced interactive plot using Plotly Express.
    Features larger fonts, thicker lines, and bordered markers for clarity.
    """
    error_col_for_filtering = y_err_lower_col or y_err_upper_col
    # 1. Conditionally remove zero-error points based on the UI toggle
    if not show_lower_limits and error_col_for_filtering:
        plot_df = df[(df[error_col_for_filtering] > 0) & (df['failed_runs'] < df['total_sim'].max()*0.9)].copy()
    else:
        plot_df = df.copy()

    # 2. Apply user-defined filters
    if filters:
        for key, value in filters.items():
            plot_df = plot_df[plot_df[key].isin(value)]

    if plot_df.empty:
        return None

    # 3. Prepare data for plotting
    plot_df[group_by_col] = plot_df[group_by_col].astype(str)
    
    symbol_arg = None
    color_legend_title = group_by_col.replace('_', ' ').title()

    if show_lower_limits and y_err_upper_col in plot_df.columns:
        symbol_legend_title = 'Point Type'
        if marker_col:
            symbol_legend_title = marker_col.replace('_', ' ').title()
            plot_df[marker_col] = plot_df[marker_col].astype(str)
            plot_df[symbol_legend_title] = np.where(plot_df[y_err_upper_col] > 0, plot_df[marker_col], 'Lower Limit')
            symbol_arg = symbol_legend_title
        else:
            plot_df[symbol_legend_title] = np.where(plot_df[y_err_upper_col] > 0, 'Data', 'Lower Limit')
            symbol_arg = symbol_legend_title
    elif marker_col:
        symbol_arg = marker_col.replace('_', ' ').title()
        plot_df[symbol_arg] = plot_df[marker_col].astype(str)

    # 4. Prepare other plot arguments
    error_y_arg = None
    error_y_minus_arg = None
    if show_error_bars and y_err_upper_col and y_err_lower_col:
        if y_err_upper_col in plot_df.columns and y_err_lower_col in plot_df.columns:
            error_y_arg = y_err_upper_col
            error_y_minus_arg = y_err_lower_col # This is for the lower error bar
    #error_y_arg = y_err_col if show_error_bars else None
    reject_list = ['t_start', 't_stop', 'det', 'trigger_number', 'peak_time_ratio', 'background_level', 'pulse']
    hover_cols = [col for col in plot_df.columns if col not in reject_list]

    # 5. Create the figure
    fig = px.scatter(
        plot_df,
        x=x_axis_col,
        y=y_axis_col,
        color=group_by_col,
        symbol=symbol_arg,
        error_y=error_y_arg,
        error_y_minus=error_y_minus_arg,
        log_x=use_log_x,
        log_y=use_log_y,
        labels={
            x_axis_col: x_axis_col.replace('_', ' ').title(),
            y_axis_col: y_axis_col.replace('_', ' ').title(),
            group_by_col: color_legend_title
        },
        #title=f"{y_axis_col.replace('_', ' ').title()} vs. {x_axis_col.replace('_', ' ').title()}",
        template="plotly_white",
        hover_data=hover_cols,
        symbol_map={'Lower Limit': 'diamond-open'}
    )

    # 6. Apply custom theme styling
    theme_styles = {
        "Contrast White": {'paper_bgcolor': "white", 'plot_bgcolor': "white", 'font_color': "black", 'gridcolor': '#D3D3D3', 'zerolinecolor': '#C0C0C0'},
        "Contrast Dark": {'paper_bgcolor': "#1E1E1E", 'plot_bgcolor': "#2E2E2E", 'font_color': "white", 'gridcolor': '#4A4A4A', 'zerolinecolor': '#7A7A7A'},
        "Seaborn-like": {'paper_bgcolor': "#F0F2F6", 'plot_bgcolor': "#F0F2F6", 'font_color': "black", 'gridcolor': 'white', 'zerolinecolor': 'white'}
    }
    font_color = "black"
    # --- NEW: Define border color based on theme ---
    border_color = "black" 
    if plot_theme in theme_styles:
        style = theme_styles[plot_theme]
        font_color = style['font_color']
        if plot_theme == "Contrast Dark":
            border_color = "white" # Use white border for dark theme
        fig.update_layout(paper_bgcolor=style['paper_bgcolor'], plot_bgcolor=style['plot_bgcolor'], font_color=font_color)
        fig.update_xaxes(gridcolor=style['gridcolor'], zerolinecolor=style['zerolinecolor'])
        fig.update_yaxes(gridcolor=style['gridcolor'], zerolinecolor=style['zerolinecolor'])
    elif plot_theme == "simple_white":
        border_color = "white"
        font_color = "white"
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", font_color="black")
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', zerolinecolor='gray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', zerolinecolor='gray')

    # 7. Final layout updates for publication quality
    fig.update_traces(
        marker=dict(
            size=12,
            # --- NEW: Add a border to all markers ---
            line=dict(
                width=1.5,
                color=border_color  # Use the theme-aware border color
            )
        ),
        error_y=dict(thickness=2.0)
    )

    fig.update_layout(
        height=plot_height,
        title_x=0.5,
        xaxis_range=x_range,
        yaxis_range=y_range,
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


import pandas as pd
import numpy as np
import plotly.express as px

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
    # ... (Filtering logic is the same) ...
    error_col_for_filtering = y_err_lower_col or y_err_upper_col
    if not show_lower_limits and error_col_for_filtering and error_col_for_filtering in df.columns:
        plot_df = df[df[error_col_for_filtering] > 0].copy()
    else:
        plot_df = df.copy()

    if filters:
        for key, value in filters.items():
            plot_df = plot_df[plot_df[key].isin(value)]

    if plot_df.empty:
        # Using st.warning requires importing streamlit as st in this file
        # For simplicity, we just return None and let the app handle the message.
        return None
    
    # --- NEW: Check if we should use the advanced intensity coloring ---
    use_intensity_coloring = (
        color_intensity_col is not None and
        'successful_runs' in plot_df.columns and
        'total_sim' in plot_df.columns
    )

    # Prepare columns for either plotting mode
    plot_df[group_by_col] = plot_df[group_by_col].astype(str)
    if marker_col:
        plot_df[marker_col] = plot_df[marker_col].astype(str)
    
    if use_intensity_coloring:
        plot_df['success_percent'] = (plot_df['successful_runs'] / plot_df['total_sim']) * 100
        color_arg = 'success_percent'
        symbol_arg = group_by_col # Use group for symbol in this mode
    else:
        color_arg = group_by_col # Use group for color in simple mode
        symbol_arg = marker_col

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
        color_continuous_scale='Viridis', # Used only for intensity mode
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
    # 6. Apply custom theme styling
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


    # --- Inside your plot_plotly function, replace the final fig.update_layout call ---
    # --- Inside your plot_plotly function, replace the final fig.update_layout call ---

    fig.update_layout(
        height=plot_height,
        title_x=0.5,
        xaxis_range=x_range,
        yaxis_range=y_range,
        # <<< NEW: Conditionally hide the legend >>>
        showlegend=(not use_intensity_coloring),
        font=dict(
            family="Arial, sans-serif",
            size=18,
            color=font_color
        ),
        title_font_size=24,
        xaxis=dict(title_font_size=22),
        yaxis=dict(title_font_size=22),
        # <<< CHANGE: Manual positioning is no longer needed >>>
        legend=dict(
            title_font_size=20,
            font_size=18,
            traceorder="normal"
        )
    )

    return fig

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
    

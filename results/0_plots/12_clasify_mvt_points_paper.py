# batch_classify_plot.py
import numpy as np
import matplotlib.pyplot as plt

from clasify_mvt_point import load_model, classify_mvt_point

# ---- your data ----
grbs = {
  "GRB 211211A": {"median mvt": 3.9, "mvt err hi": 1.0, "mvt err low": 1.0, "SNR MVT": 116.4},
  "GRB 230307A": {"median mvt": 1.7, "mvt err hi": 0.7, "mvt err low": 0.4, "SNR MVT": 90.9},
  "GRB 170817A": {"median mvt": 344.6, "mvt err hi": 133.9, "mvt err low": 141.7, "SNR MVT": 7.2},
  "GRB 231115A": {"median mvt": 9.9, "mvt err hi": 2.1, "mvt err low": 2.5, "SNR MVT": 24.1},
  "GRB 250919A": {"median mvt": 28.8, "mvt err hi": 10.7, "mvt err low": 5.5, "SNR MVT": 180.5},
}

# --------- Plot config (match your single-point script style) ----------
FONT_LABEL  = 20
FONT_TICK   = 16
FONT_LEGEND = 12
X_MIN, X_MAX = 10, 2000
Y_MIN, Y_MAX = 0.3, 3000

def main():
    model = load_model()
    if model is None:
        return
    interpolators, mvt_range_log = model

    # Classify all first (also capture bounds if you want)
    classes = {}
    for name, v in grbs.items():
        c, _ = classify_mvt_point(v["median mvt"], v["SNR MVT"], interpolators, mvt_range_log)
        # Fix small typo in original label
        c = c.replace("Robast", "Robust")
        classes[name] = c

    fig, ax = plt.subplots(figsize=(8, 7))

    # --- Model band and median line (CORRECT ORIENTATION) ---
    mvt_grid_log = np.linspace(mvt_range_log[0], mvt_range_log[1], 400)
    snr_lower_log = interpolators['lower'](mvt_grid_log)
    snr_upper_log = interpolators['upper'](mvt_grid_log)
    snr_median_log = interpolators['median'](mvt_grid_log)

    # Fill between SNR lower/upper along vertical axis of MVT => use fill_betweenx
    ax.fill_betweenx(
        10**mvt_grid_log, 10**snr_lower_log, 10**snr_upper_log,
        color='orange', alpha=0.2, zorder=1, label='95% CI (Bootstrap)'
    )

    # Median line: x = SNR(mvt), y = MVT
    ax.plot(10**snr_median_log, 10**mvt_grid_log, color='red', lw=3, zorder=2, label='Median fit')

    tag_map = {
    "Below 95% CI (Upper Limit)"           : "UL",
    "Within 95% CI (Likely Upper Limit)"   : "LUL",
    "Above 95% CI (Robust Measurement)"    : "R",
    }


    # --- Color map by class ---
    color_map = {
    "Below 95% CI (Upper Limit)"           : "red",     # UL
    "Within 95% CI (Likely Upper Limit)"   : "blue",    # LUL
    "Above 95% CI (Robust Measurement)"    : "green",   # robust
    }


    # --- Plot GRB points (with optional vertical error bars from MVT errors) ---
    for name, v in grbs.items():
        x = float(v["SNR MVT"])
        y = float(v["median mvt"])
        yerr = None
        if "mvt err hi" in v and "mvt err low" in v:
            # guard against non-positive lower error on log axis
            lo = max(1e-6, float(v["mvt err low"]))
            hi = max(1e-6, float(v["mvt err hi"]))
            yerr = [[lo], [hi]]

        cls = classes[name]
        color = color_map.get(cls, "gray")

        # errorbar draws the marker and optional vertical bars
        ax.errorbar(
            x, y, yerr=yerr, fmt='o', ms=7, color=color, ecolor=color,
            elinewidth=1.5, capsize=3, mec='k', mew=0.8, zorder=10
        )
        # label slightly to the right
        #ax.text(x * 1.06, y, name, fontsize=10, va='center')
        #ax.text(x * 1.06, y, f"{name} ({tag_map[cls]})", fontsize=10, va='center')
        # name in black
        ax.text(x * 1.06, y, f"{name}", fontsize=14, va='center', ha='left', color='black', bbox=dict(facecolor='white', edgecolor='none', pad=1.0, alpha=0.7))

        # colored tag
        ax.text(x * 1.06 * 1.10, y*.75, f"({tag_map[cls]})", fontsize=14, va='center', ha='left', color=color, fontweight='bold')


    # --- Axes formatting ---
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(X_MIN-5, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel(r'SNR$_{\mathrm{MVT}}$', fontsize=FONT_LABEL)
    ax.set_ylabel('MVT (ms)', fontsize=FONT_LABEL)
    ax.tick_params(axis='both', which='major', labelsize=FONT_TICK)
    ax.grid(True, which='both', ls='--', alpha=0.5)

    # Legend for classes
    handles = [
        plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=color_map[k],
                   markeredgecolor='k', markersize=9, label=k)
        for k in color_map.keys()
    ]
    # plus band/median entries
    from matplotlib.lines import Line2D
    band_patch = Line2D([], [], color='orange', lw=8, alpha=0.2, label='95% CI (Bootstrap)')
    median_line = Line2D([], [], color='red', lw=3, label='Median fit')
    all_handles = handles + [band_patch, median_line]
    all_labels = [h.get_label() for h in all_handles]
    ax.legend(all_handles, all_labels, fontsize=FONT_LEGEND, loc='upper right')

    plt.tight_layout()
    plt.savefig('12_batch_mvt_classification.png', dpi=200)
    plt.close()
    print("Saved: 12_batch_mvt_classification.png")

if __name__ == "__main__":
    main()

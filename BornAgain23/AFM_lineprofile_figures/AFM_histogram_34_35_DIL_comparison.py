from GISAXS_Analysis import height_radius_from_lineprofiles as h_r
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, pi

# ───────────── CONFIG ─────────────
#FILENAMES    = ["4824_40gPL_DIL_3gPL_16Pre_FAPbBr_48hrStir_Unetched_ManualLineprofiles.txt"]
DPI_SAVE     = 300
BINS         = 30        # int or "auto"
THRESHOLD    = 0.2       # fraction of peak height
#PLOT_FIRST   = 5         # how many profiles to plot with markers
FIGSIZE_HIST = (6, 4)
FIGSIZE_PROF = (7.5, 6.5)

# ────── always work in script’s folder ──────
#SCRIPT_DIR = Path(__file__).resolve().parent
#os.chdir(SCRIPT_DIR)

def plot_2d_histogram(x, y, bins=100, range=None, cmap="viridis"):
    """
    Plots a 2D histogram (heatmap) of (x, y) data.

    Parameters
    ----------
    x, y : np.ndarray
        Input arrays of equal length.
    bins : int or [int, int], optional
        Number of bins along each axis.
    range : [[xmin, xmax], [ymin, ymax]], optional
        Limits of the histogram. If None, determined from data.
    cmap : str, optional
        Colormap for the 2D histogram.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    h = ax.hist2d(x, y, bins=bins, range=range, cmap=cmap)
    plt.colorbar(h[3], ax=ax, label="Counts")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("2D Histogram of (X, Y)")
    plt.tight_layout()
    

    return fig, ax

# ────── helpers for Gaussian overlay ──────
def normal_pdf(x, mu, sigma):
    if sigma <= 0:
        return np.zeros_like(x)
    return (1.0 / (sigma * sqrt(2 * pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def add_gaussian_overlay(ax, data, bins):
    """Plot Gaussian fit + annotate μ and σ."""
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    if data.size == 0:
        return
    mu = float(np.mean(data))
    sigma = float(np.std(data, ddof=1)) if data.size > 1 else 0.0

    # Create a continuous overlay over the plotted histogram range
    if isinstance(bins, str) and bins == "auto":
        # infer bounds from data
        lo, hi = np.min(data), np.max(data)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            return
        # choose ~30 segments for overlay if auto
        xfit = np.linspace(lo, hi, 400)
        bin_width = (hi - lo) / 30.0 if hi > lo else 1.0
    else:
        # explicit bin edges or count
        if np.isscalar(bins):
            # re-derive edges from data with that bin count
            counts, edges = np.histogram(data, bins=int(bins))
        else:
            edges = np.asarray(bins)
        xfit = np.linspace(edges[0], edges[-1], 400)
        bin_width = np.diff(edges).mean() if len(edges) > 1 else 1.0

    yfit = len(data) * bin_width * normal_pdf(xfit, mu, sigma)
    ax.plot(xfit, yfit, lw=1.5, label="Gaussian fit")
    ax.text(
        0.98, 0.98,
        f"μ = {mu:.2f} nm\nσ = {sigma:.2f} nm",
        transform=ax.transAxes,
        ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.7),
    )

def plot_hist_with_fit(data, bins, title, xlabel, outfile_base):
    fig, ax = plt.subplots(figsize=FIGSIZE_HIST)
    ax.hist(data[~np.isnan(data)], bins=bins, edgecolor="k")
    add_gaussian_overlay(ax, data, bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    fig.tight_layout()
    #fig.savefig(f"{outfile_base}.png", dpi=DPI_SAVE)
    #fig.savefig(f"{outfile_base}.pdf", dpi=DPI_SAVE)
    return fig, ax


import numpy as np
import matplotlib.pyplot as plt

def plot_hist_overlaid_with_gaussians(
    datasets,
    labels=None,
    bins="auto",
    title="",
    xlabel="",
    ylabel="Frequency (0–1)",
    fig_size=(7.2, 4.5),
    alpha=0.45,          # bar transparency so overlaps are visible
    edgecolor="k",
    linewidth=4,
    show_legend=True,
):
    """
    Overlay *filled* histograms for one or more datasets, normalized to frequency (0–1),
    and add Gaussian overlays (μ, σ) for each dataset on the same y-scale.

    Notes
    -----
    • Frequencies: bar heights are counts/total, so each dataset sums to 1 across bins.
    • Gaussian overlay: y = PDF * avg_bin_width so the curve matches the frequency scale.
    • Uses shared bin edges derived from all data for apples-to-apples overlays.
    """

    # Ensure iterable
    if not hasattr(datasets, "__iter__") or isinstance(datasets, (str, bytes)):
        datasets = [datasets]

    # Clean NaNs/inf
    cleaned = []
    for d in datasets:
        a = np.asarray(d)
        a = a[np.isfinite(a)]
        cleaned.append(a)

    if labels is None:
        labels = [f"Series {i+1}" for i in range(len(cleaned))]

    # If everything empty, draw frame
    if not any(a.size for a in cleaned):
        fig, ax = plt.subplots(figsize=fig_size)
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.text(0.5, 0.5, "No finite data to plot", transform=ax.transAxes,
                ha="center", va="center")
        return fig, ax

    # Shared bin edges
    all_concat = np.concatenate([a for a in cleaned if a.size > 0])
    if np.isscalar(bins) or isinstance(bins, str):
        edges = np.histogram_bin_edges(all_concat, bins=bins)
    else:
        edges = np.asarray(bins, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("`bins` must be 1D with at least two edges.")

    widths = np.diff(edges)
    avg_w  = widths.mean()

    # Plot
    fig, ax = plt.subplots(figsize=fig_size)

    old_fc = []
    for data, lab in zip(cleaned, labels):
        if data.size == 0:
            continue

        counts, _ = np.histogram(data, bins=edges)
        total = counts.sum()
        freq = counts / total if total > 0 else counts.astype(float)

        # pick the next color from the cycle so bars and lines match

        # use FaceColor with transparency, EdgeColor opaque
        ax.bar(
            edges[:-1],
            freq,
            width=widths,
            align="edge",          # fill color (opaque base)
            edgecolor='k',          # same color for border
            linewidth=linewidth-1,            # match spines
            label=lab
        )
    
        # Fix transparency: reset edge alpha to 1
        
        for patch in ax.patches[-(len(edges) - 1):]:
            fc = patch.get_facecolor()
            ec = patch.get_edgecolor()
            # overwrite edgecolor to fully opaque (same RGB, alpha=1)
            patch.set_edgecolor((ec[0], ec[1], ec[2], 1.0))
            patch.set_facecolor((fc[0], fc[1], fc[2], 0.45))
        old_fc.append(fc)


    # Gaussian overlays scaled to frequency (PDF * avg_bin_width)
    xfit = np.linspace(edges[0], edges[-1], 600)
    for data, lab in zip(cleaned, labels):
        if data.size == 0:
            continue
        mu = float(np.mean(data))
        sigma = float(np.std(data, ddof=1)) if data.size > 1 else 0.0
        if sigma <= 0 or not np.isfinite(sigma):
            continue
        # uses your normal_pdf if already defined in your scope
        y_pdf = normal_pdf(xfit, mu, sigma)
        y_freq = y_pdf * avg_w  # scale PDF to frequency scale
        ax.plot(xfit, y_freq, lw=linewidth, label=f"{lab} Gaussian (μ={mu:.2f} nm, σ={sigma:.2f}) nm")
 
    # --- Labels & style ---
    ax.set_xlim(edges[0], edges[-1])
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(bottom=0, top = ymax * 1.25)
    ax.set_title(title)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)

    # Tick styling
    ax.tick_params(
        direction="in",          # ticks point inward
        length=5, width=linewidth,     # thicker, slightly longer ticks
        top=True, right=True,
        labelsize = 20    # show ticks on all sides
    )

    # Spine styling (axes border lines)
    for spine in ax.spines.values():
        spine.set_linewidth(linewidth)   # make all sides equally thick
    show_legend = True
    if show_legend:
        handles, labels_out = [], []

        # We'll get the color of each histogram patch group (1 per dataset)
        # Since bars are drawn sequentially for each dataset, take first patch of each group
        n_datasets = len(cleaned)
        n_bins = len(edges) - 1
        for i, data in enumerate(cleaned):
            if data.size == 0:
                continue
            # Each dataset created n_bins patches in order
            patch = ax.patches[i * n_bins]  # first patch of that dataset
            color = old_fc[i]

            mu = float(np.mean(data))
            sigma = float(np.std(data, ddof=1)) if data.size > 1 else 0.0
            #label_text = f"h={mu:.2f} nm, σ={sigma:.2f} nm"        #Label with h and sigma
            label_text = labels[i]

            # Make a colored line for legend
            handles.append(
                plt.Line2D([], [], color=color, lw=linewidth)
            )
            labels_out.append(label_text)

        ax.legend(handles, labels_out, frameon=False, ncol=1, handlelength=2.8,
        loc="upper left")

    fig.tight_layout()
    return fig, ax




# Example usage:
# plot_2d_histogram(x, y)

# Minimal test — adjust file path as needed
lineprofile_dir =  r"C:\Users\Pedro\OneDrive - McMaster University\PhD - School\Research\Projects\X Ray Scattering and Diffraction\AFM Analysis\Dilution Experiments\3gPL_unetched\4824_40gPL_DIL_3gPL_16Pre_FAPbBr_48hrStir_Unetched_ManualLineprofiles.txt"

xc, yc = h_r.load_lineprofiles(lineprofile_dir)
hsub_nm_dil, dmin_nm_dil = h_r.extract_hsub_and_dmin(xc, yc, frac=0.2)

lineprofile_dir =  r"C:\BornAgainSimulations\data\AFM-lineprofiles\lineProfiles_35_Big_OnePerParticle.txt"

xc, yc = h_r.load_lineprofiles(lineprofile_dir)
hsub_nm_35, dmin_nm_35 = h_r.extract_hsub_and_dmin(xc, yc, frac=0.2)

lineprofile_dir =  r"C:\BornAgainSimulations\data\AFM-lineprofiles\lineProfiles_34_Big_OnePerParticle_2.txt"

xc, yc = h_r.load_lineprofiles(lineprofile_dir)
hsub_nm_34, dmin_nm_34 = h_r.extract_hsub_and_dmin(xc, yc, frac=0.2)



stem = 'dil'

#plot_hist_with_fit(
#            hsub_nm_dil,
#            BINS,
#            f"Histogram — Raw maxima Dil",
#            "Maximum height per profile (nm)",
#            f"{stem}_hist_maxima",
#        )

#plot_2d_histogram(dmin_nm_dil, hsub_nm_dil, bins=20)
#plot_2d_histogram(dmin_nm_dil, hsub_nm_dil, bins=40)
#plot_2d_histogram(dmin_nm_dil, hsub_nm_dil, bins=60)
#fig, ax = plt.subplots(figsize=(6, 5))
#plt.hist(hsub_nm_dil)
#fig, ax = plt.subplots(figsize=(6, 5))
#plt.hist(hsub_nm_35)
#plt.scatter(dmin_nm_dil, hsub_nm_dil, color='black')
#plt.scatter(dmin_nm_35, hsub_nm_35, color='red')

plot_hist_overlaid_with_gaussians([dmin_nm_dil, dmin_nm_34, dmin_nm_35], ['After Dilution (monolayer)', 'Polymer-Perovskite (multi-layer)', 'Polymer only (multi-layer)'], xlabel='diameter (nm)', ylabel='Probability Density', fig_size=(7,5))
plt.savefig('AFM_Comparison_Histogram_Dil_34_35_diameter_noLegend.png', dpi = DPI_SAVE)
plt.savefig('AFM_Comparison_Histogram_Dil_34_35_diameter_noLegend.pdf', dpi = DPI_SAVE)

plot_hist_overlaid_with_gaussians([hsub_nm_dil, hsub_nm_34, hsub_nm_35], ['After Dilution (monolayer)', 'Polymer-Perovskite (multi-layer)', 'Polymer only (multi-layer)'], xlabel='height (nm)', ylabel='Probability Density', bins=24, fig_size=(7,5))
plt.savefig('AFM_Comparison_Histogram_Dil_34_35_height_noLegend.png', dpi= DPI_SAVE)
plt.savefig('AFM_Comparison_Histogram_Dil_34_35_height_noLegend.pdf', dpi= DPI_SAVE)

plt.show()

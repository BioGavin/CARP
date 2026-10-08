"""Shared matplotlib style for the CARP paper figures.

Every plotting script in this directory does ``from _style import ...`` and calls
``apply()`` once before building figures, so all panels share fonts, sizes, colors,
and output formats. Sizes target a two-column journal page (full width ~7.1 in).
"""
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots in fixed order (validated for colorblind separation on adjacent
# pairs). Assign by entity, never by rank; <=3 slots when every pair must separate.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"      # primary text
INK2 = "#52514e"     # secondary text / annotations
MUTED = "#8a8985"    # reference lines
GRID = "#e4e3df"     # recessive grid

FULL_W = 7.1         # double-column width (in)
HALF_W = 3.4         # single-column width (in)
BAR_W = 0.13         # bar width (in), shared by every bar chart
BAR_GAP = 0.02       # gap between bars inside a group (in)
GROUP_GAP = 0.30     # gap between neighbouring groups (in); fits a tick label like "AUROC"
# In-panel secondary annotations -- bar value labels, percentages, bracket captions -- as a
# fraction of the tick-label size. One constant so a composite figure cannot show the same
# kind of annotation at three different sizes.
ANNOT_FS = 0.92


def apply(font_scale=1.0):
    """Set the shared rcParams; ``font_scale`` multiplies every font size."""
    fs = font_scale
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8 * fs,
        "axes.titlesize": 9 * fs,
        "axes.labelsize": 8 * fs,
        "xtick.labelsize": 7.5 * fs,
        "ytick.labelsize": 7.5 * fs,
        "legend.fontsize": 7.5 * fs,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK2,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "mathtext.default": "regular",
        "mathtext.fontset": "custom",   # $\it{E. coli}$ in the body font, not DejaVu
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "pdf.fonttype": 42,          # embed TrueType -> text stays editable in Illustrator
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def panel_label(ax, s):
    """Bold (a)/(b) tag at the top-left corner of an axes."""
    ax.set_title(s, loc="left", fontweight="bold")


def bar_x(n_groups, n_per):
    """Bar geometry in inches (use the result as x data, with bar width ``BAR_W``).

    -> (pos, centers, xlim): ``pos[k]`` are the x of bar k in every group, ``centers``
    the group centers (tick positions), ``xlim`` leaves half a group gap at each end.
    Pair with ``fit_size`` so one data unit is one inch on paper (use these as y
    positions for horizontal bars)."""
    pitch = n_per * BAR_W + (n_per - 1) * BAR_GAP + GROUP_GAP
    centers = np.arange(n_groups) * pitch
    offsets = (np.arange(n_per) - (n_per - 1) / 2) * (BAR_W + BAR_GAP)
    return centers[None, :] + offsets[:, None], centers, (-pitch / 2, centers[-1] + pitch / 2)


def group_layout(n_groups, n_per, group_frac=None):
    """-> (pos, centers, span, bar_w) for grouped bars.

    ``group_frac=None``: the shared physical geometry of ``bar_x`` (inches; pair with
    ``fit_size``). Otherwise a plain data-unit layout for fixed-size figures: each group
    gets one unit, its bars fill ``group_frac`` of it with a hairline gap between them."""
    if group_frac is None:
        pos, centers, span = bar_x(n_groups, n_per)
        return pos, centers, span, BAR_W
    centers = np.arange(n_groups, dtype=float)
    step = group_frac / n_per
    offsets = (np.arange(n_per) - (n_per - 1) / 2) * step
    return (centers[None, :] + offsets[:, None], centers, (-0.5, n_groups - 0.5),
            step * 0.92)


def fit_size(fig, ax, layout, axis="x", iters=20):
    """Resize the figure so ``ax`` spans exactly its data range in inches along ``axis``.

    With bar positions in inches (``bar_x``), bars, in-group gaps and group gaps then
    have the same physical size in every figure: axis="x" fits the figure width (vertical
    bars), axis="y" the height (horizontal bars). ``layout`` re-runs the figure's
    tight_layout; axes sharing that dimension should use gridspec width/height ratios
    equal to their data ranges."""
    lo, hi = ax.get_xlim() if axis == "x" else ax.get_ylim()
    get_size = fig.get_figwidth if axis == "x" else fig.get_figheight
    set_size = fig.set_figwidth if axis == "x" else fig.set_figheight
    frac = (lambda: ax.get_position().width) if axis == "x" else (lambda: ax.get_position().height)
    target, prev = abs(hi - lo), None
    for _ in range(iters):
        layout()
        size = get_size()
        actual = frac() * size
        if abs(actual - target) < 1e-3:
            break
        # d(axes size)/d(figure size): the axes' share of the figure at first (an
        # underestimate, margins are fixed), then the secant through the last two steps
        slope = frac() if prev is None else (actual - prev[1]) / (size - prev[0])
        prev = (size, actual)
        set_size(size + (target - actual) / max(slope, 1e-3))


def save_csv(df, figdir, name, **kw):
    """Write a plot's source table to <figdir>/data/<name> (the one place CSVs live)."""
    path = os.path.join(figdir, "data", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, **kw)
    print(f"csv    -> {path}")
    return path


def save(fig, base, exts=("pdf",), **savefig_kw):
    """Write <base>.<ext> for each ext. ``savefig_kw`` goes straight to ``fig.savefig``;
    pass ``bbox_inches=fig.bbox_inches, pad_inches=0`` to keep the figure's own size.
    The shared rcParams crop to a tight bbox, and ``bbox_inches=None`` will NOT turn that
    off -- matplotlib reads None as "use rcParams" -- so an explicit Bbox is the only way
    to get a page of exactly the requested size."""
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    for ext in exts:
        fig.savefig(f"{base}.{ext}", **savefig_kw)
    plt.close(fig)
    for ext in exts:
        print(f"figure -> {base}.{ext}")

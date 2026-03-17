"""
Fluent .prof File Inspector
============================
Parses and visualizes Fluent profile (.prof) files.
Handles cross-sectional planes (constant x) with 2D scatter plots (y vs z).

Requires: numpy, matplotlib
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from pathlib import Path
from itertools import cycle

# ══════════════════════════════════════════════════════════════
#  ►  SET YOUR FOLDER PATH HERE
# ══════════════════════════════════════════════════════════════
PROF_FOLDER = r"Q:\IPK-Projekte\OE31000_Brooks_Digital-Twin\05_Arbeitsunterlagen\10_BRKS\Nozzles\Export_Vel_Profile"
# ══════════════════════════════════════════════════════════════

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0f1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#e6edf3",
    "axes.titlecolor":   "#e6edf3",
    "axes.grid":         True,
    "grid.color":        "#21262d",
    "grid.linewidth":    0.5,
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#e6edf3",
    "legend.facecolor":  "#161b22",
    "legend.edgecolor":  "#30363d",
    "font.family":       "monospace",
    "font.size":         9,
})

PALETTE = ["#58a6ff", "#f78166", "#3fb950", "#d2a8ff",
           "#ffa657", "#79c0ff", "#ff7b72", "#56d364"]

# ── Units lookup ───────────────────────────────────────────────────────────────
FIELD_UNITS = {
    "pressure":            "Pa",
    "total-pressure":      "Pa",
    "velocity-magnitude":  "m/s",
    "x-velocity":          "m/s",
    "y-velocity":          "m/s",
    "z-velocity":          "m/s",
    "density":             "kg/m³",
    "turb-kinetic-energy": "m²/s²",
    "turb-diss-rate":      "m²/s³",
    "temperature":         "K",
    "x":                   "m",
    "y":                   "m",
    "z":                   "m",
}

def field_label(name: str) -> str:
    """Return 'fieldname [unit]' if unit known, else just 'fieldname'."""
    unit = FIELD_UNITS.get(name.lower())
    return f"{name} [{unit}]" if unit else name

def pct_clim(vals: np.ndarray, lo: float = 2.0, hi: float = 98.0):
    """Percentile-based colour limits — ignores outliers."""
    return float(np.percentile(vals, lo)), float(np.percentile(vals, hi))

# ── Parser ─────────────────────────────────────────────────────────────────────

def parse_prof(filepath: str) -> dict:
    """
    Bracket-counting parser for Fluent .prof files.
    Handles values spread across multiple lines.
    Returns: { surface_name: { field_name: np.ndarray } }
    """
    with open(filepath, "r") as f:
        text = f.read()

    surfaces = {}
    n = len(text)
    i = 0

    while i < n:
        # A surface block starts with ((
        if text[i] == "(" and i + 1 < n and text[i + 1] == "(":
            # Parse the surface header: ((name type count)
            header_end = text.index(")", i + 1)
            header     = text[i + 2 : header_end].split()
            surf_name  = header[0] if header else f"surface_{len(surfaces)}"

            # Find the matching closing paren of the outer block
            depth = 1
            k = header_end + 1
            while k < n and depth > 0:
                if   text[k] == "(": depth += 1
                elif text[k] == ")": depth -= 1
                k += 1
            block = text[header_end + 1 : k - 1]

            # Parse each field sub-block (field_name val val ...)
            fields = {}
            bi, bn = 0, len(block)
            while bi < bn:
                if block[bi] == "(":
                    fj, fd = bi + 1, 1
                    while fj < bn and fd > 0:
                        if   block[fj] == "(": fd += 1
                        elif block[fj] == ")": fd -= 1
                        fj += 1
                    tokens = block[bi + 1 : fj - 1].split()
                    if tokens:
                        field_name = tokens[0]
                        try:
                            values = np.array([float(t) for t in tokens[1:]])
                            if values.size > 0:
                                fields[field_name] = values
                        except ValueError:
                            pass
                    bi = fj
                else:
                    bi += 1

            if fields:
                # Handle duplicate surface names across files
                key = surf_name
                c = 1
                while key in surfaces:
                    key = f"{surf_name}_{c}"; c += 1
                surfaces[key] = fields
            i = k
        else:
            i += 1

    if not surfaces:
        raise ValueError(f"No surface blocks found in {filepath}.")
    return surfaces


def load_files(paths: list) -> dict:
    """Returns { 'filestem/surfname': { field: array } }"""
    all_data = {}
    for p in paths:
        print(f"  Loading: {Path(p).name}")
        data = parse_prof(p)
        stem = Path(p).stem
        for surf, fields in data.items():
            key = f"{stem}/{surf}"
            all_data[key] = fields
            print(f"    └─ '{surf}'  pts={next(iter(fields.values())).size}"
                  f"  fields={list(fields.keys())}")
    return all_data

# ── Helpers ────────────────────────────────────────────────────────────────────

COORD_KEYS = {"x", "y", "z", "arc-length", "arc_length", "radial-coordinate"}

def scalar_fields(fields: dict) -> list:
    return [k for k in fields if k.lower() not in COORD_KEYS]

# ── Plot 1: 2D scatter map for every field ─────────────────────────────────────

def plot_2d_maps(all_data: dict, output_dir: Path):
    """
    For a cross-sectional plane (x ≈ const), scatter-plot y vs z
    coloured by each scalar field.
    """
    cmaps = ["plasma", "inferno", "viridis", "cividis",
             "magma", "coolwarm", "RdYlBu_r", "turbo"]

    for label, fields in all_data.items():
        file_stem, surf_name = label.split("/", 1)

        if "y" not in fields or "z" not in fields:
            print(f"  Skipping 2D map for '{surf_name}' — no y/z coords.")
            continue

        y  = fields["y"]
        z  = fields["z"]
        sf = scalar_fields(fields)
        if not sf:
            continue

        ncols = min(3, len(sf))
        nrows = (len(sf) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5.5 * ncols, 4.5 * nrows),
                                 constrained_layout=True)
        axes = np.array(axes).flatten() if len(sf) > 1 else [axes]

        x_info = f"x ≈ {fields['x'].mean():.5f}" if "x" in fields else ""
        fig.suptitle(f"Surface: {surf_name}   [{x_info}]   —   {file_stem}",
                     fontsize=12, fontweight="bold")

        for idx, (field, cmap_name) in enumerate(zip(sf, cycle(cmaps))):
            ax   = axes[idx]
            vals = fields[field]

            if vals.size != y.size:
                ax.set_visible(False)
                continue

            vmin, vmax = pct_clim(vals)
            sc   = ax.scatter(y, z, c=vals, cmap=cmap_name,
                              vmin=vmin, vmax=vmax,
                              s=22, alpha=0.88, edgecolors="none")
            cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046,
                                extend="both")   # arrows show clipped outliers
            cbar.ax.tick_params(labelsize=7, colors="#8b949e")
            cbar.set_label(field_label(field), fontsize=7, color="#8b949e")

            ax.set_xlabel("y  [m]", fontsize=8)
            ax.set_ylabel("z  [m]", fontsize=8)
            ax.set_title(field_label(field), fontsize=9, fontweight="bold")
            ax.set_aspect("equal", adjustable="box")

            # Stats annotation inside plot (true min/max shown so you still see them)
            ax.annotate(
                f"min {vals.min():.3g}   μ {vals.mean():.3g}   max {vals.max():.3g}",
                xy=(0.02, 0.02), xycoords="axes fraction",
                fontsize=6.5, color="#8b949e",
                bbox=dict(boxstyle="round,pad=0.3", fc="#0f1117", alpha=0.7)
            )

        for idx in range(len(sf), len(axes)):
            axes[idx].set_visible(False)

        fname = output_dir / f"{surf_name}_2d_maps.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Saved → {fname.name}")
        plt.show()

# ── Plot 2: velocity vector field ─────────────────────────────────────────────

def plot_velocity_vectors(all_data: dict, output_dir: Path):
    """
    Tricontourf background (velocity magnitude) + quiver arrows (vy, vz).
    """
    for label, fields in all_data.items():
        file_stem, surf_name = label.split("/", 1)

        if not {"y", "z", "y-velocity", "z-velocity"}.issubset(fields):
            print(f"  Skipping vector plot for '{surf_name}' — missing y/z-velocity.")
            continue

        y  = fields["y"]
        z  = fields["z"]
        vy = fields["y-velocity"]
        vz = fields["z-velocity"]
        vm = fields.get("velocity-magnitude",
                        np.sqrt(vy**2 + vz**2 + fields.get("x-velocity", 0)**2))

        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        fig.suptitle(f"{surf_name}  —  velocity vectors", fontsize=12, fontweight="bold")

        # tricontourf background
        try:
            triang = tri.Triangulation(y, z)
            tcf    = ax.tricontourf(triang, vm, levels=24, cmap="plasma", alpha=0.75)
            cbar   = fig.colorbar(tcf, ax=ax, label="|v|  [m/s]",
                                  fraction=0.046, pad=0.02)
            cbar.ax.tick_params(colors="#8b949e")
        except Exception:
            sc   = ax.scatter(y, z, c=vm, cmap="plasma", s=18, alpha=0.8)
            cbar = fig.colorbar(sc, ax=ax, label="|v|  [m/s]",
                                fraction=0.046, pad=0.02)
            cbar.ax.tick_params(colors="#8b949e")

        # Quiver — subsample to ~50 arrows
        step = max(1, len(y) // 50)
        ax.quiver(y[::step], z[::step], vy[::step], vz[::step],
                  color="#e6edf3", alpha=0.8, width=0.003)

        ax.set_xlabel("y  [m]", fontsize=9)
        ax.set_ylabel("z  [m]", fontsize=9)
        ax.set_aspect("equal")

        fname = output_dir / f"{surf_name}_vectors.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Saved → {fname.name}")
        plt.show()

# ── Plot 3: side-by-side comparison (same field, both surfaces) ────────────────

def plot_surface_comparison(all_data: dict, output_dir: Path):
    """
    Same field plotted side by side across all surfaces, shared colour scale.
    """
    labels = list(all_data.keys())
    sf_all = [set(scalar_fields(all_data[l])) for l in labels]
    common = sorted(set.intersection(*sf_all)) if sf_all else []
    if not common or len(labels) < 2:
        return

    cmaps   = ["plasma", "inferno", "viridis", "cividis",
               "magma",  "turbo", "coolwarm", "RdYlBu_r"]
    n_surf  = len(labels)
    n_field = len(common)

    fig, axes = plt.subplots(n_field, n_surf,
                             figsize=(7 * n_surf, 6 * n_field),
                             constrained_layout=True, squeeze=False)
    fig.suptitle("Cross-surface comparison  (shared colour scale per row)",
                 fontsize=12, fontweight="bold")

    for fi, field in enumerate(common):
        vmin = float(np.percentile(
            np.concatenate([all_data[l][field] for l in labels]), 2))
        vmax = float(np.percentile(
            np.concatenate([all_data[l][field] for l in labels]), 98))
        cmap = cmaps[fi % len(cmaps)]

        for si, label in enumerate(labels):
            ax     = axes[fi][si]
            f      = all_data[label]
            _, sn  = label.split("/", 1)

            if "y" not in f or "z" not in f:
                ax.set_visible(False)
                continue

            sc = ax.scatter(f["y"], f["z"], c=f[field],
                            cmap=cmap, vmin=vmin, vmax=vmax,
                            s=18, alpha=0.88, edgecolors="none")
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(sn, fontsize=8, fontweight="bold")
            ax.set_xlabel("y  [m]", fontsize=7)
            if si == 0:
                ax.set_ylabel(f"{field_label(field)}\n\nz  [m]", fontsize=8)
            if si == n_surf - 1:
                cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04,
                                    extend="both")
                cbar.set_label(field_label(field), fontsize=7, color="#8b949e")
                cbar.ax.tick_params(labelsize=7, colors="#8b949e")

    fname = output_dir / "surface_comparison.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Saved → {fname.name}")
    plt.show()

# ── Plot 4: mean ± std bar chart ───────────────────────────────────────────────

def plot_stats_bars(all_data: dict, output_dir: Path):
    labels      = list(all_data.keys())
    surf_labels = [l.split("/", 1)[1] for l in labels]
    sf_union    = sorted({f for flds in all_data.values()
                          for f in scalar_fields(flds) if f != "density"})
    if not sf_union:
        return

    n = len(sf_union)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4), constrained_layout=True)
    if n == 1:
        axes = [axes]
    fig.suptitle("Mean ± Std  per surface", fontsize=12, fontweight="bold")

    bar_colors = list(PALETTE)[:len(labels)]

    for ax, field in zip(axes, sf_union):
        means = [float(np.nanmean(all_data[l].get(field, [np.nan]))) for l in labels]
        stds  = [float(np.nanstd (all_data[l].get(field, [np.nan]))) for l in labels]
        x     = np.arange(len(labels))

        ax.bar(x, means, yerr=stds, color=bar_colors, alpha=0.82,
               capsize=5, error_kw={"elinewidth": 1.5, "ecolor": "#8b949e"})
        ax.set_xticks(x)
        ax.set_xticklabels(surf_labels, rotation=20, ha="right", fontsize=7)
        ax.set_title(field, fontsize=9, fontweight="bold")

    fname = output_dir / "stats_bars.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Saved → {fname.name}")
    plt.show()

# ── Console summary ────────────────────────────────────────────────────────────

def print_summary(all_data: dict):
    SEP = "─" * 76
    for label, fields in all_data.items():
        file_stem, surf_name = label.split("/", 1)
        n_pts = next(iter(fields.values())).size
        print(f"\n{'═'*76}")
        print(f"  FILE: {file_stem}.prof   SURFACE: {surf_name}   POINTS: {n_pts}")
        print(f"{'═'*76}")
        print(f"  {'Field':<30} {'Min':>11} {'Mean':>11} {'Max':>11} {'Std':>11}")
        print(f"  {SEP}")
        for field, vals in fields.items():
            print(f"  {field:<30} {vals.min():>11.4g} {vals.mean():>11.4g} "
                  f"{vals.max():>11.4g} {vals.std():>11.4g}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    folder = Path(PROF_FOLDER)
    paths  = sorted(folder.glob("*.prof"))

    if not paths:
        print(f"\nNo .prof files found in:\n  {folder}\n"
              "Check that PROF_FOLDER is set correctly at the top of the script.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("  Fluent .prof Inspector")
    print(f"{'='*60}\n")

    all_data   = load_files([str(p) for p in paths])
    output_dir = folder

    print_summary(all_data)
    print(f"\n  Generating plots  →  {output_dir}\n")

    plot_2d_maps(all_data, output_dir)           # coloured scatter per field
    plot_velocity_vectors(all_data, output_dir)  # contour + quiver
    if len(all_data) > 1:
        plot_surface_comparison(all_data, output_dir)  # side-by-side
    plot_stats_bars(all_data, output_dir)        # mean ± std bars

    print("\n  Done. ✓\n")


if __name__ == "__main__":
    main()

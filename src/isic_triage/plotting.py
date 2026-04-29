"""Plotting helpers used by the ISIC 2024 notebook."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

DEFAULT_COLORS = {
    'healthy': '#2ca02c',
    'malignant': '#d62728',
    'benign': 'skyblue',
    'bar_fill': 'lightgrey',
    'risk_line': 'darkred',
}


def show_images(image_ids, images_dir=None, titles=None, max_cols=4, img_size=(3, 3)):
    """Display a small grid of lesion images by ISIC id."""
    if images_dir is None:
        images_dir = Path('data/train-image/image')
    else:
        images_dir = Path(images_dir)

    n = len(image_ids)
    if n == 0:
        return

    n_cols = min(n, max_cols)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(img_size[0] * n_cols, img_size[1] * n_rows),
    )
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, img_id in enumerate(image_ids):
        fname = f'{img_id}.jpg' if not str(img_id).endswith('.jpg') else str(img_id)
        img_path = images_dir / fname

        if img_path.exists():
            axes[i].imshow(plt.imread(img_path))
        else:
            axes[i].text(0.5, 0.5, 'not found', ha='center', va='center')

        if titles is not None and i < len(titles):
            axes[i].set_title(titles[i], fontsize=9)
        axes[i].axis('off')

    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def add_volume_risk_legend(ax, line_color=None, colors=None):
    colors = colors or DEFAULT_COLORS
    if line_color is None:
        line_color = colors['risk_line']
    legend_elements = [
        Patch(facecolor=colors['bar_fill'], edgecolor='grey', label='Volume (Count)'),
        Line2D([0], [0], color=line_color, marker='o', markersize=8, linewidth=2, label='Risk (Probability)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=True, fontsize=11)


def format_axis_as_percent(ax, axis='y'):
    formatter = FuncFormatter(lambda x, _: f'{x:.1%}')
    if axis == 'y':
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)


def show_family_table_simple(df, features, target, decimals=3):
    """Summarize univariate target association for a feature family."""
    rows = []
    for feature in features:
        x = pd.to_numeric(df[feature], errors='coerce')
        y = df[target].astype(int)

        mask = x.notna()
        x_valid, y_valid = x[mask], y[mask]
        x0 = x_valid[y_valid == 0].values
        x1 = x_valid[y_valid == 1].values

        try:
            auc_raw = roc_auc_score(y_valid, x_valid)
            auc_dir = 'higher→malig' if auc_raw >= 0.5 else 'lower→malig'
            auc_value = max(auc_raw, 1 - auc_raw)
        except Exception:
            auc_value, auc_dir = np.nan, 'n/a'

        if len(x0) > 0 and len(x1) > 0:
            delta_med = float(np.median(x1) - np.median(x0))
            u_stat = mannwhitneyu(x1, x0, alternative='two-sided').statistic
            cliff_delta = float((2 * u_stat) / (len(x0) * len(x1)) - 1)
        else:
            delta_med, cliff_delta = np.nan, np.nan

        med = float(x_valid.median()) if len(x_valid) else np.nan
        q25 = float(x_valid.quantile(0.25)) if len(x_valid) else np.nan
        q75 = float(x_valid.quantile(0.75)) if len(x_valid) else np.nan

        rows.append({
            'Feature': feature,
            'AUC': auc_value,
            'AUC_dir': auc_dir,
            'Δmedian': delta_med,
            'Cliffδ': cliff_delta,
            'median': med,
            'IQR': f'[{q25:.{decimals}g}, {q75:.{decimals}g}]' if pd.notna(q25) else '',
        })

    out = pd.DataFrame(rows)
    out = (
        out.assign(_abs_cliff=out['Cliffδ'].abs())
        .sort_values(['AUC', '_abs_cliff'], ascending=[False, False])
        .drop(columns='_abs_cliff')
        .reset_index(drop=True)
    )

    styled = (
        out.style
        .format({
            'AUC': '{:.3f}',
            'Δmedian': '{:.3f}',
            'Cliffδ': '{:.3f}',
            'median': '{:.3f}',
        }, na_rep='-')
        .hide(axis='index')
        .set_table_styles([
            {'selector': 'thead th', 'props': [('background-color', '#e8e8e8'), ('font-weight', 'bold'), ('text-align', 'center')]},
            {'selector': 'td', 'props': [('text-align', 'center')]},
            {'selector': 'td:first-child', 'props': [('text-align', 'left'), ('font-weight', '500')]},
        ])
    )

    display(styled)
    return out


def plot_feature_simple(
    df,
    feature,
    target,
    log_y=False,
    clip=(0.01, 0.99),
    sample_n=80000,
    bins=60,
    seed=49,
    colors=None,
):
    colors = colors or DEFAULT_COLORS
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    x_full = pd.to_numeric(df[feature], errors='coerce')
    y_full = df[target].astype(int)

    lo, hi = x_full.quantile(clip[0]), x_full.quantile(clip[1])
    mask = (x_full >= lo) & (x_full <= hi) & x_full.notna()
    x_clip = x_full[mask]
    y_clip = y_full[mask]

    if sample_n and len(x_clip) > sample_n:
        idx = x_clip.sample(sample_n, random_state=seed).index
        x_clip = x_clip.loc[idx]
        y_clip = y_clip.loc[idx]

    ax = axes[0]
    ax.hist(x_clip, bins=bins, density=True, alpha=0.7, color='gray', edgecolor='white')
    try:
        sns.kdeplot(x=x_clip, ax=ax, linewidth=2, color='steelblue')
    except Exception:
        pass
    ax.set_xlabel(feature)
    ax.set_ylabel('Density')
    ax.set_title('Overall Distribution', fontweight='bold')
    if log_y:
        ax.set_yscale('log')

    ax = axes[1]
    plot_df = pd.DataFrame({'x': x_clip, 'Class': y_clip.map({0: 'Benign', 1: 'Malignant'})})

    sns.boxenplot(
        data=plot_df,
        x='Class',
        y='x',
        palette={'Benign': colors['healthy'], 'Malignant': colors['malignant']},
        ax=ax,
        linewidth=1,
    )

    sample_pts = plot_df.sample(min(1000, len(plot_df)), random_state=seed)
    sns.stripplot(data=sample_pts, x='Class', y='x', color='black', alpha=0.15, size=2, ax=ax)

    ax.set_xlabel('')
    ax.set_ylabel(feature)
    ax.set_title('Benign vs Malignant', fontweight='bold')

    fig.suptitle(f'Feature: {feature}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

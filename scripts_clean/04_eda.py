import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rcParams


# global plot style - keeping it clean, no top/right spines
rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 10
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['axes.titlesize'] = 11
rcParams['axes.titleweight'] = 'bold'
rcParams['figure.dpi'] = 300

PRIMARY   = '#2C5F8A'
SECONDARY = '#E8824A'
GREY      = '#AAAAAA'
ANNOTATE  = '#C0392B'

DPI = 300


DATA_PATH   = os.path.join('data', 'processed', 'clo_dataset_clean.csv')
FIGURES_DIR = os.path.join('outputs', 'figures')
TABLES_DIR  = os.path.join('outputs', 'tables')

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR,  exist_ok=True)


df = pd.read_csv(DATA_PATH)
df['vintage'] = df['vintage'].astype(int)

print(f"Dataset loaded: {len(df)} deals, {df.shape[1]} variables")
print(f"Vintage range: {df['vintage'].min()} -- {df['vintage'].max()}")


# ccc excluded (constant); deal_size_log is a regressor not a summary var
NUMERIC_VARS = [
    'total_deal_size_mn',
    'reinvestment_period',
    'non_call_period',
    'oc_ratio_class_a',
    'class_a_pct',
    'sub_notes_pct',
]

VAR_LABELS = {
    'total_deal_size_mn' : 'Total Deal Size (mn)',
    'reinvestment_period': 'Reinvestment Period (yrs)',
    'non_call_period'    : 'Non-Call Period (yrs)',
    'oc_ratio_class_a'   : 'OC Ratio - Class A (%)',
    'class_a_pct'        : 'Class A Tranche (% of deal)',
    'sub_notes_pct'      : 'Subordinated Notes (% of deal)',
}

# descriptive stats table: percentiles at 25/50/75 for the TFG results section
stats = df[NUMERIC_VARS].describe(percentiles=[0.25, 0.50, 0.75]).T
stats = stats[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
stats.columns = ['N', 'Mean', 'Std Dev', 'Min', 'P25', 'Median', 'P75', 'Max']
stats.index = [VAR_LABELS[v] for v in NUMERIC_VARS]

stats['N'] = stats['N'].astype(int)
stats_display = stats.copy()
for col in stats_display.columns[1:]:
    stats_display[col] = stats_display[col].round(2)

csv_path = os.path.join(TABLES_DIR, 'descriptive_stats.csv')
stats_display.to_csv(csv_path)
print(f"Saved: {csv_path}")

fig, ax = plt.subplots(figsize=(12, 3.2))
ax.axis('off')

table = ax.table(
    cellText  = stats_display.values,
    rowLabels = stats_display.index,
    colLabels = stats_display.columns,
    cellLoc   = 'center',
    rowLoc    = 'left',
    loc       = 'center',
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.6)

# header row styling
for j in range(len(stats_display.columns)):
    cell = table[0, j]
    cell.set_facecolor(PRIMARY)
    cell.set_text_props(color='white', fontweight='bold')

# highlight the Max column
for i in range(1, len(stats_display) + 1):
    cell = table[i, -1]
    cell.set_facecolor('#F0F4F8')
    cell.set_text_props(fontweight='bold')

# alternating row background for readability
for i in range(1, len(stats_display) + 1):
    bg = '#FFFFFF' if i % 2 == 1 else '#F7F9FB'
    for j in range(len(stats_display.columns)):
        table[i, j].set_facecolor(bg)

fig.suptitle('Table 1 - Descriptive Statistics of Main Structural Variables',
             fontsize=11, fontweight='bold', y=0.98)

plt.tight_layout()
png_path = os.path.join(TABLES_DIR, 'descriptive_stats.png')
fig.savefig(png_path, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved: {png_path}")


# histograms for all main variables - Figure 1 in the thesis
HIST_XLABELS = {
    'total_deal_size_mn' : 'Total Deal Size (mn)',
    'reinvestment_period': 'Reinvestment Period (years)',
    'non_call_period'    : 'Non-Call Period (years)',
    'oc_ratio_class_a'   : 'OC Ratio - Class A (%)',
    'class_a_pct'        : 'Class A Tranche (% of deal)',
    'sub_notes_pct'      : 'Subordinated Notes (% of deal)',
}

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
axes = axes.flatten()

for i, var in enumerate(NUMERIC_VARS):
    ax = axes[i]
    series = df[var].dropna()

    # bin count: sqrt rule with floor/ceiling to avoid over/under-binning on n=36
    n_bins = min(15, max(8, int(np.sqrt(len(series)))))
    ax.hist(series, bins=n_bins, color=PRIMARY, edgecolor='white', linewidth=0.6, alpha=0.88)

    median_val = series.median()
    ax.axvline(median_val, color=SECONDARY, linewidth=1.5, linestyle='--',
               label=f'Median: {median_val:.2f}')

    ax.set_title(VAR_LABELS[var])
    ax.set_xlabel(HIST_XLABELS[var], fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis='y', color=GREY, linewidth=0.5, alpha=0.6)
    ax.tick_params(labelsize=8)

fig.suptitle('Figure 1 - Distribution of Key Structural Variables (n = 36)',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()

hist_path = os.path.join(FIGURES_DIR, 'histograms_main_vars.png')
fig.savefig(hist_path, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved: {hist_path}")


def annotate_outliers(ax, df_sub, x_col, y_col, threshold_fn, label_col='deal_name'):
    # annotate outlier points; threshold_fn takes a Series and returns a bool mask
    mask = threshold_fn(df_sub[y_col])
    for _, row in df_sub[mask].iterrows():
        ax.annotate(
            row[label_col],
            xy         = (row[x_col], row[y_col]),
            xytext     = (6, 6),
            textcoords = 'offset points',
            fontsize   = 7.5,
            color      = ANNOTATE,
            fontweight = 'bold',
            arrowprops = dict(arrowstyle='->', color=ANNOTATE, lw=0.8),
        )


# boxplot config for 4 variables - outlier thresholds chosen per variable
BOXPLOT_CONFIG = [
    {
        'var'       : 'oc_ratio_class_a',
        'ylabel'    : 'OC Ratio - Class A (%)',
        'title'     : 'Figure 2 - OC Ratio (Class A) by Vintage',
        'filename'  : 'boxplot_oc_ratio.png',
        'outlier_fn': lambda s: np.abs(s - s.mean()) > 2 * s.std(),
    },
    {
        'var'       : 'reinvestment_period',
        'ylabel'    : 'Reinvestment Period (years)',
        'title'     : 'Figure 3 - Reinvestment Period by Vintage',
        'filename'  : 'boxplot_reinvestment.png',
        # Known structural outlier: CQS deal with ~1.8 years (probable reset)
        'outlier_fn': lambda s: s < 2.5,
    },
    {
        'var'       : 'sub_notes_pct',
        'ylabel'    : 'Subordinated Notes (% of deal)',
        'title'     : 'Figure 4 - Subordinated Notes Size by Vintage',
        'filename'  : 'boxplot_sub_notes_pct.png',
        'outlier_fn': lambda s: np.abs(s - s.mean()) > 2 * s.std(),
    },
    {
        'var'       : 'total_deal_size_mn',
        'ylabel'    : 'Total Deal Size (mn)',
        'title'     : 'Figure 5 - Total Deal Size by Vintage',
        'filename'  : 'boxplot_deal_size.png',
        'outlier_fn': lambda s: np.abs(s - s.mean()) > 2 * s.std(),
    },
]

vintages_sorted = sorted(df['vintage'].unique())

for cfg in BOXPLOT_CONFIG:
    var = cfg['var']

    groups = [df.loc[df['vintage'] == v, var].dropna().values for v in vintages_sorted]

    fig, ax = plt.subplots(figsize=(10, 5))

    bp = ax.boxplot(
        groups,
        patch_artist = True,
        notch        = False,
        widths       = 0.5,
        medianprops  = dict(color=SECONDARY, linewidth=2),
        boxprops     = dict(facecolor='#D6E4F0', color=PRIMARY, linewidth=1.2),
        whiskerprops = dict(color=PRIMARY, linewidth=1.2),
        capprops     = dict(color=PRIMARY, linewidth=1.2),
        flierprops   = dict(marker='o', color=GREY, markersize=4),
    )

    # jitter individual points on top of the boxplot for transparency (n is small)
    for j, (v, grp) in enumerate(zip(vintages_sorted, groups), start=1):
        x_jitter = np.random.default_rng(seed=42).uniform(-0.15, 0.15, size=len(grp))
        ax.scatter(np.full(len(grp), j) + x_jitter, grp,
                   color=PRIMARY, alpha=0.55, s=20, zorder=3)

    df_annot = df[['vintage', var, 'deal_name']].dropna(subset=[var]).copy()
    df_annot['x_pos'] = df_annot['vintage'].map(
        {v: j for j, v in enumerate(vintages_sorted, start=1)}
    )
    annotate_outliers(ax, df_annot, x_col='x_pos', y_col=var,
                      threshold_fn=cfg['outlier_fn'])

    ax.set_xticks(range(1, len(vintages_sorted) + 1))
    ax.set_xticklabels([str(v) for v in vintages_sorted], fontsize=9)
    ax.set_xlabel('Vintage (issuance year)', fontsize=10)
    ax.set_ylabel(cfg['ylabel'], fontsize=10)
    ax.set_title(cfg['title'], fontsize=11, fontweight='bold', pad=10)
    ax.grid(axis='y', color=GREY, linewidth=0.5, alpha=0.6)
    ax.tick_params(axis='y', labelsize=9)

    # sample size per vintage below x-axis
    for j, v in enumerate(vintages_sorted, start=1):
        n = len(df[df['vintage'] == v])
        ax.text(j, ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.07,
                f'n={n}', ha='center', va='top', fontsize=7.5, color='#666666')

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, cfg['filename'])
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")


# Figure 6 - Class A tranche scatter with outlier annotation
fig, ax = plt.subplots(figsize=(9, 5))

ax.scatter(df['vintage'], df['class_a_pct'],
           color=PRIMARY, alpha=0.7, s=40, zorder=3, label='All deals')

mean_val = df['class_a_pct'].mean()
std_val  = df['class_a_pct'].std()
ax.axhline(mean_val,             color=SECONDARY, linewidth=1.5, linestyle='--', label=f'Mean ({mean_val:.1f}%)')
ax.axhline(mean_val + 2*std_val, color=GREY,      linewidth=1.0, linestyle=':',  label=f'Mean +/- 2 SD')
ax.axhline(mean_val - 2*std_val, color=GREY,      linewidth=1.0, linestyle=':')

# label deals that exceed mean + 2SD threshold
outlier_mask = df['class_a_pct'] > mean_val + 2 * std_val
for _, row in df[outlier_mask].iterrows():
    ax.annotate(
        row['deal_name'],
        xy         = (row['vintage'], row['class_a_pct']),
        xytext     = (10, 6),
        textcoords = 'offset points',
        fontsize   = 8,
        color      = ANNOTATE,
        fontweight = 'bold',
        arrowprops = dict(arrowstyle='->', color=ANNOTATE, lw=0.9),
    )

ax.set_xlabel('Vintage', fontsize=10)
ax.set_ylabel('Class A Tranche (% of deal)', fontsize=10)
ax.set_title('Figure 6 - Class A Tranche Size by Vintage\n(annotated outlier: > mean + 2 SD)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9, frameon=False)
ax.grid(color=GREY, linewidth=0.5, alpha=0.5)
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

plt.tight_layout()
scatter_path = os.path.join(FIGURES_DIR, 'scatter_class_a_outlier.png')
fig.savefig(scatter_path, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved: {scatter_path}")


# console output used when writing sections 4.1 and 4.2
print("\n" + "="*60)
print("KEY OBSERVATIONS -- for TFG sections 3.1 / 3.2")
print("="*60)

print("\n--- Descriptive stats (main variables) ---")
print(stats_display.to_string())

print("\n--- Class A outliers (> mean + 2 SD) ---")
outliers_ca = df[df['class_a_pct'] > mean_val + 2 * std_val][['deal_name', 'vintage', 'class_a_pct', 'manager']]
print(outliers_ca.to_string(index=False))

print("\n--- Reinvestment period outliers (< 2.5 years) ---")
outliers_rp = df[df['reinvestment_period'] < 2.5][['deal_name', 'vintage', 'reinvestment_period', 'manager']]
print(outliers_rp.to_string(index=False))

print("\n--- Deals per vintage ---")
print(df.groupby('vintage').size().rename('n_deals').to_string())

print("\n--- OC ratio trend (median by vintage) ---")
print(df.groupby('vintage')['oc_ratio_class_a'].median().round(2).to_string())

print("\n--- Sub notes pct trend (median by vintage) ---")
print(df.groupby('vintage')['sub_notes_pct'].median().round(2).to_string())

print("\nAll outputs saved.")

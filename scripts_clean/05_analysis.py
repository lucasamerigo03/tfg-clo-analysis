import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "clo_dataset_clean.csv"
FIG_DIR   = BASE_DIR / "outputs" / "figures"
TAB_DIR   = BASE_DIR / "outputs" / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# two dependent variables, same set of regressors for both models
DEP_VARS   = ["oc_ratio_class_a", "class_a_pct"]
REGRESSORS = ["reinvestment_period", "sub_notes_pct", "vintage"]
CORR_VARS  = DEP_VARS + REGRESSORS


# Pearson correlation matrix across dependent vars and regressors
corr_matrix = df[CORR_VARS].corr()

# display labels for the heatmap - shorter names fit better in the grid
DISPLAY_LABELS = {
    "oc_ratio_class_a":    "OC Ratio\n(Class A)",
    "class_a_pct":         "Class A\n(%)",
    "reinvestment_period": "Reinvestment\nPeriod",
    "non_call_period":     "Non-Call\nPeriod",
    "sub_notes_pct":       "Sub Notes\n(%)",
    "deal_size_log":       "Deal Size\n(log)",
    "vintage":             "Vintage",
}
labels = [DISPLAY_LABELS[v] for v in CORR_VARS]

fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(
    corr_matrix,
    ax=ax,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    vmin=-1, vmax=1,
    linewidths=0.5,
    linecolor="white",
    annot_kws={"size": 9},
    xticklabels=labels,
    yticklabels=labels,
    cbar_kws={"shrink": 0.8, "label": "Pearson r"},
)
ax.set_title("Correlation Matrix - Key Structural Variables", fontsize=12, pad=14)
ax.tick_params(axis="both", labelsize=8.5)
plt.tight_layout()
fig.savefig(FIG_DIR / "correlation_heatmap.png", dpi=300)
plt.close(fig)
print("Saved: correlation_heatmap.png")


def run_ols(dep_var, regressors, data):
    # fit OLS with constant; returns model, summary df, vif df
    y = data[dep_var]
    X = data[regressors]
    X_const = sm.add_constant(X)

    model = sm.OLS(y, X_const).fit()

    # build a clean summary table from the statsmodels output
    summary_df = pd.DataFrame({
        "Variable":    ["Constant"] + regressors,
        "Coefficient": model.params.values,
        "Std Error":   model.bse.values,
        "t-stat":      model.tvalues.values,
        "p-value":     model.pvalues.values,
    })
    summary_df["Sig."] = summary_df["p-value"].apply(
        lambda p: "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
    )

    # VIF to check for multicollinearity between regressors
    vif_df = pd.DataFrame({
        "Variable": regressors,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
    })

    return model, summary_df, vif_df


def format_regression_stats(model, n):
    # model-level fit stats as a one-row df
    return pd.DataFrame([{
        "N":       n,
        "R^2":     round(model.rsquared, 4),
        "Adj. R^2": round(model.rsquared_adj, 4),
        "F-stat":  round(model.fvalue, 3),
        "Prob(F)": round(model.f_pvalue, 4),
    }])


def save_table_png(summary_df, vif_df, fit_stats, title, filepath):
    # save regression results as PNG (coefs + VIF + fit stats)
    coef_display = summary_df.copy()
    for col in ["Coefficient", "Std Error", "t-stat"]:
        coef_display[col] = coef_display[col].round(4)
    coef_display["p-value"] = coef_display["p-value"].round(4)

    vif_display = vif_df.copy()
    vif_display["VIF"] = vif_display["VIF"].round(2)

    fig = plt.figure(figsize=(10, 5.5))
    fig.suptitle(title, fontsize=11, fontweight="bold", y=0.98)

    # coefficient table takes the upper portion of the figure
    ax_coef = fig.add_axes([0.02, 0.42, 0.96, 0.50])
    ax_coef.axis("off")
    tbl = ax_coef.table(
        cellText=coef_display.values.tolist(),
        colLabels=list(coef_display.columns),
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.35)
    for j in range(len(coef_display.columns)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    ax_vif = fig.add_axes([0.02, 0.16, 0.42, 0.22])
    ax_vif.axis("off")
    vif_tbl = ax_vif.table(
        cellText=vif_display.values.tolist(),
        colLabels=list(vif_display.columns),
        loc="center",
        cellLoc="center",
    )
    vif_tbl.auto_set_font_size(False)
    vif_tbl.set_fontsize(8.5)
    vif_tbl.scale(1, 1.3)
    for j in range(len(vif_display.columns)):
        vif_tbl[0, j].set_facecolor("#2c3e50")
        vif_tbl[0, j].set_text_props(color="white", fontweight="bold")
    ax_vif.set_title("VIF Diagnostics", fontsize=8.5, loc="left", pad=4)

    ax_fit = fig.add_axes([0.52, 0.16, 0.46, 0.22])
    ax_fit.axis("off")
    fit_tbl = ax_fit.table(
        cellText=fit_stats.values.tolist(),
        colLabels=list(fit_stats.columns),
        loc="center",
        cellLoc="center",
    )
    fit_tbl.auto_set_font_size(False)
    fit_tbl.set_fontsize(8.5)
    fit_tbl.scale(1, 1.3)
    for j in range(len(fit_stats.columns)):
        fit_tbl[0, j].set_facecolor("#2c3e50")
        fit_tbl[0, j].set_text_props(color="white", fontweight="bold")
    ax_fit.set_title("Model Fit", fontsize=8.5, loc="left", pad=4)

    fig.text(0.02, 0.04,
             "Significance codes:  *** p<0.01   ** p<0.05   * p<0.10",
             fontsize=7.5, color="grey")

    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath.name}")


# Model 1: OC ratio as dependent variable
model1, summary1, vif1 = run_ols("oc_ratio_class_a", REGRESSORS, df)
fit1 = format_regression_stats(model1, len(df))

save_table_png(
    summary1, vif1, fit1,
    title="OLS Regression - Dependent Variable: OC Ratio (Class A)",
    filepath=FIG_DIR / "regression_oc_ratio.png",
)
summary1.to_csv(TAB_DIR / "regression_oc_ratio.csv", index=False)
print("Saved: regression_oc_ratio.csv")


# Model 2: Class A tranche size as dependent variable
model2, summary2, vif2 = run_ols("class_a_pct", REGRESSORS, df)
fit2 = format_regression_stats(model2, len(df))

save_table_png(
    summary2, vif2, fit2,
    title="OLS Regression - Dependent Variable: Class A (%)",
    filepath=FIG_DIR / "regression_class_a_pct.png",
)
summary2.to_csv(TAB_DIR / "regression_class_a_pct.csv", index=False)
print("Saved: regression_class_a_pct.csv")


# full statsmodels output to console for reference when writing section 4.3
print("\n" + "="*60)
print("REGRESSION 1: oc_ratio_class_a")
print("="*60)
print(model1.summary())

print("\n" + "="*60)
print("REGRESSION 2: class_a_pct")
print("="*60)
print(model2.summary())

print("\nVIF -- Regression 1:")
print(vif1.to_string(index=False))
print("\nVIF -- Regression 2:")
print(vif2.to_string(index=False))

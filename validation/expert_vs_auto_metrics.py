import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, pearsonr, mannwhitneyu

# Load data
expert_df = pd.read_csv("expert_concentration.csv")
auto_df = pd.read_csv("auto_concentration.csv")

# Prepare data - use first 9 automatic measurements to match expert counts
expert_samples = [expert_df.iloc[i, 1:-1].values for i in range(len(expert_df))]
auto_samples = [
    auto_df.iloc[i, 1:10].values for i in range(len(auto_df))
]  # Using only im1-im9

# Initialize results storage
results = []

# 1. Per-sample analysis
for sample, (expert, auto) in enumerate(zip(expert_samples, auto_samples), start=1):
    # Statistical tests (using non-paired tests since we're comparing distributions)
    ks_stat, ks_p = ks_2samp(expert, auto)
    mw_stat, mw_p = mannwhitneyu(expert, auto)
    r, p_r = pearsonr(expert, auto)

    # Error metrics
    mae = np.mean(np.abs(expert - auto))
    rmse = np.sqrt(np.mean((expert - auto) ** 2))

    results.append(
        {
            "Sample": expert_df["Sample"].iloc[sample - 1],
            "KS_p": ks_p,
            "MW_p": mw_p,
            "Pearson_r": r,
            "MAE": mae,
            "RMSE": rmse,
            "Expert_Mean": np.mean(expert),
            "Auto_Mean": np.mean(auto),
            "Expert_SD": np.std(expert),
            "Auto_SD": np.std(auto),
        }
    )

results_df = pd.DataFrame(results)

# 2. Global analysis
all_expert = np.concatenate(expert_samples)
all_auto = np.concatenate(auto_samples)
global_r, global_p = pearsonr(all_expert, all_auto)
global_ks, global_ks_p = ks_2samp(all_expert, all_auto)


print(
    """
=== STATISTICAL REPORT ===

1. GLOBAL AGREEMENT:
- Pearson correlation: r = {:.3f} (p = {:.2e})
- Distribution similarity (KS test): p = {:.3f} {}

2. SAMPLE-LEVEL ANALYSIS (N={}):
- Mean ± SD Pearson r: {:.3f} ± {:.3f}
- Samples with similar distributions (KS p > 0.05): {}/{}
- Mean Absolute Error: {:.2e} ± {:.2e} cells/ml
- Relative Error: {:.1f}% ± {:.1f}%

3. CONCLUSION:
The automated method shows {} agreement with expert counts ({}).
""".format(
        global_r,
        global_p,
        global_ks_p,
        "similar distributions" if global_ks_p > 0.05 else "different distributions",
        len(results_df),
        results_df["Pearson_r"].mean(),
        results_df["Pearson_r"].std(),
        sum(results_df["KS_p"] > 0.05),
        len(results_df),
        results_df["MAE"].mean(),
        results_df["MAE"].std(),
        (results_df["MAE"] / results_df["Expert_Mean"]).mean() * 100,
        (results_df["MAE"] / results_df["Expert_Mean"]).std() * 100,
        "strong" if global_r > 0.9 else "moderate",
        (
            "statistically indistinguishable distributions"
            if global_ks_p > 0.05
            else "statistically significant distribution differences"
        ),
    )
)
results_df.to_csv("per_sample_results.csv", index=False)

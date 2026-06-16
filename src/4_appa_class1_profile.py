import pandas as pd
import numpy as np

df_combined = pd.read_csv("./data/positive_cases_profiles.csv")

cluster_col = 'Pos_Profile' if 'Pos_Profile' in df_combined.columns else 'Profile'
compare_cols = ['n_selfev', 'n_selfre', 'n_selfas', 'n_teach', 'n_friend', 'n_study', 'n_sclike', 'n_home', 'n_life', 'n_emost', 'n_comm', 'n_resi', 'n_bully', 'n_dev', 'absent', 'violen', 'hsrela', 'ses']

n_bootstraps = 1000
np.random.seed(42)

clusters = sorted(df_combined[cluster_col].dropna().unique())
all_results = []
summary_dict = {}

for c in clusters:
    df_target = df_combined[df_combined[cluster_col] == c]
    df_other = df_combined[df_combined[cluster_col] != c]
    
    sig_high = []
    sig_low = []
    
    for col in compare_cols:
        target_vals = df_target[col].dropna().values
        other_vals = df_other[col].dropna().values
        
        if len(target_vals) == 0 or len(other_vals) == 0:
            continue
            
        obs_diff = np.mean(target_vals) - np.mean(other_vals)
        
        boot_diffs = []
        for _ in range(n_bootstraps):
            boot_target = np.random.choice(target_vals, size=len(target_vals), replace=True)
            boot_other = np.random.choice(other_vals, size=len(other_vals), replace=True)
            boot_diffs.append(np.mean(boot_target) - np.mean(boot_other))
            
        lower_ci = np.percentile(boot_diffs, 2.5)
        upper_ci = np.percentile(boot_diffs, 97.5)
        
        is_significant = (lower_ci > 0) or (upper_ci < 0)
        
        if is_significant and obs_diff > 0:
            interpretation = "High"
            sig_high.append(col)
        elif is_significant and obs_diff < 0:
            interpretation = "Low"
            sig_low.append(col)
        else:
            interpretation = "NS"
            
        all_results.append({
            'Cluster': c,
            'Variable': col,
            'Target Mean': round(np.mean(target_vals), 2),
            'Other Mean': round(np.mean(other_vals), 2),
            'Diff': round(obs_diff, 2),
            '95% CI Lower': round(lower_ci, 2),
            '95% CI Upper': round(upper_ci, 2),
            'Interpretation': interpretation
        })
        
    summary_dict[c] = {'High': sig_high, 'Low': sig_low}

df_results = pd.DataFrame(all_results)
df_results.to_csv("./data/all_clusters_bootstrap_comparisons.csv", index=False, encoding="utf-8-sig")

for c in clusters:
    print(f"【クラスタ {int(c)} の特徴 (vs その他)】")
    print(f"有意に高い要因: {summary_dict[c]['High']}")
    print(f"有意に低い要因: {summary_dict[c]['Low']}\n")

print("すべての比較結果の詳細を 'all_clusters_bootstrap_comparisons.csv' として保存しました。")
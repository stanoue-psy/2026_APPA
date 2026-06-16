import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

df_combined = pd.read_csv("./data/rc_calculated_data.csv")

np.random.seed(42)

rc_cols = [col for col in df_combined.columns if str(col).startswith('RC_')]
df_rc = df_combined[rc_cols]

plt.figure(figsize=(10, 6))
plt.title("Dendrogram for Significant RC-based Profiles (Ward's Method)")
dendrogram = sch.dendrogram(sch.linkage(df_rc, method='ward'))
plt.ylabel("Euclidean Distances")
plt.xlabel("Individuals")
plt.show()

n_clusters = 8
hc = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
df_combined['Profile'] = hc.fit_predict(df_rc)

df_combined.to_csv("./data/final_clustered_profiles.csv", index=False, encoding="utf-8-sig")

profile_rc_means = df_combined.groupby('Profile')[rc_cols].mean()
print("=== プロファイルごとの平均RC値 ===")
print(profile_rc_means.T.round(3))

def calculate_odds_ratio(df, target_col, cluster_col, threshold):
    results = []
    df_temp = df.copy()
    df_temp['Positive'] = (df_temp[target_col] == threshold).astype(int)
    clusters = sorted(df_temp[cluster_col].unique())
    
    for c in clusters:
        a = ((df_temp[cluster_col] == c) & (df_temp['Positive'] == 1)).sum()
        b = ((df_temp[cluster_col] == c) & (df_temp['Positive'] == 0)).sum()
        c_other = ((df_temp[cluster_col] != c) & (df_temp['Positive'] == 1)).sum()
        d = ((df_temp[cluster_col] != c) & (df_temp['Positive'] == 0)).sum()
        
        if a == 0 or b == 0 or c_other == 0 or d == 0:
            a, b, c_other, d = a + 0.5, b + 0.5, c_other + 0.5, d + 0.5
            
        or_val = (a * d) / (b * c_other)
        log_or = np.log(or_val)
        se = np.sqrt(1/a + 1/b + 1/c_other + 1/d)
        
        lower_ci = np.exp(log_or - 1.96 * se)
        upper_ci = np.exp(log_or + 1.96 * se)
        
        results.append({
            'Threshold': f'== {threshold}',
            'Profile': c,
            'Odds Ratio': round(or_val, 2),
            '95% CI Lower': round(lower_ci, 2),
            '95% CI Upper': round(upper_ci, 2),
            'Positives (N)': int(a) if a % 1 == 0 else a,
            'Total in Profile (N)': int(a + b) if (a + b) % 1 == 0 else (a + b)
        })
        
    return pd.DataFrame(results)

df_or_2 = calculate_odds_ratio(df_combined, 'absent', 'Profile', 1)


print("\n=== オッズ比および95%信頼区間 ===")
print(df_or_2.to_string(index=False))
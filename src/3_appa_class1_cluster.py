import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

df_combined = pd.read_csv("./data/rc_calculated_data.csv")

threshold = 1
df_positive = df_combined[df_combined['absent'] >= threshold].copy()

rc_cols = [col for col in df_positive.columns if str(col).startswith('RC_')]
df_rc_pos = df_positive[rc_cols]

plt.figure(figsize=(10, 6))
plt.title(f"Dendrogram for Positive Cases Only (absent >= {threshold})")
dendrogram = sch.dendrogram(sch.linkage(df_rc_pos, method='ward'))
plt.ylabel("Euclidean Distances")
plt.xlabel("Positive Individuals")
plt.show()

n_pos_clusters = 7
hc_pos = AgglomerativeClustering(n_clusters=n_pos_clusters, metric='euclidean', linkage='ward')
df_positive['Pos_Profile'] = hc_pos.fit_predict(df_rc_pos)

profile_rc_means_pos = df_positive.groupby('Pos_Profile')[rc_cols].mean()

print(f"=== 陽性者（absent >= {threshold}）内のサブプロファイル別 平均RC値 ===")
print(profile_rc_means_pos.T.round(3))

df_positive.to_csv("./data/positive_cases_profiles.csv", index=False, encoding="utf-8-sig")
print("\n陽性者のみの分類結果を 'positive_cases_profiles.csv' として保存しました。")
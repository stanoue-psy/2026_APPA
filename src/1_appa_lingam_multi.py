import pandas as pd
import numpy as np
import lingam
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import scipy.stats as stats
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import concurrent.futures
import multiprocessing
import functools

np.random.seed(42)

def _bootstrap_task(seed, df, pk):
    df_resampled = resample(df, random_state=seed)
    model_boot = lingam.DirectLiNGAM(prior_knowledge=pk)
    model_boot.fit(df_resampled)
    return model_boot.adjacency_matrix_

if __name__ == '__main__':
    np.random.seed(42)

    df = pd.read_csv("./data/data.csv")
    cols = ["self1", "self2", "life", "family", "teach", "friend", "motiv", "sclike", "phyill", "bull", "emo", "devel","kishi"]
    df_clean = df[cols].dropna().astype(float)

    scaler = StandardScaler()
    df_std = pd.DataFrame(scaler.fit_transform(df_clean), columns=df_clean.columns)

    target_index = df_std.columns.get_loc("kishi")
    pk_matrix = np.full((df_std.shape[1], df_std.shape[1]), -1)
    pk_matrix[:, target_index] = 0
    np.fill_diagonal(pk_matrix, 0)

    model = lingam.DirectLiNGAM(prior_knowledge=pk_matrix, random_state=42)
    model.fit(df_std)
    adj_matrix = model.adjacency_matrix_

    n_sampling = 1000
    max_workers = multiprocessing.cpu_count()
    
    print(f"ブートストラップ法を実行中（{n_sampling}回）...")
    print(f"CPUの全コア（{max_workers}コア）を使用して並列処理を行います。しばらくお待ちください。")

    task_func = functools.partial(_bootstrap_task, df=df_std, pk=pk_matrix)
    seeds = range(42, 42 + n_sampling)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(task_func, seeds), total=n_sampling, desc="Parallel Bootstrap"))

    boot_matrices = np.array(results)

    significant_edges = []
    features = df_std.columns

    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            coef = adj_matrix[i, j]
            if coef != 0:
                se = boot_matrices[:, i, j].std()
                p_val = 0.0 if se == 0 else 2 * (1 - stats.norm.cdf(abs(coef / se)))
                
                if p_val < 0.05:
                    significant_edges.append((features[j], features[i], coef))

    parents_dict = {col: [] for col in features}
    for cause, effect, coef in significant_edges:
        parents_dict[effect].append((cause, coef))

    rc_matrix = []
    for idx, row in df_std.iterrows():
        rc_vector = {}
        for effect, parents in parents_dict.items():
            if not parents: continue
            
            ecv_dict = {}
            for cause, coef in parents:
                ecv_dict[f"RC_{cause}->{effect}"] = coef * row[cause]
                
            max_abs_ecv = max([abs(v) for v in ecv_dict.values()]) if ecv_dict else 0
            
            for edge, ecv in ecv_dict.items():
                if max_abs_ecv == 0:
                    rc_vector[edge] = 0.0
                else:
                    rc_vector[edge] = abs(ecv) / max_abs_ecv
                    
        rc_matrix.append(rc_vector)

    df_rc = pd.DataFrame(rc_matrix)
    df_combined = pd.concat([df_clean.reset_index(drop=True), df_rc], axis=1)

    df_combined.to_csv("./data/rc_calculated_data.csv", index=False, encoding="utf-8-sig")

    print("=== 抽出された有意なパス ===")
    for edge in significant_edges:
        print(f"{edge[0]} -> {edge[1]} (β: {edge[2]:.3f})")
    print("\n計算処理が完了しました。結果を 'rc_calculated_data.csv' として保存しました。")

    paths = []
    G = nx.DiGraph()

    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            coef = adj_matrix[i, j]
            if coef != 0:
                boot_coefs = boot_matrices[:, i, j]
                se = boot_coefs.std()
                
                if se == 0:
                    p_val = 0.0
                else:
                    z = coef / se
                    p_val = 2 * (1 - stats.norm.cdf(abs(z)))

                paths.append({
                    'Cause': features[j],
                    'Effect': features[i],
                    'Std_Beta': round(coef, 3),
                    'SE': round(se, 3),
                    'p-value': round(p_val, 4)
                })
                
                G.add_edge(features[j], features[i], weight=coef, p_value=p_val)

    df_paths = pd.DataFrame(paths)
    df_paths['abs_beta'] = df_paths['Std_Beta'].abs()
    df_paths = df_paths.sort_values('abs_beta', ascending=False).drop(columns=['abs_beta'])

    print("=== 標準化パス係数および有意確率（影響力が強い順） ===")
    print(df_paths.to_string(index=False))

    pos = nx.spring_layout(G, k=0.8, seed=42)
    plt.figure(figsize=(12, 8))

    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color="skyblue", edgecolors="white")
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

    for u, v, data in G.edges(data=True):
        weight = data['weight']
        p_val = data['p_value']
        
        edge_color = "red" if weight < 0 else "blue"
        edge_style = "solid" if p_val < 0.05 else "dashed"
        
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            width=abs(weight) * 10,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=20,
            node_size=3000,
            edge_color=edge_color,
            style=edge_style,
            connectionstyle="arc3,rad=0.05",
            alpha=0.8
        )

    plt.title("DirectLiNGAM Network (Standardized Beta & Significance)")
    plt.axis("off")
    plt.show()
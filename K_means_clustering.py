import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 生成示例數據
np.random.seed(42)
data = {
    'x': np.random.rand(100) * 100,
    'y': np.random.rand(100) * 100
}

# 創建 DataFrame
df = pd.DataFrame(data)

# 數據標準化
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# 訓練 K-means 模型
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(df_scaled)

# 獲取聚類結果
df['cluster'] = kmeans.labels_

# 繪製聚類結果
plt.figure(figsize=(10, 6))
sns.scatterplot(x='x', y='y', hue='cluster', data=df, palette='viridis')
plt.title('K-means Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(title='Cluster')
plt.savefig('kmeans_clustering.png')
plt.show()

# 輸出聚類中心
print("Cluster Centers:")
print(scaler.inverse_transform(kmeans.cluster_centers_))
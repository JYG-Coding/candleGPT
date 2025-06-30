import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time

#Step 1: 加载你的3K形态数据
# 假设 X 的 shape 为 (N_samples, 12)
X_tmp = np.load("F:\PythonProject\candleGPT\DataTools/norm_windows.npy")  
X = X_tmp.reshape(X_tmp.shape[0], -1)
print("数据shape:", X.shape)
print("数据类型:", X.dtype)
print("前两行数据:\n", X[:2])

"""
# 展平成一维向量，每个窗口 shape: (3,4) -> (12,)
norm_windows_flat = [w.flatten() for w in norm_windows]
X = np.array(norm_windows_flat)
"""

#Step 2: 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Step 3: 聚类 
n_clusters = 3  # 你可以试试 50 / 100 / 200
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_scaled)

#Step 4: 获取每个窗口的token标签（即属于哪个聚类中心）
labels = kmeans.labels_  # 每个样本的token ID
cluster_centers = kmeans.cluster_centers_

#Step 5: 保存标准化器、聚类模型和token序列以及聚类中心，便于后续复用
joblib.dump(scaler, 'F:\PythonProject\candleGPT\KmeansResult\scaler.pkl')
joblib.dump(kmeans, 'F:\PythonProject\candleGPT\KmeansResult\kmeans_tokenizer.pkl')
np.save('F:\PythonProject\candleGPT\KmeansResult/token_sequence.npy', labels)
np.save('F:\PythonProject\candleGPT\KmeansResult\cluster_centers.npy', cluster_centers)  

#打印结果
print("聚类中心:\n", cluster_centers)
print("每个样本的token ID:\n", labels)

#Step 6 (可选): 可视化
X_2d = TSNE(n_components=2, perplexity=10).fit_transform(X_scaled)
plt.figure(figsize=(10, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab20', s=5)
plt.colorbar(label='Token ID')
plt.title('3K形态聚类结果可视化')
plt.show()

time.sleep(1)  # 等待1秒，确保图形窗口正常显示


#Step 6 (可选): PCA可视化聚类结果
X_pca = PCA(n_components=2).fit_transform(X_scaled)
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab20', s=5)
plt.colorbar(label='Token ID')
plt.title('PCA聚类结果可视化')
plt.show()

time.sleep(1)  # 等待1秒，确保图形窗口正常显示
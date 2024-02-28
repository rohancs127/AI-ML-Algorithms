import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris['data'], columns= iris['feature_names'])
df['target'] = iris['target']

X = df.iloc[:,:-1]
y = df['target']

scaler = preprocessing.StandardScaler()
scaler.fit(X)
X_scaled_array = scaler.transform(X)
X_scaled = pd.DataFrame(X_scaled_array, columns= X.columns)

plt.figure(figsize=(14,7))
colormap = np.array(['red', 'green', 'blue'])

plt.subplot(1,3,1)
plt.scatter(X_scaled['petal length (cm)'], X_scaled['petal width (cm)'], c=colormap[y], s=40)
plt.title("Real")

model = KMeans(n_clusters=3)
y_pred = model.fit_predict(X)
y_pred = np.choose(y_pred, [1,0,2]).astype(np.int64)
plt.subplot(1,3,2)
plt.scatter(X_scaled['petal length (cm)'], X_scaled['petal width (cm)'], c=colormap[y_pred], s=40)
plt.title("KMeans")

gmm = GaussianMixture(n_components=3, max_iter=200)
y_cluster = gmm.fit_predict(X_scaled)
y_cluster = np.choose(y_cluster, [2,0,1]).astype(np.int64)
plt.subplot(1,3,3)
plt.scatter(X['petal length (cm)'], X['petal width (cm)'], c=colormap[y_cluster], s=40)
plt.title("GMM Classification")

plt.show()
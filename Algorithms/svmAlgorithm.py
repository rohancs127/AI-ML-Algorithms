# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
import seaborn as sns; sns.set()

# Working with Perfectly Linear Dataset
X_linear, y_linear = make_blobs(n_samples=50, centers=2,
random_state=0, cluster_std=0.60)

# Scatter plot of the perfectly linear dataset
plt.scatter(X_linear[:, 0], X_linear[:, 1], c=y_linear, s=50, cmap='winter')

# Train a Support Vector Classifier (SVC) with a linear kernel on the perfectly linear dataset
model_linear = SVC(kernel='linear', C=1)
model_linear.fit(X_linear, y_linear)

# Function to plot decision function for a 2D SVC
def plot_svc_decision_function(model, ax=None, plot_support=True):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)

    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
    levels=[-1, 0, 1], alpha=0.5,
    linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
        model.support_vectors_[:, 1],
        s=300, linewidth=1, facecolors='none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

# Plot decision function and support vectors for the perfectly linear dataset
plot_svc_decision_function(model_linear)

# Show the plot for the perfectly linear dataset
plt.show()

# Working with Almost Linearly Separable Dataset
X_almost_linear, y_almost_linear = make_blobs(n_samples=100, centers=2,
random_state=0, cluster_std=1.2)

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

for axi, C in zip(ax, [100.0, 0.1]):
    model_almost_linear = SVC(kernel='linear', C=C).fit(X_almost_linear, y_almost_linear)
    axi.scatter(X_almost_linear[:, 0], X_almost_linear[:, 1], c=y_almost_linear, s=50,
    cmap='winter')
    plot_svc_decision_function(model_almost_linear, axi)
    axi.scatter(model_almost_linear.support_vectors_[:, 0],
    model_almost_linear.support_vectors_[:, 1],
    s=300, lw=1, facecolors='none')
    axi.set_title('C = {0:.1f}'.format(C), size=14)

# Show the plot for the almost linearly separable dataset
plt.show()
#%%
# imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


#%%
#K-Means

df = pd.read_csv("USArrests.csv", index_col = 0)
df.head()
#df.info()
#df.describe()

#df.hist(figsize = (10,10))

kmeans = KMeans(n_clusters = 4)
kfit = kmeans.fit(df)
kfit.n_clusters
kfit.cluster_centers_
kfit.labels_

# Plot
kmeans = KMeans(n_clusters = 2)
kmeans.fit(df)

labels = kmeans.labels_
plt.scatter(df.iloc[:,0], df.iloc[:,1], c = labels, s = 50, cmap = "viridis");

centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c = "red", s=200, alpha = 0.5)



#%%
# Elbow
ssd = []

K = range(1,30)
for i in K:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(df)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Sums of Distance Against K Values")
plt.title("Elbow")
# YellowBrick
from yellowbrick.cluster import KElbowVisualizer

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k = (2,20))
elbow.fit(df)
elbow.poof()
#%%
# Final KMeans
kmeans = KMeans(n_clusters = 4)
kmeans.fit(df)

labels = kmeans.labels_
plt.scatter(df.iloc[:,0], df.iloc[:,1], c = labels, s = 50, cmap = "viridis");

centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c = "red", s=200, alpha = 0.5);

clusters = kmeans.labels_
pd.DataFrame({"States" : df.index, "Clusters":clusters})
df["Clusters No"] = clusters



#%%
# Hierarchical Cluster
from scipy.cluster.hierarchy import linkage
hc_complete = linkage(df,"complete")
hc_average  = linkage(df,"average")

from scipy.cluster.hierarchy import dendrogram

plt.figure(figsize= (14,5))
plt.title("Hierarchical Cluster Dendrogram")
plt.xlabel("X")
plt.ylabel("Y")
dendrogram(hc_complete,
           truncate_mode = "lastp",
           p = 10,
           show_contracted= True,
           leaf_font_size=10);
plt.show()

plt.figure(figsize= (14,5))
plt.title("Hierarchical Cluster Dendrogram")
plt.xlabel("X")
plt.ylabel("Y")
dendrogram(hc_average,
           truncate_mode = "lastp",
           p = 10,
           show_contracted= True,
           leaf_font_size=10);
plt.show()

#%%
# Principal Component Analysis
df = pd.read_csv("Hitters.csv")
df.dropna(inplace = True)
df = df._get_numeric_data()
df.head()

from sklearn.preprocessing import StandardScaler
df = StandardScaler().fit_transform(df)

from sklearn.decomposition import PCA

pca = PCA(n_components= 2)
pca_fit = pca.fit_transform(df)

component_df = pd.DataFrame(data = pca_fit, columns = ["First Component", "Second Component"])

pca.explained_variance_ratio_
#%%
# optimum component number

pca = PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Component Number")
plt.ylabel("Variance Ratio");

# final model
pca = PCA(n_components= 3)
pca_fit = pca.fit_transform(df)
pca.explained_variance_ratio_



#%%
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn_som.som import SOM
from minisom import MiniSom
from sklearn.utils.multiclass import unique_labels
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
#######################################################################################

# data loading and splitting
df = pd.read_csv('Assignment3_dataset.csv')
x = df.iloc[:, 0:8]
y = df.iloc[:, 8]

df_statistics = df.describe()

x_train, x_test, y_train, y_test = train_test_split(x, y,
                        test_size=0.25, random_state=42)

# models fitting and prediction
def models(model, x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    model = model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return  model, y_pred, accuracy, report, cm

def tsne(x,y, title):
    tsne = TSNE(n_components=2, random_state=0)
    x_tsne = tsne.fit_transform(x)
    #visualize the data
    target_ids = range(len(y))
    plt.figure(figsize=(7,7))
    colors = 'orange', 'purple'
    for i, c, label in zip(target_ids, colors, y.unique()):
        plt.scatter(x_tsne[y == i, 0], x_tsne[y == i, 1], c=c, label=label)
    plt.title(title)
    plt.legend()
    plt.show()
      
#---------------------------------------------MAIN---------------------------------

#-------------------------- Question (1-a) -----------------------------

KNN, LR  = KNeighborsClassifier(n_neighbors=10), LogisticRegression()
# Logistic Regression (LR)
model_LR, y_pred_LR, accuracy_LR, report_LR, cm_LR = models(LR, x, y)
# K-Nearest Neighbor (KNN)
model_KNN, y_pred_KNN, accuracy_KNN, report_KNN, cm_KNN = models(KNN, x, y)

#-------------------------- Question (1-b) -----------------------------
#first tnse on training data
tsne_train = tsne(x_train,y_train, 'first T-SNE on training set')

#first tnse on testing data
tsne_test = tsne(x_test,y_test, 'first T-SNE on testing set')

#-------------------------- Question (2- a & b) -----------------------------
k_range = [2, 3, 4, 5, 6]
silhouette_avg = []
for k in k_range:
     # initialise kmeans
     kmeans = KMeans(n_clusters=k)
     kmeans = kmeans.fit(x)
     cluster_labels = kmeans.labels_ 
     # silhouette score
     silhouette_avg.append(silhouette_score(x, cluster_labels, metric='euclidean'))

plt.plot(k_range,silhouette_avg,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Silhouette score') 
plt.title('Silhouette analysis For Optimal k')
plt.show()
#from the plot we got that the best k is 2 because it gives the highset silhouette score
  
#-------------------------- Question (2-c) -----------------------------
# K-Means
KM = KMeans(n_clusters = 2)
model_KM, y_pred_KM, accuracy_KM, report_KM, cm_KM = models(KM, x, y)

x_pca = PCA(2).fit_transform(x)
#plotting the results:
sns.scatterplot(x = x_pca[:,0], y = x_pca[:,1] , hue = y)

#-------------------------- Question (3- a & b) -----------------------------
# PCA with LR
acc_LR_dict = {}
for n in range(1,9):
  xs_pca = PCA(n_components=n, random_state=0).fit_transform(x)
  x_train_pca, x_test_pca, y_train_pca, y_test_pca = train_test_split(xs_pca, y,
                          test_size=0.25, random_state=42)
  LR.fit(x_train_pca, y_train)
  y_pred_pca = LR.predict(x_test_pca)
  acc_pca = accuracy_score(y_test, y_pred_pca)
  acc_LR_dict[n] = acc_pca

# PCA with KNN
acc_KNN_dict = {}
for n in range(1,9):
  xs_pca = PCA(n_components=n, random_state=0).fit_transform(x)
  x_train_pca, x_test_pca, y_train_pca, y_test_pca = train_test_split(xs_pca, y,
                          test_size=0.25, random_state=42)
  KNN.fit(x_train_pca, y_train)
  y_pred_pca = KNN.predict(x_test_pca)
  acc_pca = accuracy_score(y_test, y_pred_pca)
  acc_KNN_dict[n] = acc_pca
#-------------------------- Question (6-c) -----------------------------

# Initialization and training
som_shape = (1,2)
som = MiniSom(som_shape[0], som_shape[1], x_pca.shape[1], sigma=.5, learning_rate=.5,
              neighborhood_function='gaussian', random_seed=10)
#som.random_weights_init(x_pca)
initial_weights = np.array(som.get_weights())
som.train_batch(x_pca, 1000, verbose=True)
final_weights = np.array(som.get_weights())

# each neuron represents a cluster
winner_coordinates = np.array([som.winner(x) for x in x_pca]).T
# with np.ravel_multi_index we convert the bidimensional
# coordinates to a monodimensional index
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)


# plotting the clusters using the first 2 dimentions of the data
for c in np.unique(cluster_index):
    plt.scatter(x_pca[cluster_index == c, 0],
                x_pca[cluster_index == c, 1], label='cluster='+str(c), alpha=.7)

# plotting initial centroids
plt.scatter(initial_weights[:, 0], initial_weights[:, 1], marker='x', 
                    s=80, linewidths=2, color='k', label='centroid')
plt.legend();
plt.title('SOM Initial Positions for BMUs/ Centroids')

# plotting the clusters using the first 2 dimentions of the data
for c in np.unique(cluster_index):
    plt.scatter(x_pca[cluster_index == c, 0],
                x_pca[cluster_index == c, 1],label='cluster='+str(c), alpha=.7)

# plotting final centroids
plt.scatter(final_weights[:, 0], final_weights[:, 1], marker='x', 
                s=80, linewidths=2, color='k', label='centroid')
plt.legend();
plt.title('SOM Final Positions for BMUs/ Centroids')




#-------------------------- Question (7) -----------------------------


# Compute DBSCAN
epsilon_range = np.arange(0.3,0.8,0.1)
mid_points_range = np.arange(2,16)
all_clusters = []
all_epsilons = []
all_mid_points = []
epsilon_midpoints_combinations = []

for epsilon in epsilon_range:
    for mid_points in mid_points_range:
        DB = DBSCAN(eps=epsilon, min_samples=mid_points)
        DB_predictions = DB.fit_predict(x)
        clusters = np.unique(DB_predictions)
        all_clusters.append(len(unique_labels(clusters)))
        all_epsilons.append(epsilon)
        all_mid_points.append(mid_points)
        epsilon_midpoints_combinations.append((epsilon, mid_points))
       
      
#Q7-a Epsilon vs number of clusters
plt.figure(figsize=(10,5))
plt.plot(all_epsilons, all_clusters)
plt.title('Epsilon Vs. Number of Clusters')
plt.xlabel("Epsilon")
plt.ylabel("Number of clusters")


#Q7-b MidPoints vs number of clusters
plt.figure(figsize=(10,5))
plt.plot(all_mid_points, all_clusters)
plt.title('MidPoints Vs. Number of Clusters')
plt.xlabel("MidPoints")
plt.ylabel("Number of clusters")


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


dataSet=pd.read_csv("mall.csv")
X=dataSet.iloc[:,[3,4]].values




from scipy.cluster.hierarchy import dendrogram,linkage
 

dendograms=dendrogram(linkage(X, 'ward'))
plt.show()


from sklearn.cluster import AgglomerativeClustering as Ac

hierarchialClustering=Ac(n_clusters=5,affinity="euclidean",linkage='ward')
Y_pred=hierarchialClustering.fit_predict(X)

plt.scatter(X[Y_pred==0,0], X[Y_pred==0,1], color='red')
plt.scatter(X[Y_pred==1,0], X[Y_pred==1,1], color='blue')
plt.scatter(X[Y_pred==2,0], X[Y_pred==2,1], color='green')
plt.scatter(X[Y_pred==3,0], X[Y_pred==3,1], color='orange')

plt.scatter(X[Y_pred==4,0], X[Y_pred==4,1], color='violet')






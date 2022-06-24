#import libraries
import numpy as np
##from scipy.spatial import distance
data=np.genfromtxt("Digits089.csv",delimiter=",")
Xtrain=data[data[:,0]!=5,2:]
ytrain=data[data[:,0]!=5,1]
Xtest=data[data[:,0]==5,2:]
ytest=data[data[:,0]==5,1]


class Kmeans:
    def __init__(self,k=8): # k is number of clusters
        self.num_cluster = k
        self.center = None# centers for different clusters
        self.cluster_label = np.zeros([k,]) # class labels for different clusters
        self.error_history = []

    def fit(self, X, y):
        # initialize the centers of clutsers as a set of pre-selected samples
        init_idx = [1, 200, 500, 1000, 1001, 1500, 2000, 2005]

        num_iter = 0 # number of iterations for convergence
        # initialize the cluster assignment
        prev_cluster_assignment = np.zeros([len(X),]).astype('int')
        cluster_assignment = np.zeros([len(X),]).astype('int')
        is_converged = False

        self.center = X[init_idx]
        # iteratively update the centers of clusters till convergence

        while not is_converged:

            # iterate through the samples and compute their cluster assignment (E step)
            for i in range(len(X)):
                # use euclidean distance to measure the distance between sample and cluster centers
                dis = [0,0,0,0,0,0,0,0]
                
                for j in range(len(self.center)):
                    dis[j] = np.square(np.linalg.norm(X[i] - self.center[j]))

                # determine the cluster assignment by selecting the cluster whose center is closest to the sample
                cluster_assignment[i] = dis.index(np.min(dis))
            for i in range(len(self.center)):
                self.center[i] = np.mean(X[cluster_assignment == i], axis=0)

            cur_error = self.compute_error(X, cluster_assignment)
            self.error_history.append(cur_error)

            # reach convergence if the assignment does not change anymore
            is_converged = True if (cluster_assignment==prev_cluster_assignment).sum() == len(X) else False
            prev_cluster_assignment = np.copy(cluster_assignment)
            num_iter += 1


        # compute the class label of each cluster based on majority voting (remember to update the corresponding class attribute)
        l1 = [0,0,0,0,0,0,0,0]
        l2 = [0,0,0,0,0,0,0,0]
        l3 = [0,0,0,0,0,0,0,0]
        ## summary total number of voting for each cluster
        for i in range(len(X)):  
            if y[i] == 0:
                l1[cluster_assignment[i]] += 1
                
            elif y[i] == 8:
                l2[cluster_assignment[i]] += 1
                           
            elif y[i] == 9:
                l3[cluster_assignment[i]] += 1
            
        ## determine lebel based on majority voting    
        for i in range(len(self.cluster_label)):
            if np.max([l1[i],l2[i],l3[i]]) == l1[i]:
                self.cluster_label[i] = 0
            elif np.max([l1[i],l2[i],l3[i]]) == l2[i]:
                self.cluster_label[i] = 8
            else:
                self.cluster_label[i] = 9



        return num_iter, self.error_history
        

    def predict(self,X):
        # predicting the labels of test samples based on their clustering results
        prediction = np.ones([len(X),]) # placeholder

        # iterate through the test samples
        for i in range(len(X)):
            # find the cluster of each sample
            dis_min= 100000000
            minIndex = -1
            for j in range(len(self.center)):
                dis = np.square(np.linalg.norm(X[i]-self.center[j]))
                if(dis<dis_min):   
                    dis_min=dis
                    minIndex = j
            prediction[i] = self.cluster_label[minIndex]
            # use the class label of the selected cluster as the predicted class

            

        return prediction

    def compute_error(self,X,cluster_assignment):
        # compute the reconstruction error for given cluster assignment and centers
        error = 0 # placeholder
        for i in range(len(X)):
            error = error + np.linalg.norm(X[i] - self.center[cluster_assignment[i]])**2
        return error

    def params(self):
        return self.center, self.cluster_label

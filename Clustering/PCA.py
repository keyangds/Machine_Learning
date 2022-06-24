import numpy as np

class PCA():
    def __init__(self,num_dim=None):
        self.num_dim = num_dim
        self.mean = np.zeros([1,784]) # means of training data
        self.W = None # projection matrix

    def fit(self,X):
        # normalize the data to make it centered at zero (also store the means as class attribute)
        self.mean = np.mean(X,axis=0)
        X_center = X - self.mean
        # finding the projection matrix that maximize the variance (Hint: for eigen computation, use numpy.eigh instead of numpy.eig)
        lam, w = np.linalg.eigh(np.cov(np.transpose(X_center)))
        
        if self.num_dim is None:
            # select the reduced dimension that keep >90% of the variance
            sum = 0
            var = 0
            dim = 0
            
            for i in range(len(lam)):
                sum = sum + lam[i]     ## sum of all variance
            
            lam = lam[::-1]    ## make lambda in descending order
            w = w.T[::-1] ## make eigenvector's sequence corresponding to eienvalue
            ratio = 0
            while ratio < 0.9:    ## when the POV arrive 90% , stop adding
                var = var + lam[dim]
                ratio = var/sum
                dim = dim + 1
            
            # store the projected dimension
            self.num_dim = dim # placeholder
            
            

        # determine the projection matrix and store it as class attribute
     
        self.W = w[:self.num_dim] # placeholder

        # project the high-dimensional data to low-dimensional one
        X = X_center.dot(np.transpose(self.W)) # placeholder
        X_pca = X
        return X_pca, self.num_dim

    def predict(self,X):
        # normalize the test data based on training statistics
        X_center = X - self.mean
        
        # project the test data
        
        
        X_pca = X_center.dot(np.transpose(self.W)) # placeholder

        return X_pca

    def params(self):
        return self.W, self.mean, self.num_dim

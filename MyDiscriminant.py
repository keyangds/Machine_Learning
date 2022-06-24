import numpy as np

df = np.genfromtxt("training_data.txt",delimiter=",")
dftest = np.genfromtxt("test_data.txt",delimiter=",")
Xtrain = df[:,0:8]
ytrain = df[:,8]
Xtest = dftest[:,0:8]
ytest = dftest[:,8]

class GaussianDiscriminant:
    def __init__(self,k=2,d=8,priors=None,shared_cov=False):
        self.mean = np.zeros((k,d)) # mean
        self.shared_cov = shared_cov # using class-independent covariance or not
        if self.shared_cov:
            self.S = np.zeros((d,d)) # class-independent covariance (S1=S2)
        else:
            self.S = np.zeros((k,d,d)) # class-dependent covariance (S1!=S2)
        if priors is not None:
            self.p = priors
        else:
            self.p = [1.0/k for i in range(k)] # assume equal priors if not given
        self.k = k
        self.d = d

    def fit(self, Xtrain, ytrain):
        # compute the mean for each class
        c1 = []
        c2 = []
        
        for i in range(np.size(ytrain)):
            if ytrain[i] == 1:  ## split test set based on classfication result
                c1.append(Xtrain[i,:]) 
            else:
                c2.append(Xtrain[i,:])
        
        c1 = np.asarray(c1)
        c2 = np.asarray(c2)
        
        ## do mean of each column in dataset
        m1 = np.mean(c1, axis=0)
        m2 = np.mean(c2, axis=0)
        self.mean[0] = m1
        self.mean[1] = m2
        
        
        S1 = np.zeros((self.d, self.d))
        S2 = np.zeros((self.d, self.d))
        if self.shared_cov:
            # compute the class-independent covariance
            self.S = np.cov(Xtrain, rowvar = False, ddof=0)
        
            pass # placeholder
        else:
            # compute the class-dependent covariance
            S1 = np.cov(c1,rowvar = False, ddof=0)
            S2 = np.cov(c2,rowvar = False, ddof=0)
            self.S[0] = S1
            self.S[1] = S2

    def predict(self, Xtest):
        # predict function to get predictions on test set
        predicted_class = np.ones(Xtest.shape[0]) # placeholder
        g = np.zeros(self.k)
        for i in np.arange(Xtest.shape[0]): # for each test set example
            
            # calculate the value of discriminant function for each class
            for c in range(0,self.k):
                if self.shared_cov: 
                    w = np.dot(np.linalg.inv(self.S), self.mean[c,:]) 
                    
                    w0 = -(1/2) * np.dot(np.dot(np.transpose(self.mean[c]), np.linalg.inv(self.S)), self.mean[c,:]) + np.log(self.p[c])
                    
                    g[c] = np.dot(np.transpose(w), Xtest[i,:]) + w0    ## (Wi)'x+Wi0
                    
                    pass # placeholder
                else:   
                    W = (-1/2) * (np.linalg.inv(self.S[c]))
                    w0 = (-1/2) * np.dot(np.dot(np.transpose(self.mean[c]), np.linalg.inv(self.S[c])), self.mean[c]) - (1/2) * np.log(np.linalg.det(self.S[c])) + np.log(self.p[c])
                    wi = np.dot(np.linalg.inv(self.S[c]), self.mean[c])
                    g[c] = np.dot(np.dot(Xtest[i], W), np.transpose(Xtest[i])) + np.dot(np.transpose(wi), Xtest[i]) + w0
                    pass

            # determine the predicted class based on the values of discriminant function
            if g[0] > g[1]:
                predicted_class[i] = 1
            else:
                predicted_class[i] = 2

        return predicted_class

    def params(self):
        if self.shared_cov:
            return self.mean[0], self.mean[1], self.S
        else:
            return self.mean[0],self.mean[1],self.S[0],self.S[1]


class GaussianDiscriminant_Diagonal:
    def __init__(self,k=2,d=8,priors=None):
        self.mean = np.zeros((k,d)) # mean
        self.S = np.zeros((d,)) # variance
        if priors is not None:
            self.p = priors
        else:
            self.p = [1.0/k for i in range(k)] # assume equal priors if not given
        self.k = k
        self.d = d
        
    def fit(self, Xtrain, ytrain):
        # compute the mean for each class
        c1 = []
        c2 = []
        
        for i in range(np.size(ytrain)):
            if ytrain[i] == 1:
                c1.append(Xtrain[i,:])
            else:
                c2.append(Xtrain[i,:])
        
        c1 = np.asarray(c1)
        c2 = np.asarray(c2)

        m1 = np.mean(c1, axis=0)
        m2 = np.mean(c2, axis=0)
        self.mean[0] = m1
        self.mean[1] = m2

        # compute the variance of different features
        self.S = np.var(Xtrain,axis=0)

        pass # placeholder

    def predict(self, Xtest):
        # predict function to get prediction for test set
        predicted_class = np.ones(Xtest.shape[0]) # placeholder
        
        g = np.zeros(self.k)
        
        for i in range(0,Xtest.shape[0]): # for each test set example
            # calculate the value of discriminant function for each class
            for c in range(0,self.k):
                s = 0
                for j in range(0,self.d):
                    s += ((Xtest[i,j] - self.mean[c,j]) / np.sqrt(self.S[j]))**2
                g[c] = (-1/2) * s + np.log(self.p[c])
                pass
                
            if g[0] > g[1]:
                predicted_class[i] = 1
            else:
                predicted_class[i] = 2

            # determine the predicted class based on the values of discriminant function

        return predicted_class

    def params(self):
        return self.mean[0], self.mean[1], self.S

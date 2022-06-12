"""
Below is the pseudo code that you may follow to create your python user defined function.

Your function is expected to return :-
    1. number of iterations / passes it takes until your weight vector stops changing
    2. final weight vector w
    3. error rate (fraction of training samples that are misclassified)

def keyword can be used to create user defined functions. Functions are a great way to
create code that can come in handy again and again. You are expected to submit a python file
with def MyPerceptron(X,y,w) function.

"""
# Hints
# one can use numpy package to take advantage of vectorization
# matrix multiplication can be done via nested for loops or
# matmul function in numpy package


# Header
import numpy as np

data = np.genfromtxt('AltData.csv',delimiter=',')
X = data[:,:2]
r = data[:,2]

# Implement the Perceptron algorithm
def MyPerceptron(X,y,w0=[1.0,-1.0]):
    k = 0 # initialize variable to store number of iterations it will take
          # for your perceptron to converge to a final weight vector
    w=w0
    error_rate = 1.00

    # loop until convergence (w does not change at all over one pass)
    # or until max iterations are reached
    # (current pass w ! = previous pass w), then do:
    w_past=[0.0,0.0]
    while np.array_equal(w,w_past) == False:
    
        w_past = w
        # for each training sample (x,y):
        for i in range(len(X)):
        
            # if actual target y does not match the predicted target value, update the weights
            if y[i]*np.matmul(w,X[i])<=0:
                w = w + np.dot(y[i],X[i])

            # calculate the number of iterations as the number of updates
        
        k += 1
        


    # make prediction on the csv dataset using the feature set
    # Note that you need to convert the raw predictions into binary predictions using threshold 0
    predict = np.empty([100,1])
    for i in range(100):
        predict[i] = np.sign(np.matmul(w,X[i]))

    # compute the error rate
    # error rate = ( number of prediction ! = y ) / total number of training examples
    error = 0
    for i in range(100):
        if predict[i] != y[i]:
            error +=1
            
    error_rate = error/len(X)  

    return (w, k, error_rate)

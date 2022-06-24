import numpy as np
import math as m
class Normalization:
    def __init__(self,):
        self.mean = np.zeros([1,64]) # means of training features
        self.std = np.zeros([1,64]) # standard deviation of training features

    def fit(self,x):
        # compute the statistics of training samples (i.e., means and std)
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        pass # placeholder

    def normalize(self,x):
        # normalize the given samples to have zero mean and unit variance (add 1e-15 to std to avoid numeric issue)
        x = (x - self.mean) / (self.std+1e-15)
        return x

def process_label(label):
    # convert the labels into one-hot vector for training
    one_hot = np.zeros([len(label),10])
    for i in range(len(label)):
        one_hot[i][label[i]] = 1

    return one_hot

def tanh(x):
    # implement the hyperbolic tangent activation function for hidden layer
    x = np.clip(x,a_min=-100,a_max=100) # for stablility, do not remove this line

    f_x = (m.e**x-m.e**(-x))/(m.e**x+m.e**(-x))

    return f_x

def softmax(x):
    # implement the softmax activation function for output layer
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    f_x = e_x / np.sum(e_x, axis=-1, keepdims=True)

    return f_x

class MLP:
    def __init__(self,num_hid):
        # initialize the weights
        self.weight_1 = np.random.random([64,num_hid])
        self.bias_1 = np.random.random([1,num_hid])
        self.weight_2 = np.random.random([num_hid,10])
        self.bias_2 = np.random.random([1,10])

    def fit(self,train_x,train_y, valid_x, valid_y):
            # learning rate
        lr = 5e-3
            # counter for recording the number of epochs without improvement
        count = 0
        best_valid_acc = 0
      
        while count<=50:
            # training with all samples (full-batch gradient descents)
            # implement the forward pass (from inputs to predictions)
            hidden_output = self.get_hidden(train_x)
            
         
            y_pred = softmax(hidden_output.dot(self.weight_2) + self.bias_2)
            
      
            grad_out = (y_pred - train_y) * y_pred * (1 - y_pred)
            grad_v1 = hidden_output.T.dot(grad_out)
            grad_v0 = np.sum(grad_out, axis=0)
               
            grad_hidden = grad_out.dot(self.weight_2.T)*(1 - hidden_output**2)
            grad_w = train_x.T.dot(grad_hidden)
            grad_w0 = np.sum(grad_hidden, axis=0)



            #update the parameters based on sum of gradients for all training samples
            self.weight_1 -= lr * grad_w
            self.bias_1 -= lr * grad_w0
            self.weight_2 -= lr * grad_v1
            self.bias_2 -= lr * grad_v0
            
            # evaluate on validation data
            predictions = self.predict(valid_x)
            valid_acc = np.count_nonzero(predictions.reshape(-1)==valid_y.reshape(-1))/len(valid_x)
            # compare the current validation accuracy with the best one
            if valid_acc>best_valid_acc:
                best_valid_acc = valid_acc
                count = 0
            else:
                count += 1
        return best_valid_acc

    def predict(self,x):
        # generate the predicted probability of different classes
        hidden_input = x.dot(self.weight_1) + self.bias_1
        hidden_output = tanh(hidden_input)
        output_layer_input = hidden_output.dot(self.weight_2) + self.bias_2
        y_pred = softmax(output_layer_input)
        y_pred = np.argmax(y_pred, axis=1)
        # convert class probability to predicted labels
        # y = np.zeros([len(x),]).astype('int') # placeholder
        return y_pred

    def get_hidden(self,x):
        # extract the intermediate features computed at the hidden layers (after applying activation function)
        hidden_input = x.dot(self.weight_1) + self.bias_1
        hidden_output = tanh(hidden_input)
        z = hidden_output


        return z

    def params(self):
        return self.weight_1, self.bias_1, self.weight_2, self.bias_2

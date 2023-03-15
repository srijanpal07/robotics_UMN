#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

def preprocess_data(data, mean=None, std=None):
    # for this function, you will need to normalize the data, to make it have zero mean and unit variance
    # to avoid the numerical issue, we can add 1e-15 to the std
    # it has different process in train set, and validation/test set
    if mean is not None or std is not None:
        # mean and std is precomputed with the training data
        # ------------------------------------------------------------------------------------------------------------
        # complete your code here
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i][j] = (data[i][j]-mean[j])/(std[j] + 1e-15)

        # ------------------------------------------------------------------------------------------------------------
        return data
    else:
        # compute the mean and std based on the training data
        # ------------------------------------------------------------------------------------------------------------
        # complete your code here
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)  # the placeholder for mean and std   
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i][j] = (data[i][j]-mean[j])/(std[j] + 1e-15)
        # ------------------------------------------------------------------------------------------------------------


        return data, mean, std

def preprocess_label(label):
    # to handle the loss function computation, convert the labels into one-hot vector for training
    one_hot = np.zeros([len(label), 10])
    # ------------------------------------------------------------------------------------------------------------
    # complete your code here
    for i in range(len(label)):
        for j in range(10):
            if label[i] == j:
                one_hot[i][j] = 1
    # ------------------------------------------------------------------------------------------------------------
    return one_hot

def sigmoid(x):
    # implement the sigmoid activation function for hidden layer
    # ------------------------------------------------------------------------------------------------------------
    # complete your code here
    f_x = 1/(1 + np.exp(-x))  # placeholder, you need to change it to the corresponding function
    # ------------------------------------------------------------------------------------------------------------

    return f_x

def Relu(x):
    # implement the Relu activation function for hidden layer
    # ------------------------------------------------------------------------------------------------------------
    # complete your code here
    f_x = np.maximum(0,x)
    # ------------------------------------------------------------------------------------------------------------

    return f_x

def tanh(x):
    # implement the tanh activation function for hidden layer
    # ------------------------------------------------------------------------------------------------------------
    # complete your code here
    f_x = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x)) # placeholder, you need to change it to the corresponding function
    # ------------------------------------------------------------------------------------------------------------

    return f_x


def softmax(x):
    # implement the softmax activation function for output layer
    e = np.exp(x - np.max(x))
    f_x = e / np.sum(e, axis = 1, keepdims = True) # placeholder, you need to change it to the corresponding function

    return f_x

class MLP:
    def __init__(self, num_hid, activation="Relu"):
        # initialize the weights
        np.random.seed(2022)
        self.weight_1 = np.random.random([64, num_hid]) / 100
        self.bias_1 = np.random.random([1, num_hid]) / 100
        self.weight_2 = np.random.random([num_hid, 10]) / 100
        self.bias_2 = np.random.random([1, 10]) / 100

        # not that in your implementation, you need to consider your selected activation.
        self.activation = activation

    def fit(self, train_x, train_y, valid_x, valid_y):
        # initialize learning rate
        lr = 5e-4
        # initialize the counter of recording the number of epochs that the model does not improve
        # and log the best validation accuracy
        count = 0
        best_valid_acc = 0

        """
        You also need to stopping criteria the training if we find no improvement over the best_valid_acc for more than 100 iterations
        """
        while count <= 100:
            # in this case, you will train all the samples (full-batch gradient descents)
            # implement the forward pass,
            # you also need to consider the specific selected activation function on hidden layer
            # ------------------------------------------------------------------------------------------------------------
            # complete your code here
            z1 = np.dot(train_x, self.weight_1) + self.bias_1
            if self.activation == "Sigmoid":
                a1 = sigmoid(z1)
            elif self.activation == "Relu":
                a1 = Relu(z1)
            elif self.activation == "tanh":
                a1 = tanh(z1)
            z2 = np.dot(a1, self.weight_2) + self.bias_2
            yHat = softmax(z2)
            # ------------------------------------------------------------------------------------------------------------

            
            
            # implement the backward pass (also called the backpropagation)
            # compute the gradients for different parameters, e.g. self.weight_1, self.bias_1, self.weight_2, self.bias_2
            # you also need to consider the specific selected activation function on hidden layer
            # ------------------------------------------------------------------------------------------------------------
            # complete your code here
            delta2 = yHat - train_y
            dE_dw2 = np.dot(np.transpose(a1), delta2)
            dE_db2 = np.sum(delta2, axis=0)
            
            if self.activation == "Sigmoid":
                bk = a1*(1 - a1)
            elif self.activation == "Relu":
                a1[a1<=0] = 0
                a1[a1>0] = 1
                bk = a1
            elif self.activation == "tanh":
                bk = 1 - np.square(a1)
            delta1 = bk * np.dot(delta2, self.weight_2.T)
            dE_dw1 = np.dot(train_x.T, delta1)
            dE_db1 = np.sum(delta1, axis=0)
            # ------------------------------------------------------------------------------------------------------------

            
            
            # update the corresponding parameters based on sum of gradients for above the training samples
            # ------------------------------------------------------------------------------------------------------------
            # complete your code here
            self.weight_1 = self.weight_1 - lr*dE_dw1
            self.bias_1 = self.bias_1 - lr*dE_db1.sum(axis=0)
            
            self.weight_2 = self.weight_2 - lr*dE_dw2
            self.bias_2 = self.bias_2 - lr*dE_db2.sum(axis=0)
            # ------------------------------------------------------------------------------------------------------------
                
            # evaluate the accuracy on the validation data
            predictions = self.predict(valid_x)
            cur_valid_acc = (predictions.reshape(-1) == valid_y.reshape(-1)).sum() / len(valid_x)

            # compare the current validation accuracy, if cur_valid_acc > best_valid_acc, we will increase count by it
            if cur_valid_acc > best_valid_acc:
                best_valid_acc = cur_valid_acc
                count = 0
            else:
                count += 1

        return best_valid_acc

    def predict(self, x):
        # go through the MLP and then obtain the probability of each category
        # you also need to consider the specific selected activation function on hidden layer
        # ------------------------------------------------------------------------------------------------------------
        # complete your code here

        z1 = np.dot(x, self.weight_1) +self.bias_1
        if self.activation == "Sigmoid":
            a1 = sigmoid(z1)
        elif self.activation == "Relu":
            a1 = Relu(z1)
        elif self.activation == "tanh":
            a1 = tanh(z1)
        z2 = np.dot(a1, self.weight_2) + self.bias_2
        y1 = softmax(z2)
        
        # ------------------------------------------------------------------------------------------------------------

        # convert category probability to the corresponding predicted labels
        # ------------------------------------------------------------------------------------------------------------
        # complete your code here
        y = np.argmax(y1, axis=1)
        # ------------------------------------------------------------------------------------------------------------

        return y

    def get_hidden(self, x):
        # obtain the hidden layer features, the one after applying activation function
        # you also need to consider the specific selected activation function on hidden layer
        # ------------------------------------------------------------------------------------------------------------
        # complete your code here
        #z = x  # placeholder
        z1 = np.dot(x, self.weight_1) +self.bias_1
        if self.activation == "Sigmoid":
            a1 = sigmoid(z1)
        elif self.activation == "Relu":
            a1 = Relu(z1)
        elif self.activation == "tanh":
            a1 = tanh(z1)
        z2 = np.dot(a1, self.weight_2) + self.bias_2
        y1 = softmax(z2)
        z = [a1, y1]

        # ------------------------------------------------------------------------------------------------------------

        return z

    def params(self):
        return self.weight_1, self.bias_1, self.weight_2, self.bias_2


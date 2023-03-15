import numpy as np

def preprocess_data(data, mean=None, std=None):
    # for this function, you will need to normalize the data, to make it have zero mean and unit variance
    # to avoid the numerical issue, we can add 1e-15 to the std

    # it has different process in train set, and validation/test set
    if mean is not None or std is not None:
        # mean and std is precomputed with the training data
        # ------------------------------------------------------------------------------------------------------------
        # complete your code here

        # ------------------------------------------------------------------------------------------------------------

        return data
    else:
        # compute the mean and std based on the training data
        # ------------------------------------------------------------------------------------------------------------
        # complete your code here
        mean = std = 0  # the placeholder for mean and std
        # ------------------------------------------------------------------------------------------------------------


        return data, mean, std

def preprocess_label(label):
    # to handle the loss function computation, convert the labels into one-hot vector for training
    one_hot = np.zeros([len(label), 10])
    # ------------------------------------------------------------------------------------------------------------
    # complete your code here

    # ------------------------------------------------------------------------------------------------------------

    return one_hot

def sigmoid(x):
    # implement the sigmoid activation function for hidden layer
    # ------------------------------------------------------------------------------------------------------------
    # complete your code here
    f_x = x  # placeholder, you need to change it to the corresponding function
    # ------------------------------------------------------------------------------------------------------------

    return f_x

def Relu(x):
    # implement the Relu activation function for hidden layer
    # ------------------------------------------------------------------------------------------------------------
    # complete your code here
    f_x = x # placeholder, you need to change it to the corresponding function
    # ------------------------------------------------------------------------------------------------------------

    return f_x

def tanh(x):
    # implement the tanh activation function for hidden layer
    # ------------------------------------------------------------------------------------------------------------
    # complete your code here
    f_x = x # placeholder, you need to change it to the corresponding function
    # ------------------------------------------------------------------------------------------------------------

    return f_x


def softmax(x):
    # implement the softmax activation function for output layer
    f_x = x # placeholder, you need to change it to the corresponding function

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

            # ------------------------------------------------------------------------------------------------------------

            # implement the backward pass (also called the backpropagation)
            # compute the gradients for different parameters, e.g. self.weight_1, self.bias_1, self.weight_2, self.bias_2
            # you also need to consider the specific selected activation function on hidden layer
            # ------------------------------------------------------------------------------------------------------------
            # complete your code here

            # ------------------------------------------------------------------------------------------------------------

            # update the corresponding parameters based on sum of gradients for above the training samples
            # ------------------------------------------------------------------------------------------------------------
            # complete your code here

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

        # ------------------------------------------------------------------------------------------------------------

        # convert category probability to the corresponding predicted labels
        # ------------------------------------------------------------------------------------------------------------
        # complete your code here
        y = np.zeros([len(x), ]).astype('int')  # a placeholder
        # ------------------------------------------------------------------------------------------------------------

        return y

    def get_hidden(self, x):
        # obtain the hidden layer features, the one after applying activation function
        # you also need to consider the specific selected activation function on hidden layer
        # ------------------------------------------------------------------------------------------------------------
        # complete your code here
        z = x  # placeholder


        # ------------------------------------------------------------------------------------------------------------

        return z

    def params(self):
        return self.weight_1, self.bias_1, self.weight_2, self.bias_2

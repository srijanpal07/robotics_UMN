#import libraries for the simple mlp
import numpy as np
from MySimpleMLP import preprocess_data, preprocess_label, MLP

def load_data(split):
    data = np.genfromtxt("optdigits_{}.txt".format(split), delimiter=",")
    x = data[:, :-1]
    y = data[:, -1].astype('int')
    return x, y

# read the data from the dataset.
# training data
train_x, train_y = load_data("train")

# validation data
valid_x, valid_y = load_data("valid")

# test data
test_x, test_y = load_data("test")


# you will need to process the data, normalize the data on the validation and test set
train_x, mean, std = preprocess_data(train_x)
valid_x = preprocess_data(valid_x, mean, std)
test_x = preprocess_data(test_x, mean, std)

# process training labels into one-hot vectors
train_y = preprocess_label(train_y)

################################## Problem a ######################################
# conduct the experiment with different numbers of hidden units
candidate_num_hid_list = [5, 10, 15, 20, 25]
activation = "Sigmoid"
valid_accuracy = []
store_models = []
for i, num_hid in enumerate(candidate_num_hid_list):
    # call the model
    clf = MLP(num_hid=num_hid, activation=activation)
    # to update the model based on training data
    # record the best validation accuracy
    cur_valid_accuracy = clf.fit(train_x, train_y, valid_x, valid_y)
    valid_accuracy.append(cur_valid_accuracy)
    store_models.append(clf)
    print('The validation accuracy for the hyper-parameter {} hidden units is {:.3f}'.
          format(candidate_num_hid_list[i], cur_valid_accuracy))

# find the model with the best validation accuracy
best_num_hid = candidate_num_hid_list[np.argmax(valid_accuracy)]
best_num_hid_idx = np.argmax(valid_accuracy)
best_clf = store_models[best_num_hid_idx]

# report the accuracy of test data
predictions = best_clf.predict(test_x)
test_accuracy = (predictions.reshape(-1) == test_y.reshape(-1)).sum() / len(test_x)

print('The Test accuracy with {} hidden units is {:.3f}'.format(best_num_hid, test_accuracy))



################################## Problem b ######################################
# experiment with different type of the activation function
# conduct the experiment with different activation function
num_hid = 10
activations_list = ["Sigmoid", "Relu", "tanh"]
valid_accuracy = []
store_models = []
for i, activation in enumerate(activations_list):
    # call the model
    clf = MLP(num_hid=num_hid, activation=activation)
    # to update the model based on training data
    # record the best validation accuracy
    cur_valid_accuracy = clf.fit(train_x, train_y, valid_x, valid_y)
    valid_accuracy.append(cur_valid_accuracy)
    store_models.append(clf)
    print('The validation accuracy for the hyper-parameter {} activation is {:.3f}'.
          format(activations_list[i], cur_valid_accuracy))

# find the model with the best validation accuracy
best_activations = activations_list[np.argmax(valid_accuracy)]
best_activations_idx = np.argmax(valid_accuracy)
best_clf = store_models[best_activations_idx]

# report the accuracy of test data
predictions = best_clf.predict(test_x)
test_accuracy = (predictions.reshape(-1) == test_y.reshape(-1)).sum() / len(test_x)


print('The Test accuracy with {} activation is {:.3f}'.format(best_activations, test_accuracy))

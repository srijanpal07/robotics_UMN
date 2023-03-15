#import libraries
import numpy as np
from MyTrochMLP import MyPyTorchMLP
import torch

## You can set logging_on=True to see some training logging information
## set logging_on=False, will not print such information and make the command window more clean.
## Also, you will only see the information you need to answer the question.
logging_on = False


############### Problem a ###################
# remember to implement your code in MyTrochMLP.MyPyTorchMLP.stopping_criteria


############### Problem b ###################
# remember to implement your code in MyTrochMLP.Net
print("############################## Problem b ##########################################################")
validation_set_for_each_layer = []
layer_nums = [1, 2, 3, 4]
hidden_units = [
    (512, ),
    (256, 64),
    (128, 64, 32),
    (128, 64, 32, 16),
]
store_model = []
for layer_num, hidden_unit in zip(layer_nums, hidden_units):
    trainer = MyPyTorchMLP(hidden_layer_num=layer_num, hidden_unit_num_list=hidden_unit, logging_on=logging_on)
    trainer.train()

    validation_set_for_each_layer.append(trainer.best_validation_accuracy())
    store_model.append(trainer)

print("####################################################################################################")
for idx in range(len(validation_set_for_each_layer)):
    print("For the number of hidden layer {}, with hidden unit {}, the best validation accuracy is {}".
          format(layer_nums[idx], hidden_units[idx], validation_set_for_each_layer[idx]))

best_validation_idx = np.argmax(validation_set_for_each_layer)
best_test_accuracy = store_model[best_validation_idx].evaluation("test")
print("##################################################")
print("The accuracy of test set is {}".format(best_test_accuracy))
print("The corresponding hyper-parameter is hidden layer {}, with hidden unit {}".
          format(layer_nums[best_validation_idx], hidden_units[best_validation_idx]))
print("##################################################")


############### Problem c ###################
# remember to implement your code in MyTrochMLP.Net
print("############################## Problem c ##########################################################")
validation_set_for_each_layer = []
activation_types = ["Sigmoid", "Relu", "tanh"]
store_model = []
for activation_type in activation_types:
    trainer = MyPyTorchMLP(hidden_layer_num=2, hidden_unit_num_list=(256, 128), activation_function=activation_type, logging_on=logging_on)
    trainer.train()

    validation_set_for_each_layer.append(trainer.best_validation_accuracy())
    store_model.append(trainer)

print("####################################################################################################")
for idx in range(len(validation_set_for_each_layer)):
    print("For the number of hidden layer {}, with hidden unit {} and activation function {}, the best validation accuracy is {}".
          format(2, (256, 128), activation_types[idx], validation_set_for_each_layer[idx]))

best_validation_idx = np.argmax(validation_set_for_each_layer)
best_test_accuracy = store_model[best_validation_idx].evaluation("test")
print("##################################################")
print("The accuracy of test set is {}".format(best_test_accuracy))
print("The corresponding hyper-parameter is {} activation function".
          format(activation_types[best_validation_idx]))
print("##################################################")




############### Problem d ###################
# You will implement your code here
print("############################## Problem d ##########################################################")
# ------------------------------------------------------------------------------------------------------------
# complete your code here

# ------------------------------------------------------------------------------------------------------------

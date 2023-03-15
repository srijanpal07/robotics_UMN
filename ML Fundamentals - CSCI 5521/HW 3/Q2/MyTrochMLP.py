import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from os.path import join
import os
import json
import numpy as np
import copy


############################################################################################################
class OptDigits(Dataset):
    """
    The dataset class
    """

    def __init__(self, type="train"):
        self.type = type

        assert self.type == "train" or self.type == "valid" or self.type == "test"

        data = np.genfromtxt("optdigits_{}.txt".format(self.type), delimiter=",")
        x = (data[:, :-1] / 16 - 0.5).astype('float32')
        y = data[:, -1].astype('long')

        # initialize data
        self.data = []
        for idx in range(x.shape[0]):
            self.data.append((x[idx], y[idx]))

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        data = self.data[idx]

        return {
            "feature": data[0],
            "target": data[1]
                }

    def collate_func(self, batch):
        feature_batch = []
        target_batch = []

        for sample in batch:
            tmp_feature, tmp_target = sample["feature"], sample["target"]
            feature_batch.append(tmp_feature)
            target_batch.append(tmp_target)

        data = dict()
        data["feature"] = torch.tensor(np.stack(feature_batch))
        data["target"] = torch.tensor(np.stack(target_batch))

        return data
############################################################################################################

############################################################################################################
class Net(nn.Module):
    def __init__(self, hidden_layer_num=2, hidden_unit_num_list=(256, 128), activation_function="Sigmoid", dropout_rate=0.5):
        super().__init__()
        np.random.seed(2022)
        torch.manual_seed(2022)
        self.hidden_layer_num = hidden_layer_num
        self.hidden_unit_num_list = hidden_unit_num_list
        self.activation_function = activation_function
        self.dropout_rate = dropout_rate

        ## You will define your model architecture here
        # hint you can use nn.ModuleList, which serve the function as list in python
        # some modules you may use:
        # nn.Linear
        # ------------------------------------------------------------------------------------------------------------
        # complete your code here
        self.placeholder = nn.Linear(10, 10) # It is only a placeholder, and will not be used
        # ------------------------------------------------------------------------------------------------------------


        # initialize the weight
        self._initialize_weights()



    def forward(self, x):
        # you will define your forward function
        # some activation function you may use based on your selection of self.activation_function
        # F.softmax, torch.sigmoid, torch.tanh, F.relu
        # ------------------------------------------------------------------------------------------------------------
        # complete your code here
        y = torch.zeros(x.shape[0], 10) + self.placeholder.weight.sum(0, keepdim=True) # It is only a placeholder, and will not be used
        return y
        # ------------------------------------------------------------------------------------------------------------


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.1)
                nn.init.constant_(m.bias, 0)
############################################################################################################



############################################################################################################
class MyPyTorchMLP(nn.Module):
    def __init__(self, batch_size=32, lr=1e-1, epoch=200, hidden_layer_num=2, hidden_unit_num_list=(128, 64),
                 activation_function="Relu", dropout_rate=0.5, check_continue_epoch=10, logging_on=False):
        super().__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.epoch = epoch
        self.hidden_layer_num = hidden_layer_num
        self.hidden_unit_num_list = hidden_unit_num_list
        self.activation_function = activation_function
        self.dropout_rate = dropout_rate
        self.check_continue_epoch = check_continue_epoch
        self.logging_on = logging_on


        # store the accuracy of the validation, which would be used in
        self.valid_accuracy = []

        # obtain the dataset for train, valid, test
        dataset_train = OptDigits(type="train")
        dataset_valid = OptDigits(type="valid")
        dataset_test = OptDigits(type="test")

        # obtain the dataloader
        self.train_loader = DataLoader(
            dataset=dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=dataset_train.collate_func,
            drop_last=True
        )

        self.valid_loader = DataLoader(
            dataset=dataset_valid,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=dataset_valid.collate_func,
            drop_last=False
        )

        self.test_loader = DataLoader(
            dataset=dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=dataset_test.collate_func,
            drop_last=False
        )

        assert self.hidden_layer_num == len(self.hidden_unit_num_list)


        # the instance of the network
        self.net = Net(hidden_layer_num=self.hidden_layer_num,
                       hidden_unit_num_list=self.hidden_unit_num_list,
                       activation_function=self.activation_function)

        # initialize the best net for evaluating the data in test set
        self.net_best_validation = copy.deepcopy(self.net)

        # define the optimizer
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)

        # you will define your forward function
        self.criterion = nn.CrossEntropyLoss()

    def print_msg(self, message: str):
        if self.logging_on:
            print(message)

    def evaluation(self, type="valid"):
        if type == "valid":
            data_loader = self.valid_loader
        else:
            data_loader = self.test_loader

        prediction = []
        ground_truth = []

        for i, data in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data["feature"], data["target"]

            # do not need to calculate the gradient
            with torch.no_grad():
                if type == "valid":
                    outputs = self.net(inputs)
                else:
                    outputs = self.net_best_validation(inputs)
                prediction_label = torch.max(outputs, 1)[1]
            prediction.append(prediction_label)
            ground_truth.append(labels)

        prediction = torch.cat(prediction, dim=0)
        ground_truth = torch.cat(ground_truth, dim=0)

        accuracy = ((prediction == ground_truth).sum() / ground_truth.shape[0]).item()

        return accuracy


    def stopping_criteria(self, check_continue_epoch=10):
        # You will implement the stopping_criteria
        # you can utilize the self.valid_accuracy
        # ------------------------------------------------------------------------------------------------------------
        # complete your code here
        return False
        # ------------------------------------------------------------------------------------------------------------

    def best_validation_accuracy(self):
        return max(self.valid_accuracy)


    def train(self):
        for epoch in range(self.epoch):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data["feature"], data["target"]

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 10 == 9:  # print every 10 mini-batches
                    self.print_msg(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
                    running_loss = 0.0

            self.print_msg('Finished Training for Epoch {}'.format(epoch))

            # evaluate in validation set
            self.print_msg('Prepare for the evaluation on validation set')
            valid_accuracy = self.evaluation("valid")
            self.valid_accuracy.append(valid_accuracy)
            self.print_msg("The accuracy of validation set on Epoch {} is {}".format(epoch, valid_accuracy))

            # check whether it is the current best validation accuracy, if yes, save the net work
            if valid_accuracy == max(self.valid_accuracy):
                self.net_best_validation = copy.deepcopy(self.net)

            # apply stopping_criteria
            # self.stopping_criteria() returns a bool, if True, we will terminate the training procedure
            if self.stopping_criteria(self.check_continue_epoch):
                self.print_msg("Enter the stopping_criteria, the current number of epoch is {}".format(epoch))
                break
############################################################################################################




if __name__ == "__main__":
    np.random.seed(2022)
    torch.manual_seed(2022)
    trainer = MyPyTorchMLP()
    trainer.train()
    test_accuracy = trainer.evaluation("test")
    print("##################################################")
    print("The accuracy of test set is {}".format(test_accuracy))
    print("##################################################")
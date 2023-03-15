import numpy as np


class Tree_node:
    """
    Data structure for nodes in the decision-tree
    """

    def __init__(
        self,
    ):
        self.is_leaf = False  # whether or not the current node is a leaf node
        self.feature = None  # index of the selected feature (for non-leaf node)
        self.label = None  # class label (for leaf node)
        self.left_child = None  # left child node
        self.right_child = None  # right child node


class Decision_tree:
    """
    Decision tree with binary features
    """

    def __init__(self, min_entropy):
        self.min_entropy = min_entropy
        self.root = None

    def fit(self, train_x, train_y):
        # construct the decision-tree with recursion
        self.root = self.generate_tree(train_x, train_y)

    def predict(self, test_x):
        # iterate through all samples
        prediction = np.zeros([len(test_x),]).astype(
            "int"
        )  # placeholder
        for i in range(len(test_x)):
            # traverse the decision-tree based on the features of the current sample
            pass  # placeholder

        return prediction

    def generate_tree(self, data, label):
        # initialize the current tree node
        cur_node = Tree_node()

        # compute the node entropy
        node_entropy = self.compute_node_entropy(label)

        # determine if the current node is a leaf node
        if node_entropy < self.min_entropy:
            # determine the class label for leaf node

            return cur_node

        # select the feature that will best split the current non-leaf node
        selected_feature = self.select_feature(data, label)
        cur_node.feature = selected_feature

        # split the data based on the selected feature and start the next level of recursion

        return cur_node

    def select_feature(self, data, label):
        # iterate through all features and compute their corresponding entropy
        best_feat = 0

        for i in range(len(data[0])):

            # compute the entropy of splitting based on the selected features
            cur_entropy = self.compute_split_entropy(
                None, None
            )  # You need to replace the placeholders ('None') with valid inputs

            # select the feature with minimum entropy

        return best_feat

    def compute_split_entropy(self, left_y, right_y):
        # compute the entropy of a potential split, left_y and right_y are labels for the two splits
        split_entropy = -1  # placeholder

        return split_entropy

    def compute_node_entropy(self, label):
        # compute the entropy of a tree node (add 1e-15 inside the log2 when computing the entropy to prevent numerical issue)
        node_entropy = -1  # placeholder

        return node_entropy


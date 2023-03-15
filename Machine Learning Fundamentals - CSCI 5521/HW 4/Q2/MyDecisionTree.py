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
          cur_node = self.root
          while True:
              if cur_node.label != None:
                  break
              if test_x[i][cur_node.feature] < 0.5:
                  cur_node = cur_node.left_child
              else:
                  cur_node = cur_node.right_child
          prediction[i] = cur_node.label  # placeholder

        return prediction


    def generate_tree(self, data, label):
        # initialize the current tree node
        cur_node = Tree_node()

        # compute the node entropy
        node_entropy = self.compute_node_entropy(label)

        # determine if the current node is a leaf node
        if node_entropy < self.min_entropy:
            # determine the class label for leaf node
            cur_node.label = np.argmax(np.bincount(label))
            return cur_node
        
        # select the feature that will best split the current non-leaf node
        selected_feature = self.select_feature(data, label)
        cur_node.feature = selected_feature
        
        # split the data based on the selected feature and start the next level of recursion
        cur_node.left_child = self.generate_tree(data[data[:,selected_feature]==0],label[data[:,selected_feature] == 0])
        cur_node.right_child = self.generate_tree(data[data[:,selected_feature]==1],label[data[:,selected_feature] == 1])
        
        return cur_node


    def select_feature(self, data, label):
        # iterate through all features and compute their corresponding entropy
        best_feat = 0
        prev_info_gain = 0
        
        for j in range(data.shape[1]):
          # compute the entropy of splitting based on the selected features
          left_y = []
          right_y = []
          for i in range(data.shape[0]):
            if data[i][j] == 0:
              left_y.append(label[i])
            else:
              right_y.append(label[i])
          left_y_arr = np.array(left_y).astype("int")
          right_y_arr = np.array(right_y).astype("int")

          prob_left = len(left_y_arr) / (label.shape[0])
          prob_right = len(right_y_arr) / (label.shape[0])
        
          cur_entropy = self.compute_split_entropy( left_y_arr, right_y_arr )  # You need to replace the placeholders ('None') with valid inputs
          new_info_gain = self.compute_node_entropy(label)-cur_entropy
          if new_info_gain > prev_info_gain:
            best_feat = j
            prev_info_gain = new_info_gain
        
        return best_feat


    def compute_split_entropy(self, left_y, right_y):
        # compute the entropy of a potential split, left_y and right_y are labels for the two splits
        counts_left_y = np.bincount(left_y)
        counts_right_y = np.bincount(right_y)
        probabilities_left_y = counts_left_y / len(left_y)
        probabilities_right_y = counts_right_y / len(right_y)
        n = len(left_y) + len(right_y)
        split_entropy = 0  # placeholder
        left = 0
        right = 0
        for prob_left in probabilities_left_y:
          if prob_left > 0:
            left += prob_left * np.log2(prob_left+1e-15)
        for prob_right in probabilities_right_y:
          if prob_right > 0:
            right += prob_right * np.log2(prob_right+1e-15)
        split_entropy = len(left_y)/n * left + len(right_y)/n * right
        
        return -split_entropy


    def compute_node_entropy(self, label):
        # compute the entropy of a tree node (add 1e-15 inside the log2 when computing the entropy to prevent numerical issue)
        label_arr = np.array(label)
        counts = np.bincount(label_arr)
        probabilities = counts / len(label_arr)
        
        node_entropy = 0
        for prob in probabilities:
          if prob > 0:
            node_entropy += prob * np.log2(prob + 1e-15)
        
        return -node_entropy
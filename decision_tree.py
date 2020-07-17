import numpy as np
import random


class Node():
    def __init__(self):
        self.feature = None
        self.feature_id = None
        self.value = None
        self.left_leaf = None
        self.right_leaf = None


class DecisionTree():
    def __init__(self):
        self.root = Node()
    
    def fit(self, data, features):
        self.train(self.root, data, features)

    def train(self, node, data, features):
        max_gain = 0
        max_gain_feature = None
        max_gain_value = 0
        max_feature_idx = 0
        root_h = self.cross_entropy(data)
        n = data.shape[0]
        for feature_idx, feature in enumerate(features):
            feature_data = data[:,feature_idx]
            feature_values = np.unique(feature_data)
            for value in feature_values:
                left = data[data[:, feature_idx] < value]
                right = data[data[:, feature_idx] >= value]
                n_l = left.shape[0]
                n_r = right.shape[0]
                left_h = self.cross_entropy(left)
                right_h = self.cross_entropy(right)
                gain = root_h-(n_l/n*left_h+n_r/n*right_h)
                if gain > max_gain:
                    max_gain = gain
                    max_gain_feature = feature
                    max_gain_value = value
                    max_feature_idx = feature_idx
        node.value = max_gain_value
        node.feature = max_gain_feature
        node.feature_id = max_feature_idx
        left = data[data[:, max_feature_idx] < max_gain_value]
        right = data[data[:, max_feature_idx] >= max_gain_value]
        node.left = Node()
        node.right = Node()
        if len(np.unique(left[:,-1]))>1:
            self.train(node.left, left, features)
        elif np.unique(left[:,-1]).size > 0:
            node.left.value = left[0,-1]
        if len(np.unique(right[:,-1]))>1:
            self.train(node.right, right, features)
        elif np.unique(right[:,-1]).size > 0:
            node.right.value = right[0,-1]
            
    def cross_entropy(self, data):
        y = data[:,-1]
        y_value = np.unique(y)
        count = 0
        for i in y_value:
            p = sum(y==i)/len(y)
            count -= p*np.log(p)
        return count

    def predict(self, data):
        labels = []
        for d in data:
            node = self.root
            while True:
                if node.feature is None:
                    labels.append(node.value)
                    break
                else:
                    if d[node.feature_id] < node.value:
                        node = node.left
                    else:
                        node = node.right
        return labels


def confussion(output,target):
    class_o = np.unique(output)
    class_t = np.unique(target)
    num_class = len(np.unique(class_o+class_t))
    matrix = np.zeros((num_class,num_class))
    for p,q in zip(output, target):
        matrix[int(p)-1, int(q)-1] += 1
    return matrix

data = np.loadtxt('clean_dataset.txt')
random.shuffle(data)
n_train = int(0.8*data.shape[0])
train_data = data[:n_train, :]
test_data = data[n_train:, :]


features = np.arange(data.shape[1]-1)

tree = DecisionTree()
tree.fit(train_data, features)
pred = tree.predict(test_data)

print(confussion(pred, test_data[:,-1]))


    

import numpy as np
import matplotlib.pyplot as plt

class Node(object):
    def __init__(self, criterion="gini", max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.random_state = random_state
        self.depth = None
        self.left = None
        self.right = None
        self.feature = None
        self.threshold = None
        self.label = None
        self.impurity = None
        self.info_gain = None
        self.num_sample = None
        self.num_classes = None

    def split_node(self, sample, target, depth, ini_num_classes):
        self.depth = depth

        self.num_sample = len(target)
        self.num_classes = [len(target[target==i]) for i in ini_num_classes]

        if len(np.unique(target)) == 1:
            self.label = target[0]
            self.impurity = self.criterion_func(target)
            return

        class_count = {i: len(target[target==i]) for i in np.unique(target)}
        self.label = max(class_count.items(), key=lambda x:x[1])[0]
        self.impurity = self.criterion_func(target)

        num_features = sample.shape[1]
        salf.info_gain = 0.0

        if self.random_state != None:
            np.random.seed(self.random_state)
        f_loop_order = np.random.permutation(num_features).tolist()
        for f in f_loop_order:
            uniq_feature = np.unique(sample[:, f])
            split_points = (uniq_feature[:-1] + uniq_feature[1:]) /2.0

            for threshold in split_points:
                target_l = target[sample[:, f] <= threshold]
                target_r = target[sample[:, f] >  threshold]
                val = self.calc_info_gain(target, target_l, target_r)

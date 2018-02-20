from __future__ import division

import numpy as np
from collections import Counter
import time


class DecisionNode:
    """Class to represent a single node in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """Create a decision function to select between left and right nodes.

        Note: In this representation 'True' values for a decision take us to
        the left. This is arbitrary but is important for this assignment.

        Args:
            left (DecisionNode): left child node.
            right (DecisionNode): right child node.
            decision_function (func): function to decide left or right node.
            class_label (int): label for leaf node. Default is None.
        """

        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Get a child node based on the decision function.

        Args:
            feature (list(int)): vector for feature.

        Return:
            Class label if a leaf node, otherwise a child node.
        """

        if self.class_label is not None:
            return self.class_label

        elif self.decision_function(feature):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)

def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.

    Args:
        data_file_path (str): path to data file.
        class_index (int): slice output by index.

    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as nump array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if class_index == -1:
        classes = map(int, out[:, class_index])
        features = out[:, :class_index]
        return features, classes

    elif class_index == 0:
        classes = map(int, out[:, class_index])
        features = out[:, 1:]
        return features, classes

    else:
        return out


def build_decision_tree():
    """Create a decision tree capable of handling the provided data.

    Tree is built fully starting from the root.

    Returns:
        The root node of the decision tree.
    """
    decision_tree_root = DecisionNode(None,None,lambda feature: feature[0] == 0)
    decision_tree_node_a4 = DecisionNode(None,None,lambda feature: feature[3] == 0)
    decision_tree_node_a3 = DecisionNode(None,None,lambda feature: feature[2] == 0)
    decision_tree_node_a2 = DecisionNode(None,None,lambda feature: feature[1] == 0)

    decision_tree_root.right = DecisionNode(None,None,None,1)
    decision_tree_root.left = decision_tree_node_a4
    decision_tree_node_a4.left = decision_tree_node_a3
    decision_tree_node_a4.right = decision_tree_node_a2
    decision_tree_node_a2.left = DecisionNode(None,None,None,1)
    decision_tree_node_a2.right = DecisionNode(None,None,None,0)
    decision_tree_node_a3.left = DecisionNode(None,None,None,1)
    decision_tree_node_a3.right = DecisionNode(None,None,None,0)

    return decision_tree_root

def confusion_matrix(classifier_output, true_labels):
    """Create a confusion matrix to measure classifier performance.

    Output will in the format:
        [[true_positive, false_negative],
         [false_positive, true_negative]]

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        A two dimensional array representing the confusion matrix.
    """
    confusion_mat = np.zeros((2,2))
    confusion_mat = confusion_mat.astype(int)

    for i in range(0,len(classifier_output)):
	if true_labels[i] == classifier_output[i]:
		if true_labels[i] == 1:
			# True Positive
			confusion_mat[0,0] += 1
		else:
			# True Negative
			confusion_mat[1,1] += 1
	else:
		if true_labels[i] == 1:
			# False Negative
			confusion_mat[0,1] += 1
		else:
			# False Positive
			confusion_mat[1,0] += 1

    return confusion_mat

def precision(classifier_output, true_labels):
    """Get the precision of a classifier compared to the correct values.

    Precision is measured as:
        true_positive/ (true_positive + false_positive)

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        The precision of the classifier output.
    """
    conf_mat = confusion_matrix(classifier_output, true_labels)    
    precision = conf_mat[0,0] / (conf_mat[0,0] + conf_mat[1,0])
    return precision

def recall(classifier_output, true_labels):
    """Get the recall of a classifier compared to the correct values.

    Recall is measured as:
        true_positive/ (true_positive + false_negative)

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        The recall of the classifier output.
    """
    conf_mat = confusion_matrix(classifier_output, true_labels)    
    recall = conf_mat[0,0] / (conf_mat[0,0] + conf_mat[0,1])
    return recall
    
def accuracy(classifier_output, true_labels):
    """Get the accuracy of a classifier compared to the correct values.

    Accuracy is measured as:
        correct_classifications / total_number_examples

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        The accuracy of the classifier output.
    """
    conf_mat = confusion_matrix(classifier_output, true_labels)    
    div = conf_mat[0,0] + conf_mat[0,1] + conf_mat[1,0] + conf_mat[1,1]
    accuracy = (conf_mat[0,0] + conf_mat[1,1]) / div
    return accuracy

def gini_impurity(class_vector):
    """Compute the gini impurity for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.

    Args:
        class_vector (list(int)): Vector of classes given as 0 or 1.

    Returns:
        Floating point number representing the gini impurity.
    """
    gini_impurity = 0.0
    uniq_classes = np.unique(class_vector)
    class_occ = np.bincount(class_vector)
    for i in range(0,len(uniq_classes)):
	class_prob = class_occ[uniq_classes[i]] / (len(class_vector) * 1.0)
	gini_impurity += (class_prob ** 2)
    gini_impurity = 1 - gini_impurity
    return gini_impurity

def gini_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0 or 1.
        current_classes (list(list(int): A list of lists where each list has
            0 and 1 values).
    Returns:
        Floating point number representing the information gain.
    """
    sizes = []
    I_C = []
    total_len = 0
    # Before split compute the Gini Impurity for parent
    I_A = gini_impurity(previous_classes)
    # After split compute the Gini Imputiy for all children
    nos_of_child = len(current_classes)
    gain_sum = 0.0
    for i in range(0, nos_of_child):
	I_C.append(gini_impurity(current_classes[i]))
	sizes.append(len(current_classes[i]))
        total_len += sizes[i]

    gain_sum = 0.0
    for i in range(0, nos_of_child):
	gain_sum += ((sizes[i]/total_len) * I_C[i])

    gini_gain = I_A - gain_sum

    return gini_gain

class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=float('inf')):
        """Create a decision tree with a set depth limit.

        Starts with an empty root.

        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__().

        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
        """

        self.root = self.__build_tree__(features, classes)

    def build_decision_tree(self, data, max_depth, depth):
	# No more rows left in data
	if data.shape[0] == 0:
		return DecisionNode(None,None,None,None)
	# All classes are same, so return the class
	if np.allclose(data[0,-1],data[:,-1]) is True:
		return DecisionNode(None,None,None,int(data[0,-1]))
	# Max depth reached
	if depth == max_depth:
		(vals,cnts) = np.unique(data[:,-1],return_counts=True)
		i = np.argmax(cnts)
		return DecisionNode(None,None,None,int(vals[i]))
	else:
		attr_best = None
		max_gini_gain = 0.0
		attr_gini_gain = []
		# Choose attribute based on Gini Index
		prev_class = data[:,-1]
		prev_class = prev_class.tolist()
		prev_class = map(int,prev_class)
		# (len(data[0,:])-1) will ensure the appended class label is not accesed
		for i in range(0,len(data[0,:])-1):
			# Get the attribute column and its label
			attr = data[:,[i,-1]]
			attr_split = np.mean(attr[:,0])
			left = attr[attr[:,0] <= attr_split]
			left = left[:,-1].tolist()
			left = map(int,left)
			right = attr[attr[:,0] > attr_split]
			right = right[:,-1].tolist()
			right = map(int,right)
			curr_class = [left,right]
			attr_gini_gain.append(gini_gain(prev_class,curr_class))

		# Find the attribute with max Gini Gain
		best_attr = attr_gini_gain.index(max(attr_gini_gain))
		feature = data[:,0:-1]
		best_split = np.mean(feature[:,best_attr])

		root = DecisionNode(None,None,lambda feature: feature[best_attr] <= best_split)
		root.left = self.build_decision_tree(data[data[:,best_attr] <= best_split], max_depth, depth + 1)
		root.right = self.build_decision_tree(data[data[:,best_attr] > best_split], max_depth, depth + 1)
	return root	

    def __build_tree__(self, features, classes, depth=0):
        """Build tree that automatically finds the decision functions.

        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
            depth (int): max depth of tree.  Default is 0.

        Returns:
            Root node of decision tree.
        """
        data = np.column_stack((features,classes))
	return self.build_decision_tree(data,self.depth_limit,0)

    def classify(self, features):
        """Use the fitted tree to classify a list of example features.

        Args:
            features (list(list(int)): List of features.

        Return:
            A list of class labels.
        """

        class_labels = []

	for feature in features:
		node = self.root.decide(feature)
		if node != None:
			class_labels.append(node)

        return class_labels

def generate_k_folds(dataset, k):
    """Split dataset into folds.

    Randomly split data into k equal subsets.

    Fold is a tuple (training_set, test_set).
    Set is a tuple (examples, classes).

    Args:
        dataset: dataset to be split.
        k (int): number of subsections to create.

    Returns:
        List of folds.
    """
    # Combine the features and classes. The last column of data will be classes
    rounds = 10
    data = np.column_stack((dataset[0],dataset[1]))
    n = len(data)
    num_samples = int(n/k)
    k_folds = []
    for i in range(0,k):
	dup_data = np.copy(data)
	# Just shuffle the data 
	np.random.shuffle(dup_data)	
	indices = np.random.choice(range(0,n),num_samples,replace=False)
	testing = dup_data[indices,:]
	training = np.delete(dup_data,indices,0)
	test_set = (testing[:,0:-1].tolist(),testing[:,-1].tolist())
	train_set = (training[:,0:-1].tolist(),training[:,-1].tolist())
	# create a tuple of training and testing dataset
	dataset = (train_set,test_set)
	k_folds.append(dataset)
	
    return k_folds

class RandomForest:
    """Random forest classification."""

    def __init__(self, num_trees, depth_limit, example_subsample_rate,
                 attr_subsample_rate):
        """Create a random forest.

         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """
        self.trees = []
	self.trees_attr = []  		# List of tuples containing the attributes on which the tree was trained on
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.

            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
        """
    	dataset = np.column_stack((features,classes))
	n = len(dataset)
	num_samples = int(self.example_subsample_rate * n)
	num_attr = len(features[0])
	num_attr_samples = int(self.attr_subsample_rate * num_attr)
	for i in range(0,self.num_trees):
		dup_data = np.copy(dataset)
		np.random.shuffle(dup_data)
		# Get random data samples with replacement
		data_ind = np.random.choice(range(0,n),num_samples,replace=True)
		# Get random attributes without replacement
		attr_ind = np.random.choice(range(0,num_attr),num_attr_samples,replace=False)
		attr_ind = np.sort(attr_ind)
		self.trees_attr.append(attr_ind)
		# Sorting array to access the features in the same order and appending "-1" to include the classes 
		attr_ind = np.append(attr_ind,-1)
		# First get the random data samples for all the features using the data subsample rate
		train_set = dup_data[data_ind,:]
		# Second get the data samples of only the randomly chosen features
		train_set = train_set[:,attr_ind]
		tree = DecisionTree(self.depth_limit)
		tree.fit(train_set[:,0:-1].tolist(),train_set[:,-1].tolist())
		self.trees.append(tree)

    def classify(self, features):
        """Classify a list of features based on the trained random forest.

        Args:
            features (list(list(int)): List of features.
        """
	dup_features = np.array(features)
	all_labels = []
	for i in range(0,self.num_trees):
		tree_features = dup_features[:,self.trees_attr[i]]
		tree = self.trees[i]
		class_labels = tree.classify(tree_features.tolist())
		all_labels.append(class_labels)

	all_labels = np.array(all_labels)
	voted_labels = []
	for i in range(0,len(features)):
		class_vectors = all_labels[:,i]
		(vals,cnts) = np.unique(class_vectors,return_counts=True)
		voted_labels.append(vals[np.argmax(cnts)])

	return voted_labels

class ChallengeClassifier:
    """Challenge Classifier used on Challenge Training Data."""

    def __init__(self):
        """Create challenge classifier.

        Initialize whatever parameters you may need here.
        This method will be called without parameters, therefore provide
        defaults.
        """
        self.trees = []
	self.trees_attr = []  		# List of tuples containing the attributes on which the tree was trained on
        self.num_trees = 10
	self.train_error = np.array([])		# To maintain the train error data for boosting
        self.depth_limit = 8
        self.example_subsample_rate = 0.6
        self.attr_subsample_rate = 0.8

    def boosting(self, dataset, i):
	dup_data = np.copy(dataset)
	n = len(dataset)
	num_samples = self.example_subsample_rate * n
	train_data_error = self.train_error
	nos_rem_trees = self.num_trees - i
	if len(train_data_error) == 0:
		# There are no train errors thus take data from example_subsample_rate
		# Get random data samples with replacement for dataset
		data_ind = np.random.choice(range(0,n),int(num_samples),replace=True)
		# Get the random data samples for all the features using the data subsample rate
		train_set = dup_data[data_ind,:]
	else:
		# If there are training error data then take len(train_error)/(Nos of remaining trees)% 
		# from train error and 80% of example_subsample_rate from dataset
		num_samples = int(num_samples * 0.8)
		# Get random data samples with replacement for dataset
		data_ind = np.random.choice(range(0,n),num_samples,replace=True)
		# Get the random data samples for all the features using the data subsample rate
		train_set = dup_data[data_ind,:]
		# Get the training error data sample rate
		train_sample_rate = (len(train_data_error)/nos_rem_trees) / 100 
		num_train_samples = int(len(train_data_error) * train_sample_rate)
		# Get random data samples with replacement for train error
		train_ind = np.random.choice(range(0,len(train_data_error)),num_train_samples,replace=True)
		train_error_set = train_data_error[train_ind,:]
		# Now append the train_set and train_error_set
		train_set = np.append(train_set,train_error_set,axis=0)
		# Delete the training error data that was included for this tree
		self.train_error = np.delete(train_data_error,train_ind,axis=0)

	return train_set	
 
    def chkForTrainErrors(self, dataset, nos_trained_trees):
	if nos_trained_trees == self.num_trees:
		return None

	dup_dataset = np.copy(dataset)
	features = dataset[:,0:-1]
	actual_labels = dataset[:,-1]
	all_labels = []
	for i in range(0,nos_trained_trees):
		tree_features = features[:,self.trees_attr[i]]
		tree = self.trees[i]
		class_labels = tree.classify(tree_features.tolist())
		all_labels.append(class_labels)

	all_labels = np.array(all_labels)
	predicted_labels = []
	for i in range(0,len(features)):
		class_vectors = all_labels[:,i]
		(vals,cnts) = np.unique(class_vectors,return_counts=True)
		predicted_labels.append(vals[np.argmax(cnts)])
	
	predicted_labels = np.array(predicted_labels)
	label_errors = (actual_labels == predicted_labels)
	matched_indices = np.where(label_errors)
	train_error_data = np.delete(dup_dataset,matched_indices,axis=0)
	
	if len(self.train_error) == 0:
		# Train error is empty
		self.train_error = train_error_data 
	else:	
		self.train_error = np.append(self.train_error,train_error_data,axis=0)

    def fit(self, features, classes):
        """Build the underlying tree(s).

            Fit your model to the provided features.

        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
        """
    	dataset = np.column_stack((features,classes))
	num_attr = len(features[0])
	num_attr_samples = int(self.attr_subsample_rate * num_attr)
	for i in range(0,self.num_trees):
		# Get training data weighted on previous training errors
		train_set = self.boosting(dataset, i)
		# Get random attributes without replacement
		attr_ind = np.random.choice(range(0,num_attr),num_attr_samples,replace=False)
		attr_ind = np.sort(attr_ind)
		self.trees_attr.append(attr_ind)
		# Sorting array to access the features in the same order and appending "-1" to include the classes 
		attr_ind = np.append(attr_ind,-1)
		# Get the data samples of only the randomly chosen features
		train_set = train_set[:,attr_ind]
		tree = DecisionTree(self.depth_limit)
		tree.fit(train_set[:,0:-1].tolist(),train_set[:,-1].tolist())
		self.trees.append(tree)
		self.chkForTrainErrors(dataset,i+1)

    def classify(self, features):
        """Classify a list of features.

        Classify each feature in features as either 0 or 1.

        Args:
            features (list(list(int)): List of features.

        Returns:
            A list of class labels.
        """
	dup_features = np.array(features)
	all_labels = []
	for i in range(0,self.num_trees):
		tree_features = dup_features[:,self.trees_attr[i]]
		tree = self.trees[i]
		class_labels = tree.classify(tree_features.tolist())
		all_labels.append(class_labels)

	all_labels = np.array(all_labels)
	voted_labels = []
	for i in range(0,len(features)):
		class_vectors = all_labels[:,i]
		(vals,cnts) = np.unique(class_vectors,return_counts=True)
		voted_labels.append(vals[np.argmax(cnts)])

	return voted_labels

class Vectorization:
    """Vectorization preparation for Assignment 5."""

    def __init__(self):
        pass

    def non_vectorized_loops(self, data):
        """Element wise array arithmetic with loops.

        This function takes one matrix, multiplies by itself and then adds to
        itself.

        Args:
            data: data to be added to array.

        Returns:
            Numpy array of data.
        """

        non_vectorized = np.zeros(data.shape)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row][col] = (data[row][col] * data[row][col] +
                                            data[row][col])
        return non_vectorized

    def vectorized_loops(self, data):
        """Element wise array arithmetic using vectorization.

        This function takes one matrix, multiplies by itself and then adds to
        itself.

        Bonnie time to beat: 0.09 seconds.

        Args:
            data: data to be sliced and summed.

        Returns:
            Numpy array of data.
        """
	vect_data = np.array(data)
	vect_prod = vect_data * vect_data
	vect_sum = vect_prod + vect_data

	return vect_sum

    def non_vectorized_slice(self, data):
        """Find row with max sum using loops.

        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).

        Args:
            data: data to be added to array.

        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        max_sum = 0
        max_sum_index = 0
        for row in range(100):
            temp_sum = 0
            for col in range(data.shape[1]):
                temp_sum += data[row][col]

            if temp_sum > max_sum:
                max_sum = temp_sum
                max_sum_index = row

        return max_sum, max_sum_index

    def vectorized_slice(self, data):
        """Find row with max sum using vectorization.

        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).

        Bonnie time to beat: 0.07 seconds

        Args:
            data: data to be sliced and summed.

        Returns:
            Tuple (Max row sum, index of row with max sum)
        """
	vect_data = np.array(data)
	vect_data = vect_data[0:100,:]
	row_sum = np.sum(vect_data,axis=1)
	max_sum_index = np.argmax(row_sum)
	max_sum = row_sum[max_sum_index]
	
	return max_sum, max_sum_index

    def non_vectorized_flatten(self, data):
        """Display occurrences of positive numbers using loops.

         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.

         ie, [(1203,3)] = integer 1203 appeared 3 times in data.

         Args:
            data: data to be added to array.

        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        unique_dict = {}
        flattened = np.hstack(data)
        for item in range(len(flattened)):
            if flattened[item] > 0:
                if flattened[item] in unique_dict:
                    unique_dict[flattened[item]] += 1
                else:
                    unique_dict[flattened[item]] = 1

        return unique_dict.items()

    def vectorized_flatten(self, data):
        """Display occurrences of positive numbers using vectorization.

         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.

         ie, [(1203,3)] = integer 1203 appeared 3 times in data.

         Bonnie time to beat: 15 seconds

         Args:
            data: data to be added to array.

        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """
	uniq_vals = []
        vect_flattened = np.hstack(data)
	positive_nums = vect_flattened[vect_flattened > 0.0]
	(vals,counts) = np.unique(positive_nums,return_counts=True)
	for i in range(0,len(vals)):
		uniq_vals.append((vals[i],counts[i]))

	return uniq_vals
        
def return_your_name():
    # return your name
    return "rmurali7"

from __future__ import division

import unittest
import decision_trees_submission as dt
import numpy as np

import platform
if platform.system() != 'Windows':
    import resource


class DecisionTreePart1Tests(unittest.TestCase):
    """Test tree example, confusion matrix, precision, recall, and accuracy.

    Attributes:
        hand_tree (DecisionTreeNode): root node of the built example tree.
        ht_examples (list(list(int)): features for example tree.
        ht_classes (list(int)): classes for example tree."""

    def setUp(self):
        """Setup test data.
        """

        self.hand_tree = dt.build_decision_tree()
        self.ht_examples = [[1, 0, 0, 0],
                            [1, 0, 1, 1],
                            [0, 1, 0, 0],
                            [0, 1, 1, 0],
                            [1, 1, 0, 1],
                            [0, 1, 0, 1],
                            [0, 0, 1, 1],
                            [0, 0, 1, 0]]
        self.ht_classes = [1, 1, 1, 0, 1, 0, 1, 0]

    def test_hand_tree_accuracy(self):
        """Test accuracy of the tree example.

        Asserts:
            decide return matches true class for all classes.
        """

        for index in range(0, len(self.ht_examples)):
            decision = self.hand_tree.decide(self.ht_examples[index])

            assert decision == self.ht_classes[index]

    def test_confusion_matrix(self):
        """Test confusion matrix for the example tree.

        Asserts:
            confusion matrix is correct.
        """

        answer = [1, 0, 0, 1, 0, 0, 0]
        true_label = [1, 1, 1, 0, 0, 0, 0]
        test_matrix = [[1, 2], [1, 3]]

        assert np.array_equal(test_matrix, dt.confusion_matrix(answer,
                                                               true_label))

    def test_precision_calculation(self):
        """Test precision calculation.

        Asserts:
            Precision matches for all true labels.
        """

        answer = [0, 0, 0, 0, 0]
        true_label = [1, 0, 0, 0, 0]

        for index in range(0, len(answer)):
            answer[index] = 1
            precision = 1 / (1 + index)

            assert dt.precision(answer, true_label) == precision

    def test_recall_calculation(self):
        """Test recall calculation.

        Asserts:
            Recall matches for all true labels.
        """

        answer = [0, 0, 0, 0, 0]
        true_label = [1, 1, 1, 1, 1]
        total_count = len(answer)

        for index in range(0, len(answer)):
            answer[index] = 1
            recall = (index + 1) / ((index + 1) + (total_count - (index + 1)))

            assert dt.recall(answer, true_label) == recall

    def test_accuracy_calculation(self):
        """Test accuracy calculation.

        Asserts:
            Accuracy matches for all true labels.
        """

        answer = [0, 0, 0, 0, 0]
        true_label = [1, 1, 1, 1, 1]
        total_count = len(answer)

        for index in range(0, len(answer)):
            answer[index] = 1
            accuracy = dt.accuracy(answer, true_label)

            assert accuracy == ((index + 1) / total_count)


class DecisionTreePart2Tests(unittest.TestCase):
    """Tests for Decision Tree Learning.

    Attributes:
        restaurant (dict): represents restaurant data set.
        dataset (data): training data used in testing.
        train_features: training features from dataset.
        train_classes: training classes from dataset.
    """

    def setUp(self):
        """Set up test data.
        """

        self.restaurant = {'restaurants': [0] * 6 + [1] * 6,
                           'split_patrons': [[0, 0],
                                             [1, 1, 1, 1],
                                             [1, 1, 0, 0, 0, 0]],
                           'split_food_type': [[0, 1],
                                               [0, 1],
                                               [0, 0, 1, 1],
                                               [0, 0, 1, 1]]}

        self.dataset = dt.load_csv('part23_data.csv')
        self.train_features, self.train_classes = self.dataset

    def test_gini_impurity_max(self):
        """Test maximum gini impurity.

        Asserts:
            gini impurity is 0.5.
        """

        gini_impurity = dt.gini_impurity([1, 1, 1, 0, 0, 0])

        assert  .500 == round(gini_impurity, 3)

    def test_gini_impurity_min(self):
        """Test minimum gini impurity.

        Asserts:
            entropy is 0.
        """

        gini_impurity = dt.gini_impurity([1, 1, 1, 1, 1, 1])

        assert 0 == round(gini_impurity, 3)

    def test_gini_impurity(self):
        """Test gini impurity.

        Asserts:
            gini impurity is matched as expected.
        """

        gini_impurity = dt.gini_impurity([1, 1, 0, 0, 0, 0])

        assert round(4. / 9., 3) == round(gini_impurity, 3)

    def test_gini_gain_max(self):
        """Test maximum gini gain.

        Asserts:
            gini gain is 0.5.
        """

        gini_gain = dt.gini_gain([1, 1, 1, 0, 0, 0],
                                 [[1, 1, 1], [0, 0, 0]])

        assert .500 == round(gini_gain, 3)

    def test_gini_gain(self):
        """Test gini gain.

        Asserts:
            gini gain is within acceptable bounds
        """

        gini_gain = dt.gini_gain([1, 1, 1, 0, 0, 0],
                                 [[1, 1, 0], [1, 0, 0]])
        
        assert 0.056 == round(gini_gain, 3)

    def test_gini_gain_restaurant_patrons(self):
        """Test gini gain using restaurant patrons.

        Asserts:
            gini gain rounded to 3 decimal places matches as expected.
        """

        gain_patrons = dt.gini_gain(
            self.restaurant['restaurants'],
            self.restaurant['split_patrons'])

        assert round(gain_patrons, 3) == 0.278

    def test_gini_gain_restaurant_type(self):
        """Test gini gain using restaurant food type.

        Asserts:
            gini gain is 0.
        """

        gain_type = round(dt.gini_gain(
            self.restaurant['restaurants'],
            self.restaurant['split_food_type']), 2)

        assert gain_type == 0.00

    def test_decision_tree_all_data(self):
        """Test decision tree classifies all data correctly.

        Asserts:
            classification is 100% correct.
        """

        #tree = dt.DecisionTree()
        #tree = dt.RandomForest(5,5,0.5,0.5)
        tree = dt.ChallengeClassifier()
        tree.fit(self.train_features, self.train_classes)
        output = tree.classify(self.train_features)
	print(len([i for i,j in zip(output, self.train_classes) if i == j])/ len(self.train_classes))

        assert cmp(output, self.train_classes) == 0

    def test_k_folds_test_set_count(self):
        """Test k folds returns the correct test set size.

        Asserts:
            test set size matches as expected.
        """

        example_count = len(self.train_features)
        k = 10
        test_set_count = example_count // k
        ten_folds = dt.generate_k_folds(self.dataset, k)

        for fold in ten_folds:
            training_set, test_set = fold

            assert len(test_set[0]) == test_set_count

    def test_k_folds_training_set_count(self):
        """Test k folds returns the correct training set size.

        Asserts:
            training set size matches as expected.
        """

        example_count = len(self.train_features)
        k = 10
        training_set_count = example_count - (example_count // k)
        ten_folds = dt.generate_k_folds(self.dataset, k)

        for fold in ten_folds:
            training_set, test_set = fold

            assert len(training_set[0]) == training_set_count


class VectorizationWarmUpTests(unittest.TestCase):
    """Tests the Warm Up exercises for Vectorization.

    Attributes:
        vector (Vectorization): provides vectorization test functions.
        data: vectorize test data.
    """

    def setUp(self):
        """Set up test data.
        """

        self.vector = dt.Vectorization()
        self.data = dt.load_csv('vectorize.csv', 1)

    def test_vectorized_loops(self):
        """Test if vectorized arithmetic.

        Asserts:
            vectorized arithmetic matches looped version.
        """

        real_answer = self.vector.non_vectorized_loops(self.data)
        my_answer = self.vector.vectorized_loops(self.data)

        assert np.array_equal(real_answer, my_answer)

    def test_vectorized_slice(self):
        """Test if vectorized slicing.

        Asserts:
            vectorized slicing matches looped version.
        """

        real_sum, real_sum_index = self.vector.non_vectorized_slice(self.data)
        my_sum, my_sum_index = self.vector.vectorized_slice(self.data)

        assert real_sum == my_sum
        assert real_sum_index == my_sum_index

    def test_vectorized_flatten(self):
        """Test if vectorized flattening.

        Asserts:
            vectorized flattening matches looped version.
        """

        answer_unique = self.vector.non_vectorized_flatten(self.data)
        my_unique = self.vector.vectorized_flatten(self.data)

        assert np.array_equal(answer_unique, my_unique)

    @unittest.skipUnless(platform.system() != 'Windows', "Requires Unix")
    def test_vectorized_loops_time(self):
        """Test if vectorized arithmetic speed.

        Asserts:
            vectorized arithmetic is faster than expected Bonnie time.
        """

        start_time = resource.getrusage(resource.RUSAGE_SELF).ru_utime * 1000
        self.vector.vectorized_loops(self.data)
        end_time = resource.getrusage(resource.RUSAGE_SELF).ru_utime * 1000

        assert (end_time - start_time) <= 0.09

    @unittest.skipUnless(platform.system() != 'Windows', "Requires Unix")
    def test_vectorized_slice_time(self):
        """Test if vectorized slicing speed.

        Asserts:
            vectorized slicing is faster than expected Bonnie time.
        """

        start_time = resource.getrusage(resource.RUSAGE_SELF).ru_utime * 1000
        self.vector.vectorized_slice(self.data)
        end_time = resource.getrusage(resource.RUSAGE_SELF).ru_utime * 1000

        assert (end_time - start_time) <= 0.07

    @unittest.skipUnless(platform.system() != 'Windows', "Requires Unix")
    def test_vectorized_flatten_time(self):
        """Test if vectorized flatten speed.

        Asserts:
            vectorized flatten is faster than expected Bonnie time.
        """
        start_time = resource.getrusage(resource.RUSAGE_SELF).ru_utime * 1000
        self.vector.vectorized_flatten(self.data)
        end_time = resource.getrusage(resource.RUSAGE_SELF).ru_utime * 1000

        assert (end_time - start_time) <= 15.0

if __name__ == '__main__':
    unittest.main()

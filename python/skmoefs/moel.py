from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

class MOELScheme(ABC, BaseEstimator):
    """
    A MOELScheme represents a general multi-objective evolutionary learning
    scheme for generating a set of FRBSs characterized by different trade-offs
    between accuracy and explainability.
    """

    @abstractmethod
    def fit(self, x, y):
        """
        Estimates the model parameters, namely the RB and the fuzzy partitions,
        exploiting the provided training set.
        """
        pass

    @abstractmethod
    def cross_val_score(self, X, y, num_fold):
        """
        Measures the performance of the model using K-Fold cross validation.
        """
        pass

    @abstractmethod
    def show_pareto(self):
        """
        Extracts and plots the values of accuracy and explainability. Returns a
        plot of the approximated Pareto front, both on the training and the test
        sets.
        """
        pass

    @abstractmethod
    def show_model(self, position):
        """
        Given the position of a model in the Pareto front, this method shows the
        set of fuzzy linguistic rules and the fuzzy partitions associated with
        each linguistic attribute. The predefined model of choice is, as always,
        the FIRST solution.
        """
        pass

    @abstractmethod
    def __getitem__(self, position):
        """
        Returns the model for which the position in the Pareto front is given.
        """
        pass


class MOEL_FRBC(MOELScheme, ClassifierMixin):
    """
    MOEL scheme for Fuzzy Rule-based Classifiers (FRBCs).
    """

    def __init__(self, classifiers=None):
        if classifiers is None:
            classifiers = []
        self.classifiers = classifiers

    def predict(self, X, position='first'):
        """
        In charge of predicting the class labels associated with a new set of
        input patterns.
        """
        n_classifiers = len(self.classifiers)
        if n_classifiers == 0:
            return None
        else:
            index = {'first': 0, 'median': int(n_classifiers/2), 'last': -1}
            return self.classifiers[index[position.lower()]].predict(X)

    def score(self, X, y, sample_weight=None):
        """
        Generates the values of the accuracy and explainability measures for the
        selected model.
        """
        n_classifiers = len(self.classifiers)
        if n_classifiers > 1:
            indexes = {0, int(n_classifiers / 2), n_classifiers - 1}
            for index in indexes:
                clf = self.classifiers[index]
                accuracy = clf.accuracy()
                complexity = clf.trl()
                print(accuracy, complexity)


class MOEL_FRBR(MOELScheme, RegressorMixin):
    """
    MOEL scheme for Fuzzy Rule-based Regressors (FRBRs).
    """

    def __init__(self, regressors=None):
        if regressors is None:
            regressors = []
        self.regressors = regressors

    def predict(self, X, position='first'):
        """
        In charge of predicting the values associated with a new set of input
        patterns.
        """
        n_regressors = len(self.regressors)
        if n_regressors == 0:
            return None
        elif n_regressors == 1:
            return self.regressors[0].predict()
        else:
            index = {'first': 0, 'median': int(n_regressors / 2), 'last': n_regressors - 1}
            return self.regressors[index[position]].predict(X)

    def score(self, X, y, sample_weight=None):
        """
        Generates the values of the accuracy and explainability measures for the
        selected model.
        """
        n_regressors = len(self.regressors)
        if n_regressors > 1:
            indexes = {0, int(n_regressors / 2), n_regressors - 1}
            for index in indexes:
                rgr = self.regressors[index]
                accuracy = rgr.accuracy()
                complexity = rgr.trl()
                print(accuracy, complexity)
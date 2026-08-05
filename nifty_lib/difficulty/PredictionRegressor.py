import numpy

from nifty_lib.utils.classifier_utils import get_classifier_name


class PredictionRegressor:
    """
    Class for building objects able to predict difficulty according to specific rules
    """

    def __init__(self):
        """
        Constructor
        """
        self.name = ""

    def fit(self, x_train, y_train):
        """
        Trains (if required) the PredictionRegressor
        :param x_train: the input data
        :param y_train: the label, or computed difficulty theta_hat
        :return: the function to predict theta_tilde
        """
        return self

    def predict(self, x_test) -> numpy.ndarray:
        """
        Predicts difficulty for a novel data point
        :param x_test: the data points to predict difficulty of
        :return: an array containing difficulty predictions as [0; 1] float numbers
        """
        return None

    def get_name(self) -> str:
        """
        Returns a string describing the prediction regressor
        :return: a string
        """
        return self.name


class MLRegressor(PredictionRegressor):
    """
    Class for building objects able to predict difficulty using a ML algorithm (theta_ml)
    """

    def __init__(self, ml_regressor):
        """
        Constructor
        """
        super().__init__()
        self.ml_regressor = ml_regressor
        self.name = "ML(" + get_classifier_name(ml_regressor) + ")"

    def fit(self, x_train, y_train):
        """
        Trains (if required) the PredictionRegressor
        :param x_train: the input data
        :param y_train: the label, or computed difficulty theta_hat
        :return: the function to predict theta_tilde
        """
        self.ml_regressor.fit(x_train, y_train)
        return self

    def predict(self, x_test) -> numpy.ndarray:
        """
        Predicts difficulty for a novel data point
        :param x_test: the data points to predict difficulty of
        :return: an array containing difficulty predictions as [0; 1] float numbers
        """
        return numpy.asarray(self.ml_regressor.predict(x_test))
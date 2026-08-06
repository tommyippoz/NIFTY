import numpy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from nifty_lib.difficulty.PredictionRegressor import PredictionRegressor, MLRegressor
from nifty_lib.difficulty.PredictionStrategy import PredictionStrategy
from nifty_lib.utils.general_utils import current_ms


def extract_predictor(pred_strategy, params) -> PredictionRegressor:
    """
    Returns a regression object that can be used to predict difficulty for a given input data
    :param pred_strategy:
    :param params:
    :return:
    """
    return None


class DifficultyPredictor:
    """
    Class for building objects able to predict difficulty according to specific rules
    """

    def __init__(self, verbose: bool = False):
        """
        Constructor
        """
        self.verbose = verbose
        self.diff_predictor = None
        self.d_theta = None

    def create_train_dataset(self, predictions: numpy.ndarray, labels: numpy.ndarray,
                             input_data: numpy.ndarray, tt_split: float = 0.4,
                             encoded_labels: list = []):
        """
        Function that returns the d_theta dataset used to train the predictor.
        The dataset is stored internally.
        :param predictions: the classifiers predictions
        :param labels: the (encoded) ground truth labels
        :param input_data: the input data (features) to create d_theta
        :param tt_split: the train_test split for the dataset
        :return:
        """
        # Compute difficulty
        i_fun = 1*(predictions != labels)
        diff_value = numpy.average(i_fun, axis=0)
        if len(encoded_labels) > 0:
            labels = encoded_labels[labels]
        # Prepare d_theta
        if tt_split is not None and tt_split > 0:
            x_train, x_test, y_train, y_test = train_test_split(input_data, diff_value, train_size=tt_split, shuffle=False)
            split_index = int(i_fun.shape[1]*tt_split)
        else:
            x_train = input_data
            y_train = diff_value
            x_test = None
            y_test = None
            split_index = len(labels)
        self.d_theta = {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test,
                        "clf_preds_train": i_fun[:, 0:split_index], "clf_preds_test": i_fun[:, split_index:],
                        "base_label_train": labels[0: split_index], "base_label_test": labels[split_index:]}
        return self.d_theta

    def fit(self, pred_strategy: PredictionStrategy, params: dict):
        """
        Makes the prediction rejection strategy ready to be applied.
        In this case, it identifies ranges in which predictions should be excluded
        :return:
        """
        if params is None or not isinstance(params, dict):
            print("\tUnable to process params for the prediction strategy, using default regression instead")
            pred_strategy = PredictionStrategy.IID_REG
            params = {"reg_strategy": LinearRegression()}

        # Creating predictor object
        self.fit(extract_predictor(pred_strategy, params))

    def fit(self, pred_reg: PredictionRegressor):
        """
        Makes the prediction rejection strategy ready to be applied.
        In this case, it identifies ranges in which predictions should be excluded
        :return:
        """
        if pred_reg is None or not isinstance(pred_reg, PredictionRegressor):
            print("\tUnable to process the prediction strategy, using default decision tree regression instead")
            pred_reg = MLRegressor(ml_regressor=DecisionTreeRegressor())

        # Creating predictor object
        self.diff_predictor = pred_reg
        if self.d_theta is not None:
            if self.verbose:
                print("Training predictor '%s'" % self.diff_predictor.get_name())
            start_ms = current_ms()
            self.diff_predictor.fit(self.d_theta["x_train"], self.d_theta["y_train"])
            if self.verbose:
                print("\tTraining finished in %d ms" % (current_ms() - start_ms))
        else:
            print("ERROR: Unable to train difficulty predictor, need to generate the D_theta dataset (i.e., call the 'create_train_dataset' function) before")

    def predict(self) -> numpy.ndarray:
        """
        Predicts difficulty for a novel data point
        :param x_test: the data points to predict difficulty of
        :return: an array containing difficulty predictions as [0; 1] float numbers
        """
        if self.diff_predictor is not None:
            return self.diff_predictor.predict(self.d_theta["x_test"])
        else:
            print("ERROR: Unable to predict difficulty, need to train the predictor first (i.e., call the 'fit' function) before")
            return None

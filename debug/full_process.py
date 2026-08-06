# Support libs
import os

import numpy
import pandas
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor, GradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier

from nifty_lib.difficulty.DifficultyPredictor import DifficultyPredictor
from nifty_lib.difficulty.PredictionRegressor import MLRegressor
from nifty_lib.utils.classifier_utils import get_classifier_name
from nifty_lib.utils.dataset_utils import read_tabular_dataset
from nifty_lib.utils.general_utils import current_ms

# ------- GLOBAL VARS -----------

# Name of the folder in which look for tabular (CSV) datasets
CSV_FOLDER = "../../DatasetsFolder/all"
# Name of the column that contains the label in the tabular (CSV) dataset
LABEL_NAME = 'multilabel'
# Name of the 'normal' class in datasets. This will be used only for binary classification (anomaly detection)
NORMAL_TAG = 'normal'
# Name of the file in which outputs of the analysis will be saved
SCORES_FILE = "scores.csv"
# Percentage of test data wrt train data
TVT_SPLIT = [0.4, 0.6, 0]
# True if debug information needs to be shown
VERBOSE = True
# True if a dataframe with difficulty and predictions has to be printed
PRINT_DATAFRAME = True
DF_FOLDER = "dataframes"

# --------- SUPPORT FUNCTIONS ---------------

def get_classifier_population() -> list:
    """
    Function to get a learner to use, given its string tag
    :return: the list of classifiers to be trained to approximate theta into theta_hat
    """
    learners = [
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        MultinomialNB(),
        LinearDiscriminantAnalysis()]
    return learners

def get_prediction_regressors() -> list:
    """
    Returns the prediction regressors to be tested in experiments
    :return:
    """
    predictors = [
        MLRegressor(ml_regressor=DecisionTreeRegressor()),
        MLRegressor(ml_regressor=ExtraTreesRegressor(n_estimators=10)),
    ]
    return predictors


# ----------------------- MAIN ROUTINE ---------------------


if __name__ == '__main__':

    with open(SCORES_FILE, 'w') as f:
        f.write("dataset_tag,clf,binary,tt_split,acc,mcc,time,model_size\n")

    # Iterating over CSV files in folder
    for dataset_file in os.listdir(CSV_FOLDER):
        full_name = os.path.join(CSV_FOLDER, dataset_file)
        if full_name.endswith(".csv"):
            # if file is a CSV, it is assumed to be a dataset to be processed
            data_dict = read_tabular_dataset(dataset_name=os.path.join(CSV_FOLDER, dataset_file),
                                             label_name=LABEL_NAME, limit=50000,
                                             train_size=TVT_SPLIT[0], val_size=0.0,
                                             shuffle=True, l_encoding=False)
            # Create Difficulty Function
            diff_f = DifficultyPredictor(verbose=False)
            if VERBOSE:
                print('------------------- CREATE D THETA -----------------------')

            # Loop for training and testing classifiers and finally building d_theta
            if VERBOSE:
                print("Training classifiers to generate D_theta")

            clf_preds = []
            clf_population = get_classifier_population()
            for classifier in clf_population:
                start_time = current_ms()
                classifier.fit(data_dict["x_train"], data_dict["y_train"])
                mid_time = current_ms()
                preds = classifier.predict(data_dict["x_test"])
                pred_time = current_ms() - mid_time
                train_time = mid_time - start_time
                b_acc = balanced_accuracy_score(data_dict["y_test"], preds)
                clf_preds.append(preds)
                if VERBOSE:
                    print("\t[%d/%d] Classifier %s trained in %d ms, predicted in %d ms, balanced accuracy %.4f" %
                      (len(clf_preds), len(clf_population), get_classifier_name(classifier), train_time, pred_time, b_acc))

            start_time = current_ms()
            diff_f.create_train_dataset(predictions=numpy.asarray(clf_preds), labels=data_dict["y_test"],
                                        input_data=data_dict["x_test"], tt_split=0.5)
            if VERBOSE:
                print("D_theta dataset created in %d ms" % (current_ms() - start_time))

            if VERBOSE:
                print('-------------------- CREATE THETA_TILDE PREDICTORS -----------------------')

            regressors = get_prediction_regressors()
            reg_preds = []
            if VERBOSE:
                print("Training regressors to generate theta_tilde approximations")

            for regressor in regressors:
                start_time = current_ms()
                diff_f.fit(regressor)
                mid_time = current_ms()
                pred_diff = diff_f.predict()
                pred_time = current_ms() - mid_time
                train_time = mid_time - start_time
                mse = mean_squared_error(diff_f.d_theta["y_test"], pred_diff)
                reg_preds.append(pred_diff)
                if VERBOSE:
                    print("\t[%d/%d] Regressor %s trained in %d ms, predicted in %d ms, MSE %.4f" %
                      (len(reg_preds), len(regressors), regressor.get_name(), train_time, pred_time, mse))

            if VERBOSE:
                print("Printing test dataframe ...")
            out_df = pandas.DataFrame(data=diff_f.d_theta["x_test"], columns=data_dict["feature_names"])
            out_df["label"] = diff_f.d_theta["base_label_test"]
            for i in range(0, len(clf_population)):
                out_df["ERR_" + get_classifier_name(clf_population[i])] = diff_f.d_theta["clf_preds_test"][i, :]
            out_df["TRUE_difficulty"] = diff_f.d_theta["y_test"]
            for i in range(0, len(regressors)):
                out_df["_PRED_" + regressors[i].get_name()] = reg_preds[i]
            out_df.to_csv(os.path.join(DF_FOLDER, "TESTDF_" + dataset_file), index=False)

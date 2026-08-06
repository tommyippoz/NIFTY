# Support libs
import copy
import os
import random

from confens.classifiers.ConfidenceBagging import ConfidenceBagging
from confens.classifiers.ConfidenceBoosting import ConfidenceBoosting
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier

import matplotlib.pyplot as plt

import numpy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, \
    AdaBoostClassifier
from sklearn.metrics import balanced_accuracy_score, mean_squared_error, mean_absolute_error, \
    mean_absolute_percentage_error, max_error, r2_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from nifty_lib.difficulty.DifficultyPredictor import DifficultyPredictor
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
# List of RNG seeds to reiterate RQ1 analysis and make it statistically significant
RNG_SEEDS = [9, 33, 42, 79, 131]
# Folder for Plots
PLOTS_FOLDER = "RQ1_plots"

# --------- SUPPORT FUNCTIONS ---------------

def get_classifier_population() -> list:
    """
    Function to get a learner to use, given its string tag
    :return: the list of classifiers to be trained to approximate theta into theta_hat
    """
    learners = [
        LogisticRegression(max_iter=1000),
        RidgeClassifier(),
        SGDClassifier(loss="hinge", penalty="l2", max_iter=1000),
        SGDClassifier(loss="modified_huber", penalty="l2", max_iter=1000),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(n_estimators=10),
        AdaBoostClassifier(n_estimators=100),
        Pipeline([('scale', MinMaxScaler()), ('clf', MultinomialNB())]),
        Pipeline([('scale', MinMaxScaler()), ('clf', GaussianNB())]),
        ConfidenceBoosting(n_base=20, clf=Pipeline([('scale', MinMaxScaler()), ('clf', GaussianNB())])),
        DecisionTreeClassifier(),
        XGBClassifier(n_estimators=10),
        ConfidenceBoosting(n_base=10, clf=XGBClassifier(n_estimators=10)),
        XGBClassifier(n_estimators=100),
        ConfidenceBagging(n_base=5, clf=ExtraTreeClassifier()),
        ConfidenceBagging(n_base=10, clf=ExtraTreeClassifier()),
        ConfidenceBoosting(n_base=10, clf=ExtraTreeClassifier()),
        RandomForestClassifier(n_estimators=10),
        RandomForestClassifier(n_estimators=100),
        GradientBoostingClassifier(n_estimators=10),
        GradientBoostingClassifier(n_estimators=20),
        LinearSVC(),
        KNeighborsClassifier(n_neighbors=1, algorithm="kd_tree"),
        KNeighborsClassifier(n_neighbors=3, algorithm="kd_tree"),
        KNeighborsClassifier(n_neighbors=5, algorithm="kd_tree"),
        KNeighborsClassifier(n_neighbors=11, algorithm="kd_tree"),
        ExtraTreeClassifier(),
        ExtraTreesClassifier(n_estimators=10),
        ExtraTreesClassifier(n_estimators=100),
        LinearDiscriminantAnalysis(),

    ]
    return learners


# ----------------------- MAIN ROUTINE ---------------------


if __name__ == '__main__':

    with open(SCORES_FILE, 'w') as f:
        f.write("dataset_tag,clf,binary,tt_split,acc,mcc,time,model_size\n")

    # Iterating over CSV files in folder
    for dataset_file in os.listdir(CSV_FOLDER):
        full_name = os.path.join(CSV_FOLDER, dataset_file)
        if full_name.endswith(".csv"):
            # if file is a CSV, it is assumed to be a dataset to be processed
            dataset_name = dataset_file.replace(".csv", "")
            data_dict = read_tabular_dataset(dataset_name=os.path.join(CSV_FOLDER, dataset_file),
                                             label_name=LABEL_NAME, limit=100000,
                                             train_size=TVT_SPLIT[0], val_size=0.0,
                                             shuffle=True, l_encoding=True)
            # Create Difficulty Function
            diff_f = DifficultyPredictor(verbose=False)
            if VERBOSE:
                print('------------------- CREATE D THETA -----------------------')

            # Loop for training and testing classifiers and finally building d_theta
            if VERBOSE:
                print("Training ALL classifiers to generate D_theta i.e., without approximation")

            clf_preds_dict = {}
            clf_population = get_classifier_population()
            for classifier in clf_population:
                start_time = current_ms()
                classifier.fit(data_dict["x_train"], data_dict["y_train"])
                mid_time = current_ms()
                preds = classifier.predict(data_dict["x_test"])
                pred_time = current_ms() - mid_time
                train_time = mid_time - start_time
                b_acc = balanced_accuracy_score(data_dict["y_test"], preds)
                clf_preds_dict[get_classifier_name(classifier)] = preds
                if VERBOSE:
                    print("\t[%d/%d] Classifier %s trained in %d ms, predicted in %d ms, balanced accuracy %.4f" %
                          (len(clf_preds_dict), len(clf_population), get_classifier_name(classifier), train_time,
                           pred_time, b_acc))

            start_time = current_ms()
            d_theta_all = diff_f.create_train_dataset(predictions=numpy.asarray(list(clf_preds_dict.values())),
                                                      labels=data_dict["y_test"],
                                                      input_data=data_dict["x_test"],
                                                      tt_split=0,
                                                      encoded_labels=data_dict["label_names"])
            theta = d_theta_all["y_train"]
            if VERBOSE:
                print("D_theta dataset created in %d ms" % (current_ms() - start_time))

            if VERBOSE:
                print('-------------------- RANDOM PERTURBATIONS -----------------------')

            exp_list = []
            metrics = []
            # Iterating over random seeds
            for seed in RNG_SEEDS:

                # Shuffling classifiers
                clf_shuffled = copy.deepcopy(clf_population)
                random.seed(seed)
                random.shuffle(clf_shuffled)

                exp_log = {"rng_seed": seed,
                           "clf_list": [get_classifier_name(x) for x in clf_shuffled],
                           "exp_list": []}
                # Iterating over classifiers
                relevant_preds = []
                for classifier in clf_shuffled:
                    relevant_preds.append(clf_preds_dict[get_classifier_name(classifier)])
                    d_theta_tmp = diff_f.create_train_dataset(predictions=numpy.asarray(relevant_preds),
                                                              labels=data_dict["y_test"],
                                                              input_data=data_dict["x_test"],
                                                              tt_split=0)
                    theta_hat = d_theta_tmp["y_train"]
                    exp_log["exp_list"].append({"mae": mean_absolute_error(theta, theta_hat),
                                                "mae_p": mean_absolute_percentage_error(theta, theta_hat),
                                                "max_e": max_error(theta, theta_hat),
                                                "r2": r2_score(theta, theta_hat),
                                                "mse": mean_squared_error(theta, theta_hat)})

                exp_list.append(exp_log)
                if len(metrics) == 0:
                    metrics = list(exp_log["exp_list"][0].keys())

            # Average results per seed
            exp_list_n = [x for x in range(1, len(clf_population)+1)]
            exp_list_avg = {x: [] for x in metrics}
            exp_list_std = {x: [] for x in metrics}
            for i in exp_list_n:
                for metric in metrics:
                    data = numpy.asarray([x["exp_list"][i-1][metric] for x in exp_list])
                    exp_list_avg[metric].append(numpy.average(data))
                    exp_list_std[metric].append(numpy.std(data))

            # Print plots in specific folders
            if not os.path.exists(PLOTS_FOLDER):
                os.mkdir(PLOTS_FOLDER)
            if not os.path.exists(os.path.join(PLOTS_FOLDER, dataset_name)):
                os.mkdir(os.path.join(PLOTS_FOLDER, dataset_name))
            for metric in metrics:
                fig, ax = plt.subplots()
                ax.plot(exp_list_n, exp_list_avg[metric])
                ax.set_xlabel("Population size")
                ax.set_ylabel(metric)
                ax.errorbar(exp_list_n, exp_list_avg[metric], yerr=exp_list_std[metric],
                            fmt="o", color="red", capsize=6)
                plt.savefig(os.path.join(PLOTS_FOLDER, dataset_name, metric + "_plot.png"), dpi=300)
                plt.close()

            a = 1

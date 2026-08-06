import inspect
from collections.abc import Iterable

import numpy
from sklearn.base import is_classifier
from sklearn.utils.validation import check_is_fitted, check_array

def predict_proba(clf, X, get_base:bool = False):
    """
    Function to predict probabilities of a classifier
    Needed to overcome issues in pyod's predict_proba
    :param get_base: Tue if predictions of base-learners have to be returned as well
    :param clf: the classifier to be used
    :param X: the test set
    :return:
    """
    if 'get_base' in inspect.getfullargspec(clf.predict_proba)[0]:
        return clf.predict_proba(X, get_base=get_base)
    else:
        return clf.predict_proba(X)


def get_classifier_name(clf_object):
    """
    Gets a string representing the classifier name
    :param clf_object: the object meant to be a classifier
    :return: a string
    """
    clf_name = ""
    if clf_object is not None:
        clf_name = clf_object.__class__.__name__
        if hasattr(clf_object, "base_estimator") and hasattr(clf_object, "n_estimators"):
            clf_name = clf_name + "(" + get_single_classifier_name(clf_object.base_estimator) + ";" \
                       + str(clf_object.n_estimators) + ")"
        elif hasattr(clf_object, "estimators"):
            if len(clf_object.estimators) < 5:
                clf_name = clf_name + "(" + "@".join([get_single_classifier_name(clf) for clf in clf_object.estimators]) + ";" \
                       + str(len(clf_object.estimators)) + ")"
            else:
                clf_name = clf_name + "(" + str(len(clf_object.estimators)) + ")"
        elif hasattr(clf_object, "n_estimators"):
            clf_name = clf_name + "(" + str(clf_object.n_estimators) + ")"
        elif hasattr(clf_object, "n_base"):
            clf_name = clf_name + "(" + str(clf_object.n_base) + ";" + get_classifier_name(clf_object.clf) + ")"
        elif clf_name == "Pipeline":
            clf_name = get_classifier_name(clf_object.named_steps['clf'])
        elif clf_name == "KNeighborsClassifier":
            clf_name = clf_name + "(" + str(clf_object.n_neighbors) + ")"
        elif clf_name == "SGDClassifier":
            clf_name = clf_name + "(" + str(clf_object.loss) + ")"
    return clf_name


def get_single_classifier_name(clf_object):
    """
    Gets a string representing the classifier name, assuming the object contains a single classifier
    :param clf_object: the object meant to be a classifier
    :return: a string
    """
    if hasattr(clf_object, "classifier_name") and callable(clf_object.classifier_name):
        clf_name = clf_object.classifier_name()
        if clf_name == 'Pipeline':
            for x in list(clf_object.named_steps.keys()):
                if is_classifier(clf_object[x]):
                    clf_name = get_single_classifier_name(clf_object[x])
    elif isinstance(clf_object, tuple):
        clf_name = str(clf_object[0])
        for x in clf_object:
            if is_classifier(x):
                clf_name = get_single_classifier_name(x)
    else:
        clf_name = clf_object.__class__.__name__
        if clf_name == 'Pipeline':
            for x in list(clf_object.named_steps.keys()):
                if is_classifier(clf_object[x]):
                    clf_name = get_single_classifier_name(clf_object[x])
    return clf_name


def predict_confidence(clf, X):
    """
    Method to compute the confidence in predictions of a classifier
    :param clf: the classifier
    :param X: the test set
    :return: array of confidence scores
    """
    c_conf = None
    if is_classifier(clf):
        if hasattr(clf, 'predict_confidence') and callable(clf.predict_confidence):
            c_conf = clf.predict_confidence(X)
        else:
            y_proba = predict_proba(clf, X)
            c_conf = numpy.max(y_proba, axis=1)
    return c_conf

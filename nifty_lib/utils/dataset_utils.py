import numpy
import pandas
import sklearn

def read_tabular_dataset(dataset_name: str, label_name: str, limit: int = numpy.nan, train_size: float = 0.5,
                         val_size: float = 0.2, shuffle: bool = True, l_encoding: bool = True) -> dict:
    """
    Method to process an input dataset as CSV
    :param l_encoding: if True, encodes labels as integers (useful for compatibility with some classifiers)
    :param shuffle: true if data has to be shuffled before splitting
    :param val_size: percentage of dataset to be used for validation
    :param train_size: percentage of dataset to be used for training
    :param limit: integer to cut dataset if needed.
    :param dataset_name: name of the file (CSV) containing the dataset
    :param label_name: name of the feature containing the label
    :return: many values for analysis
    """
    # Loading Dataset
    df = pandas.read_csv(dataset_name, sep=",")

    # Shuffle
    if shuffle:
        df = df.sample(frac=1.0)
    df = df.fillna(0)
    df = df.replace('null', 0)

    # Testing Purposes
    if (numpy.isfinite(limit)) & (limit < len(df.index)):
        df = df[0:limit]

    if l_encoding:
        encoding = pandas.factorize(df[label_name])
        y_enc = encoding[0]
        labels = encoding[1]
    else:
        y_enc = df[label_name].to_numpy()
        labels = numpy.unique(y_enc)

    # Basic Pre-Processing
    print("\nDataset  %s loaded: %d items" % (dataset_name, len(df.index)))

    # Train/Test Split of Classifiers
    x = df.drop(columns=[label_name])
    x_no_cat = x.select_dtypes(exclude=['object'])
    feature_list = x_no_cat.columns
    x_no_cat = x_no_cat.to_numpy()
    x_tr, x_te, y_tr, y_te = sklearn.model_selection.train_test_split(x_no_cat, y_enc,
                                                                      test_size=1 - train_size,
                                                                      shuffle=shuffle)
    if val_size > 0:
        x_val, x_te, y_val, y_te = sklearn.model_selection.train_test_split(x_te, y_te,
                                                                            test_size=1 - (val_size / (1 - train_size)),
                                                                            shuffle=shuffle)
    else:
        x_val = None
        y_val = None

    return {"x_train": x_tr, "x_test": x_te, "x_val": x_val, "y_train": y_tr, "y_test": y_te, "y_val": y_val,
            "label_names": labels, "feature_names": feature_list}


def read_binary_tabular_dataset(dataset_name: str, label_name: str, limit: int = numpy.nan, train_size: float = 0.5,
                                val_size: float = 0.2, shuffle: bool = True, l_encoding: bool = False, normal_tag: str = 'normal') -> dict:
    """
    Method to process an input dataset as CSV
    :param normal_tag: string that identifies the class that has to be treated as normal. All other classes will become "anomaly"
    :param l_encoding: if True, encodes labels as integers (useful for compatibility with some classifiers)
    :param shuffle: true if data has to be shuffled before splitting
    :param val_size: percentage of dataset to be used for validation
    :param train_size: percentage of dataset to be used for training
    :param limit: integer to cut dataset if needed.
    :param dataset_name: name of the file (CSV) containing the dataset
    :param label_name: name of the feature containing the label
    :return: many values for analysis
    """
    # Loading Dataset
    tab_dict = read_tabular_dataset(dataset_name, label_name, limit, train_size, val_size, shuffle, False)
    tab_dict["normal_perc"] = numpy.average(numpy.where(tab_dict["y_train"] == normal_tag, 1, 0))
    if l_encoding:
        tab_dict["y_train"] = numpy.where(tab_dict["y_train"] == normal_tag, 0, 1)
        tab_dict["y_test"] = numpy.where(tab_dict["y_test"] == normal_tag, 0, 1)
        if tab_dict["y_test"] is not None:
            tab_dict["y_val"] = numpy.where(tab_dict["y_val"] == normal_tag, 0, 1)
        tab_dict["label_names"] = [0, 1]
    else:
        tab_dict["y_train"] = numpy.where(tab_dict["y_train"] == normal_tag, normal_tag, "anomaly")
        tab_dict["y_test"] = numpy.where(tab_dict["y_test"] == normal_tag, normal_tag, "anomaly")
        if tab_dict["y_test"] is not None:
            tab_dict["y_val"] = numpy.where(tab_dict["y_val"] == normal_tag, normal_tag, "anomaly")
        tab_dict["label_names"] = [normal_tag, "anomaly"]
    return tab_dict

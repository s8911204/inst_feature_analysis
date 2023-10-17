#!/usr/bin/python3
import argparse
import collections
import json
import os

import numpy as np
import pandas as pd
import pydotplus
from joblib import dump
from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier

import utils


def load_data(data_folder, trial_mode):
    """
    Function Name: load_data
    Parameters:
    - data_folder (str): The folder which contains the dataset.
    - trial_mode (bool): If True, the function will stop loading data after 30000 records.
    Returns:
    - data_set (pandas.DataFrame): The loaded dataset.
    Errors/Exceptions: None.
    Example:
    >>> load_data('data/', True)
    Notes: None.
    """
    data_set = pd.DataFrame()
    min_group_size = data_set.groupby("target").size().min()
    for filename in os.listdir(data_folder):
        fullpath = os.path.join(data_folder, filename)
        if not os.path.isfile(fullpath):
            continue
        if not fullpath.endswith(".csv"):
            continue
        data = utils.toDf(fullpath)
        if data_set.empty:
            data_set = data
        else:
            data_set = pd.concat([data_set, data], ignore_index=True)
        data_set = pd.DataFrame(
            data_set.groupby("target").apply(
                lambda x: x.sample(min_group_size).reset_index(drop=True)
            )
        )
        print("Import %d records in dataframe" % len(data_set))
        if trial_mode and len(data_set) > 300000:
            break
    return data_set


def data_split(data_set):
    """
    Function Name: data_split
    Parameters:
    - data_set (pandas.DataFrame): The dataset to be split.
    Returns:
    - (pandas.DataFrame, pandas.DataFrame): The split dataset.
    Errors/Exceptions: None.
    Example:
    >>> data_split(df)
    Notes: The function splits the dataset into a training set and a test set with a test size of 0.3.
    """
    return train_test_split(data_set, test_size=0.3)


def xy(data_set):
    """
    Function Name: xy
    Parameters:
    - data_set (pandas.DataFrame): The dataset to be split into features and target.
    Returns:
    - (pandas.DataFrame, pandas.Series): The features and the target.
    Errors/Exceptions: None.
    Example:
    >>> xy(df)
    Notes: The function splits the dataset into features and target. The target is the 'target' column of the dataset.
    """
    Y = data_set["target"]
    X = data_set.drop(columns=["module name", "address", "target"])
    if "index" in X:
        X.drop(columns=["index"], inplace=True)
    return X, Y


def display_scores(scores):
    """
    Function Name: display_scores
    Parameters:
    - scores (dict): The scores to be displayed.
    Returns:
    - best_model_index (int): The index of the best model.
    Errors/Exceptions: None.
    Example:
    >>> display_scores(scores)
    Notes: The function displays the scores and returns the index of the best model.
    """
    best_acc = 0.0
    best_acc_index = 0
    for acc in scores["test_accuracy"]:
        if acc > best_acc:
            best_acc = acc
            best_acc_index = np.where(scores["test_accuracy"] == best_acc)
    best_model_index = best_acc_index[0][0]
    # feature_importance = scores['estimator'][best_model_index].feature_importances_.tolist()
    # print(feature_importance)
    print("==== Accuracy ==== ")
    print(scores["test_accuracy"])
    print("==== Recal ==== ")
    print(scores["test_recall"])
    print("==== Precision  ====")
    print(scores["test_precision"])
    print("==== Train Accuracy ==== ")
    print(scores["train_accuracy"])
    print("==== Train Recall  ==== ")
    print(scores["train_recall"])
    print("==== Train Precision  ==== ")
    print(scores["train_precision"])
    print("Best model index: %d \n" % best_model_index)
    return best_model_index


def train(data_set, mdepth, mleaf):
    """
    Function Name: train
    Parameters:
    - data_set (pandas.DataFrame): The dataset to be used for training.
    - mdepth (int): The maximum depth of the decision tree.
    - mleaf (int): The minimum number of samples required to be at a leaf node.
    Returns:
    - (sklearn.tree.DecisionTreeClassifier): The best model.
    Errors/Exceptions: None.
    Example:
    >>> train(df, 5, 3)
    Notes: The function trains a decision tree classifier and returns the best model.
    """
    X, Y = xy(data_set)
    dc = DecisionTreeClassifier(
        criterion="gini", max_depth=mdepth, min_samples_leaf=mleaf
    )
    scoring_types = ["accuracy", "precision", "recall"]
    scores = cross_validate(
        dc,
        X,
        Y,
        cv=5,
        return_estimator=True,
        scoring=scoring_types,
        return_train_score=True,
        n_jobs=5,
    )
    best_model_index = display_scores(scores)
    return scores["estimator"][best_model_index]


def test(data_set, best_model):
    """
    Function Name: test
    Parameters:
    - data_set (pandas.DataFrame): The dataset to be used for testing.
    - best_model (sklearn.tree.DecisionTreeClassifier): The model to be used for testing.
    Returns:
    - result (dict): The result of the test.
    Errors/Exceptions: None.
    Example:
    >>> test(df, model)
    Notes: The function tests the model and returns the result.
    """
    X, Y = xy(data_set)
    y_predict = best_model.predict(X)
    acc = accuracy_score(Y, y_predict)
    precision = precision_score(Y, y_predict)
    recall = recall_score(Y, y_predict)
    print("Best Model A:R:P = %f:%f:%f" % (acc, recall, precision))
    result = {"accuracy": acc, "recall": recall, "precision": precision}
    return result


def final_report(result_set, out_dir):
    """
    Function Name: final_report
    Parameters:
    - result_set (list): The list of results to be reported.
    - out_dir (str): The directory where the report will be saved.
    Returns: None.
    Errors/Exceptions: None.
    Example:
    >>> final_report(results, 'out/')
    Notes: The function generates a final report and saves it in the specified directory.
    """
    acc_list = []
    recall_list = []
    precision_list = []
    for result in result_set:
        acc_list.append(result["accuracy"])
        recall_list.append(result["recall"])
        precision_list.append(result["precision"])
    index_list = []
    for i in range(0, len(result_set)):
        index_list.append("R%d" % (i + 1))
    report_data = pd.DataFrame(
        {"accuracy": acc_list, "recall": recall_list, "precision": precision_list},
        index=index_list,
    )
    print(report_data.head(n=10))
    print("mean accuracy: %f" % report_data["accuracy"].mean())
    report_data["recall"].max()

    def persist_best_model(result_set, best_recall, out_dir):
        """
        Function Name: persist_best_model
        Parameters:
        - result_set (list): The list of results.
        - best_recall (float): The best recall score.
        - out_dir (str): The directory where the best model will be saved.
        Returns: None.
        Errors/Exceptions: None.
        Example:
        >>> persist_best_model(results, 0.9, 'out/')
        Notes: The function saves the best model in the specified directory.
        """
        idx = 0
        for result in result_set:
            if result["recall"] == best_recall:
                print("The best model is run %d" % idx)
                break
            idx += 1
        if idx < len(result_set):
            dump(
                result_set[idx]["best_model"],
                os.path.join(out_dir, "saved_model.joblib"),
            )


def get_args():
    """
    Function Name: get_args
    Parameters: None.
    Returns:
    - args (argparse.Namespace): The parsed arguments.
    Errors/Exceptions: None.
    Example:
    >>> get_args()
    Notes: The function parses the command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("data_folder", help="the folder which contains dataset.")
    parser.add_argument(
        "--out_dir", required=True, help="the folder which used as output base folder."
    )
    parser.add_argument(
        "--runs", type=int, default=5, help="total runs of cross-validation"
    )
    parser.add_argument("--depth", type=int, default=5, help="depth of decision tree.")
    parser.add_argument(
        "--min_leaf_samples",
        type=int,
        default=3,
        help="the minimum number of samples of leaf node.",
    )
    parser.add_argument(
        "--trial",
        "-t",
        dest="trial",
        action="store_true",
        help="Enable test mode. # of data less then 200,000",
    )
    return parser.parse_args()


def get_featurelist(dataset):
    """
    Function Name: get_featurelist
    Parameters:
    - dataset (pandas.DataFrame): The dataset from which to get the feature list.
    Returns:
    - (list): The list of features.
    Errors/Exceptions: None.
    Example:
    >>> get_featurelist(df)
    Notes: The function returns the list of features from the dataset.
    """
    X, Y = xy(dataset)
    return list(X.columns)


def get_feature_importance(model, feature_list):
    """
    Function Name: get_feature_importance
    Parameters:
    - model (sklearn.tree.DecisionTreeClassifier): The trained model.
    - feature_list (list): The list of features used in the model.
    Returns:
    - fii (collections.OrderedDict): An ordered dictionary of feature importances.
    Errors/Exceptions: None.
    Example:
    >>> get_feature_importance(model, features)
    Notes: The function only includes features with an importance greater than 0.0 in the returned dictionary.
    """
    fea_imp = model.feature_importances_.tolist()
    fii = {}
    idx = 0
    for feature in feature_list:
        if fea_imp[idx] > 0.0:
            fii[feature] = fea_imp[idx]
        idx += 1
    sorted_fii = sorted(fii.items(), key=lambda kv: kv[1])
    fii = collections.OrderedDict(sorted_fii)
    print(json.dumps(fii, indent=4))
    return fii


def plot_tree(model, run, feature_list, out_dir):
    """
    Function Name: plot_tree
    Parameters:
    - model (sklearn.tree.DecisionTreeClassifier): The model to be plotted.
    - run (int): The run number.
    - feature_list (list): The list of features used in the model.
    - out_dir (str): The directory where the plot will be saved.
    Returns: None.
    Errors/Exceptions: None.
    Example:
    >>> plot_tree(model, 1, features, 'out/')
    Notes: The function plots the decision tree and saves it in the specified directory.
    """
    dot_data = tree.export_graphviz(
        model, feature_names=feature_list, filled=True, out_file=None
    )
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf(os.path.join(out_dir, "dc_%d.pdf" % run))


def main():
    """
    Function Name: main
    Parameters: None.
    Returns: None.
    Errors/Exceptions: Mention any errors or exceptions that can be thrown during the execution of the function.
    Example:
    >>> main()
    Notes: The function loads data, trains a model, tests the model, gets feature importances, plots the decision tree, and generates a final report.
    """
    context = {}
    context["arg"] = get_args()
    # data_folder = sys.argv[1]
    # runs = sys.argv[2]
    data_set = load_data(context["arg"].data_folder, context["arg"].trial)
    true_percent = len(data_set[data_set["target"] == 1]) / len(data_set) * 100
    print("True is %d" % true_percent)
    print(data_set)
    result_set = []
    for i in range(0, context["arg"].runs):
        train_set, test_set = data_split(data_set)
        feature_list = get_featurelist(train_set)
        best_model = train(
            train_set, context["arg"].depth, context["arg"].min_leaf_samples
        )
        result = test(test_set, best_model)
        result["feature_importance"] = get_feature_importance(best_model, feature_list)
        plot_tree(best_model, i, feature_list, context["arg"].out_dir)
        result["best_model"] = best_model
        result_set.append(result)
        # round_report(result)
        # result_set.append(result)
    final_report(result_set, context["arg"].out_dir)


if __name__ == "__main__":
    main()

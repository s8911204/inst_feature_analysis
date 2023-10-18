import argparse
import collections
import json
import logging
import os

import matplotlib.pyplot as plt
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
    logging.info(
        "Starting function load_data with data_folder: %s, trial_mode: %s",
        data_folder,
        trial_mode,
    )
    data_set = pd.DataFrame()
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
        logging.info("Import %d records in dataframe" % len(data_set))
        if trial_mode and len(data_set) > 30000:
            break
    logging.info("Ending function load_data with data_set: %s", data_set)
    return data_set


def data_split(data_set):
    logging.info("Starting function data_split with data_set: %s", data_set)
    result = train_test_split(data_set, test_size=0.3)
    logging.info("Ending function data_split with result: %s", result)
    return result


def xy(data_set):
    logging.info("Starting function xy with data_set: %s", data_set)
    Y = data_set["target"]
    X = data_set.drop(columns=["module name", "address", "target"])
    if "index" in X:
        X.drop(columns=["index"], inplace=True)
    logging.info("Ending function xy with X: %s, Y: %s", X, Y)
    return X, Y


def display_scores(scores):
    logging.info("Starting function display_scores with scores: %s", scores)
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
    logging.info(
        "Starting function train with data_set: %s, mdepth: %s, mleaf: %s",
        data_set,
        mdepth,
        mleaf,
    )
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
    logging.info("Ending function train with best_model_index: %s", best_model_index)
    return scores["estimator"][best_model_index]


def test(data_set, best_model):
    logging.info(
        "Starting function test with data_set: %s, best_model: %s", data_set, best_model
    )
    X, Y = xy(data_set)
    y_predict = best_model.predict(X)
    acc = accuracy_score(Y, y_predict)
    precision = precision_score(Y, y_predict)
    recall = recall_score(Y, y_predict)
    logging.info("Best Model A:R:P = %f:%f:%f" % (acc, recall, precision))
    result = {"accuracy": acc, "recall": recall, "precision": precision}
    logging.info("Ending function test with result: %s", result)
    return result


def final_report(result_set, out_dir):
    logging.info(
        "Starting function final_report with result_set: %s, out_dir: %s",
        result_set,
        out_dir,
    )
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
    best_recall = report_data["recall"].max()
    persist_best_model(result_set, best_recall, out_dir)
    report_data.boxplot(column=["accuracy", "recall", "precision"])
    plt.savefig(os.path.join(out_dir, "boxplot.png"))


def persist_best_model(result_set, best_recall, out_dir):
    logging.info(
        "Starting function persist_best_model with result_set: %s, best_recall: %s, out_dir: %s",
        result_set,
        best_recall,
        out_dir,
    )
    idx = 0
    for result in result_set:
        dump(
            result_set[idx]["best_model"],
            os.path.join(out_dir, "saved_model_%d.joblib" % idx),
        )
        idx += 1
    logging.info("Ending function persist_best_model")


def get_args():
    logging.info("Starting function get_args")
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
    args = parser.parse_args()
    logging.info("Ending function get_args with args: %s", args)
    return args


def get_featurelist(dataset):
    logging.info("Starting function get_featurelist with dataset: %s", dataset)
    X, Y = xy(dataset)
    feature_list = list(X.columns)
    logging.info("Ending function get_featurelist with feature_list: %s", feature_list)
    return feature_list


def get_feature_importance(model, feature_list):
    logging.info(
        "Starting function get_feature_importance with model: %s, feature_list: %s",
        model,
        feature_list,
    )
    fea_imp = model.feature_importances_.tolist()
    fii = {}
    idx = 0
    for feature in feature_list:
        if fea_imp[idx] > 0.0:
            fii[feature] = fea_imp[idx]
        idx += 1
    sorted_fii = sorted(fii.items(), key=lambda kv: kv[1])
    fii = collections.OrderedDict(sorted_fii)
    logging.info(json.dumps(fii, indent=4))
    logging.info("Ending function get_feature_importance with fii: %s", fii)
    return fii


def plot_tree(model, run, feature_list, out_dir):
    logging.info(
        "Starting function plot_tree with model: %s, run: %s, feature_list: %s, out_dir: %s",
        model,
        run,
        feature_list,
        out_dir,
    )
    dot_data = tree.export_graphviz(
        model, feature_names=feature_list, filled=True, out_file=None
    )
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf(os.path.join(out_dir, "dc_%d.pdf" % run))
    logging.info("Ending function plot_tree")


def main():
    logging.info("Starting function main")
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

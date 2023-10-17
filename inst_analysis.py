#!/usr/bin/python3
import sys
import os
import utils
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import argparse
from sklearn.metrics import recall_score, precision_score, accuracy_score
import collections
import json
from sklearn import tree
import pydotplus
import matplotlib.pyplot as plt
from joblib import dump, load


def load_data(data_folder, trial_mode):
    data_set = pd.DataFrame() 
    for filename in os.listdir(data_folder):
        fullpath = os.path.join(data_folder, filename)
        if not os.path.isfile(fullpath):
            continue
        if not fullpath.endswith('.csv'):
            continue
        data = utils.toDf(fullpath)
        if data_set.empty:
            data_set = data
        else:
            data_set = pd.concat(
                [data_set, data], ignore_index=True)
        # g = data_set.groupby('target')
        # data_set = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))
        print('Import %d records in dataframe' % len(data_set))
        if trial_mode and len(data_set) > 30000:
        # if trial_mode and len(data_set) > 300000:
            break
    return data_set


def data_split(data_set):
    return train_test_split(data_set, test_size=0.3)


def xy(data_set):
    Y = data_set['target']
    X = data_set.drop(columns=['module name', 'address', 'target'])
    if 'index' in X:
        X.drop(columns=['index'], inplace=True)
    return X, Y


def display_scores(scores):
    best_acc = 0.0
    best_acc_index = 0
    for acc in scores['test_accuracy']:
        if acc > best_acc:
            best_acc = acc
            best_acc_index = np.where(scores['test_accuracy'] == best_acc)  
    best_model_index = best_acc_index[0][0] 
    # feature_importance = scores['estimator'][best_model_index].feature_importances_.tolist()
    # print(feature_importance)        
    print('==== Accuracy ==== ')
    print(scores['test_accuracy'])
    print('==== Recal ==== ')
    print(scores['test_recall'])
    print('==== Precision  ====')
    print(scores['test_precision'])
    print('==== Train Accuracy ==== ')
    print(scores['train_accuracy'])   
    print('==== Train Recall  ==== ')
    print(scores['train_recall'])
    print('==== Train Precision  ==== ')
    print(scores['train_precision']) 
    print("Best model index: %d \n" % best_model_index)
    return best_model_index  

def train(data_set, mdepth, mleaf):
    X, Y = xy(data_set)
    dc = DecisionTreeClassifier(criterion='gini', max_depth=mdepth, min_samples_leaf=mleaf)
    scoring_types = ['accuracy', 'precision', 'recall']
    scores = cross_validate(dc, X, Y, cv=5, return_estimator=True, scoring=scoring_types, return_train_score=True, n_jobs=5)
    best_model_index = display_scores(scores)
    return scores['estimator'][best_model_index]
    

def test(data_set, best_model):
    X, Y = xy(data_set)
    y_predict = best_model.predict(X)
    acc = accuracy_score(Y, y_predict)
    precision = precision_score(Y, y_predict)
    recall = recall_score(Y, y_predict)
    print('Best Model A:R:P = %f:%f:%f' % (acc, recall, precision))
    result = {'accuracy': acc, 'recall': recall, 'precision': precision}
    return result


def final_report(result_set, out_dir):
    acc_list = []
    recall_list = []
    precision_list = []
    for result in result_set:
        acc_list.append(result['accuracy'])
        recall_list.append(result['recall'])
        precision_list.append(result['precision'])
    index_list = []
    for i in range(0, len(result_set)):
        index_list.append('R%d' % (i+1))
    report_data = pd.DataFrame({
        'accuracy': acc_list,
        'recall': recall_list,
        'precision': precision_list
    }, index = index_list)
    print(report_data.head(n=10))
    print('mean accuracy: %f' % report_data['accuracy'].mean())
    best_recall = report_data['recall'].max()
    persist_best_model(result_set, best_recall, out_dir)
    report_data.boxplot(column=['accuracy', 'recall', 'precision'])
    plt.savefig(os.path.join(out_dir, "boxplot.png"))
    
    
    def persist_best_model(result_set, best_recall, out_dir):
        idx = 0
        for result in result_set:
            if (result['recall'] == best_recall):
               print('The best model is run %d' % idx)
               break
            dump(result['best_model'], os.path.join(out_dir, "saved_model_%d.joblib" % idx))
            idx += 1
        dump(result_set[idx]['best_model'], os.path.join(out_dir, "saved_model.joblib"))
    
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("data_folder", help='the folder which contains dataset.')
        parser.add_argument("--out_dir", required=True, help='the folder which used as output base folder.')
        parser.add_argument("--runs", type=int, default=5, help='total runs of cross-validation')
        parser.add_argument("--depth", type=int, default=5, help='depth of decision tree.')
        parser.add_argument("--min_leaf_samples", type=int, default=3, help='the minimum number of samples of leaf node.')
        parser.add_argument("--trial", "-t", dest='trial', action='store_true', help="Enable test mode. # of data less then 200,000")
        return parser.parse_args()


def get_featurelist(dataset):
    X, Y = xy(dataset)
    return list(X.columns)


def get_feature_importance(model, feature_list):
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
    dot_data = tree.export_graphviz(model, feature_names=feature_list, filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf(os.path.join(out_dir, "dc_%d.pdf" % run))


def main():
    context = {}
    context['arg'] = get_args()
    # data_folder = sys.argv[1]
    # runs = sys.argv[2]
    data_set = load_data(context['arg'].data_folder, context['arg'].trial)
    true_percent = len(data_set[data_set['target'] == 1])/len(data_set) * 100
    print('True is %d' % true_percent)
    print(data_set)
    result_set = []
    for i in range(0, context['arg'].runs):
        train_set, test_set = data_split(data_set)
        feature_list = get_featurelist(train_set)
        best_model = train(train_set, context['arg'].depth, context['arg'].min_leaf_samples)
        result = test(test_set, best_model)
        result['feature_importance'] = get_feature_importance(best_model, feature_list)
        plot_tree(best_model, i, feature_list, context['arg'].out_dir)
        result['best_model'] = best_model
        result_set.append(result)
        # round_report(result)
        # result_set.append(result)
    final_report(result_set, context['arg'].out_dir)


if __name__ == '__main__':
    main()
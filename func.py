#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 12:37:32 2024

@author: michal
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from imblearn.metrics import specificity_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from simple_colors import *
import kennard_stone as ks
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score, auc, roc_curve
# import shap
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import matthews_corrcoef
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.model_selection import cross_val_score, LeaveOneOut
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn import tree
from applicability_domain import ApplicabilityDomainDetector
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, VotingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE
import lightgbm as lgb
import shap
import xgboost as xgb
from sklearn.dummy import DummyClassifier
#%%

def domain_applcability(X_train, X_test, y_train, y_test):
    
    
# =============================================================================
#   Applicability Domain based on https://doi.org/10.1016/j.chemolab.2015.04.013
# =============================================================================

    names_train = []
    names_test = []
    
    train_arr = np.zeros(X_train.shape)
    test_arr = np.zeros(X_test.shape)   
    list_tr_delate = []    
    list_test_delate = []
    for i in range(X_train.shape[1]):
        for k in range(X_train.shape[0]):
            Ski = (X_train.iloc[k,i]-X_train.mean()[i])/X_train.std()[i]
            train_arr[k,i] = Ski
            
    for i in range(X_test.shape[1]):
        for k in range(X_test.shape[0]):
            test_s = (X_test.iloc[k,i]-X_train.mean()[i])/X_train.std()[i]
            test_arr[k,i] = test_s
    
    for _ in range(len(train_arr)):
        if max(train_arr[_]) > 3 and min(train_arr[_]) > 3:
            list_tr_delate.append(_)
        elif max(train_arr[_]) > 3 and min(train_arr[_]) < 3:
            new_S_train = np.mean(train_arr[_]) + 1.28 * np.std(train_arr[_])

            if new_S_train > 3:
                list_tr_delate.append(_)
        
    train_arr = np.delete(train_arr, list_tr_delate, 0)
        
    for _ in range(len(test_arr)):
        if max(test_arr[_]) > 3 and min(test_arr[_]) > 3:
            list_test_delate.append(_)
        elif max(test_arr[_]) > 3 and min(test_arr[_]) < 3:
            new_S_test = np.mean(test_arr[_]) + 1.28 * np.std(test_arr[_])

            if new_S_test > 3:
                list_test_delate.append(_)
            
    test_arr = np.delete(test_arr, list_test_delate, 0)
    # print(list_tr_delate, list_test_delate)    
    
    names_train.append(list(X_train.index[list_tr_delate]))
    names_test.append(list(X_test.index[list_test_delate]))
    
    X_train.drop(index=names_train[0], axis=0, inplace= True)
    X_test.drop(index=names_test[0], axis=0, inplace = True)
    
    y_train.drop(index=names_train[0], axis=0, inplace= True)
    y_test.drop(index=names_test[0], axis=0, inplace = True)
    
    print(names_train, names_test)
    
    return names_train[0], names_test[0]


def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):    
    #creating a set of all the unique classes using the actual class list
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        
        #creating a list of all the classes except the current class 
        other_class = [x for x in unique_class if x != per_class]

        #marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        #using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
        roc_auc_dict[per_class] = roc_auc

    return roc_auc_dict


def kennard_stone(X, n_select):
    """
    Implements the Kennard-Stone algorithm to select n_select representative 
    points from X for the training set. The rest of the points are placed in 
    the test/validation set.

    Parameters
    ----------
    X : ndarray of shape (N, D)
        The dataset of N samples, each with D features.
    n_select : int
        The number of points to select for the training set.

    Returns
    -------
    train_indices : list
        Indices of the samples in the training set.
    test_indices : list
        Indices of the samples in the test/validation set.
    """

    # Step 1: Compute the pairwise distance matrix
    distances = pairwise_distances(X, X)

    # Step 2: Select the two points that are farthest apart
    # np.unravel_index converts the single index from argmax to row,col indices
    i, j = np.unravel_index(np.argmax(distances), distances.shape)
    train_indices = [i, j]

    # List of candidate indices (excluding the two we already picked)
    all_indices = set(range(len(X)))
    remaining_indices = list(all_indices - set(train_indices))

    # Step 3: Iteratively select points
    while len(train_indices) < n_select:
        # Compute each candidate's min distance to the points in the training set
        min_dists_to_train = [
            min(distances[c, t] for t in train_indices) 
            for c in remaining_indices
        ]

        # Select the candidate with the largest min distance to the training set
        idx_far = np.argmax(min_dists_to_train)
        new_point = remaining_indices[idx_far]

        train_indices.append(new_point)
        remaining_indices.pop(idx_far)

    # Remaining indices become the test set
    test_indices = remaining_indices

    return train_indices, test_indices



def split_data(df, target_col, st1, st2):
    validation_set = df.iloc[st1::st2].copy() 
    train_set = df.drop(validation_set.index).copy()
    X_train = train_set.drop(target_col, axis=1)
    X_val = validation_set.drop(target_col, axis=1)
    y_train = train_set[target_col]
    y_val = validation_set[target_col]
    return X_train, X_val, y_train, y_val


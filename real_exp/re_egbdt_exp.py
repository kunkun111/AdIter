# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 12:20:27 2021

@author: Administrator
"""

from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import time
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve
import os
from skmultiflow.drift_detection.adwin import ADWIN
from sklearn.ensemble import GradientBoostingRegressor
import arff
from tqdm import tqdm
#import pixiedust

from copy import deepcopy



class GBDT(object):
    def __init__(self,
                 max_iter=50,
                 sample_rate=0.8,
                 learn_rate=0.01,
                 max_depth=4,
                 new_tree_max_iter=10):

        self.max_iter = max_iter
        self.sample_rate = sample_rate # 0 < sample_rate <= 1
        self.learn_rate = learn_rate
        self.max_depth = max_depth 
        self.dtrees = []
        self.original_f = None
        self.new_tree_max_iter = new_tree_max_iter

    def fit(self, x_train, y_train):

        n, m = x_train.shape
        
        f = np.ones(n) * np.mean(y_train)
        self.original_f = np.mean(y_train)
        self.residual_mean = np.zeros(self.max_iter)
        n_sample = int(n * self.sample_rate)

        for iter_ in range(self.max_iter): 
            sample_idx = np.random.permutation(n)[:n_sample]
            x_train_subset, y_train_subset = x_train[sample_idx, :], y_train[
                sample_idx]
            y_predict_subset = np.zeros(n_sample)
            
            for j in range(n_sample):
                k = sample_idx[j]
                y_predict_subset[j] = f[k]

            residual = y_train_subset - y_predict_subset

            dtree = DecisionTreeRegressor(max_depth=self.max_depth)
            # fit to negative gradient
            dtree.fit(x_train_subset, residual * self.learn_rate)
            self.dtrees.append(dtree)  # append new tree

            # update prediction score
            for j in range(n):
                pre = dtree.predict(np.array([x_train[j]]))
                f[j] += pre

    def predict(self, x):

        n = x.shape[0]
        y = np.zeros([n, len(self.dtrees)])
        
        for iter_ in range(len(self.dtrees)):
            dtree = self.dtrees[iter_]

            y[:, iter_] = dtree.predict(x)

        init_residual = np.ones(y.shape[0]) * self.original_f
        self.cumulated_pred_score = np.cumsum(y, axis=1)
        return np.sum(y, axis=1) + init_residual.reshape(1, -1)
    
    def best_tree_purning(self, y_test):
        init_residual = np.ones(y_test.shape[0]) * self.original_f
        residual = y_test.reshape(1, -1) - init_residual.reshape(1, -1)
        residual_mat = np.repeat(residual, len(self.dtrees), axis=0).T
        tree_purning_residual = np.abs(residual_mat - self.cumulated_pred_score)
        tree_purning_residual = np.mean(tree_purning_residual, axis=0)
        tree_purning_idx = np.argmin(tree_purning_residual)
        self.dtrees = self.dtrees[:tree_purning_idx+1]
        self.max_iter = len(self.dtrees)
        
    def incremental_fit(self, x_test, y_test, pred_score, new_tree_max_iter):
        
        n, m = x_test.shape       
        f = pred_score
        n_sample = int(n*self.sample_rate)
        
        for iter_ in range(new_tree_max_iter):
            
            sample_idx = np.random.permutation(n)[:n_sample]
            
            y_residual = y_test - f
            x_train_subset, residual_train_subset = x_test[sample_idx, :], y_residual[sample_idx]
            
            new_tree = DecisionTreeRegressor(max_depth = self.max_depth)
            new_tree.fit(x_train_subset, residual_train_subset * self.learn_rate)
            self.dtrees.append(new_tree)
            self.max_iter += 1
            
            for j in range(n):
                pre = new_tree.predict(np.array([x_test[j]]))
                f[j] += pre


# Load data
def load_arff(path, dataset_name, num_copy):
    if num_copy == -1:
        file_path = path + dataset_name + '/'+ dataset_name + '.arff'
        dataset = arff.load(open(file_path), encode_nominal=True)
    else:
        file_path = path + dataset_name + '/'+ dataset_name + str(num_copy) + '.arff'
        dataset = arff.load(open(file_path), encode_nominal=True)
    return np.array(dataset["data"])


# eGBDT
def evaluation_eGBDT(data, ini_train_size, win_size, max_tree, num_ince_tree, **GBDT_pram):

    x_train = data[0:ini_train_size, :-1]
    y_train = data[0:ini_train_size, -1]
    model = GBDT(**GBDT_pram)
    model.fit(x_train, y_train)

    kf = KFold(int((data.shape[0] - ini_train_size) / win_size))
    stream = data[ini_train_size:, :]
    pred = np.zeros(stream.shape[0])
    accuracy = []
    f1 = []
    prune_tree = []#kun
    tree_before_purning = []
    tree_after_purning =[]

    for train_index, test_index in tqdm(kf.split(stream), total=kf.get_n_splits(), desc="#batch"):

        x_test = stream[test_index, :-1]
        y_test = stream[test_index, -1]
        
        # Step 1. Make Prediction
        y_pred_score = model.predict(x_test)
        y_pred_score = np.squeeze(y_pred_score)
        y_pred_label = (y_pred_score >= 0.5)

        accuracy.append(metrics.accuracy_score(y_test, y_pred_label.T))
        f1.append(metrics.f1_score(y_test,y_pred_label.T,average='macro'))
        
        pred[test_index] = y_pred_label
        
        # Step 2. Purning GBDT
        num_tree_before_purning = len(model.dtrees)
        model.best_tree_purning(y_test)
        num_tree_after_purning = len(model.dtrees)
        #print(test_index[0], 'Purned Num Tree,', num_tree_before_purning - num_tree_after_purning)
        prune_tree.append(num_tree_before_purning - num_tree_after_purning)#kun
        tree_before_purning.append(num_tree_before_purning)
        tree_after_purning.append(num_tree_after_purning)
        
        # Step 3. Update GBDT
        # Step 3.1 Drift Detection, If num_tree < num_base
        if num_tree_after_purning < GBDT_pram['max_iter']:
            model = GBDT(**GBDT_pram)
            model.fit(x_test, y_test)
            GBDT_ensemble_dict = {}
            last_best = 1
        else:
            # Step 3.2 Incremental Update with Fixed Number of Trees
            if len(model.dtrees) <= max_tree:
                y_pred_score = np.squeeze(model.predict(x_test))
                model.incremental_fit(x_test, y_test, y_pred_score, num_ince_tree)
    tqdm.write('Num tree at the end,' + str(len(model.dtrees)))
    
    return accuracy, f1, pred



# realworld experiment
def exp_realworld(path, dataset_name, num_run, exp_function, **exp_parm):

    aver_total_acc = np.zeros(num_run)
    aver_total_f1 = np.zeros(num_run)

    for r_seed in range(0, num_run):
        
        np.random.seed(r_seed)
        
        data = load_arff(path, dataset_name, -1)
        num_eval = int((data.shape[0] - exp_parm['ini_train_size']) / exp_parm['win_size'])
        
        tqdm.write('='*20)
        tqdm.write((dataset_name + str(r_seed)).center(20))
        
        batch_acc = np.zeros([num_run, num_eval])
        batch_f1 = np.zeros([num_run, num_eval])
        
        batch_acc[r_seed], batch_f1[r_seed], pred = exp_function(data, **exp_parm)
        
        aver_total_acc[r_seed] = metrics.accuracy_score(data[exp_parm['ini_train_size']:, -1], pred)
        tqdm.write('Current r_seed acc,' + str(aver_total_acc[r_seed]))
        
        aver_total_f1[r_seed] = metrics.f1_score(data[exp_parm['ini_train_size']:, -1], pred,average='macro')
        tqdm.write('Current r_seed f1,' + str(aver_total_f1[r_seed]))
        
        # Save label
        print(pred.shape)
        print(data[exp_parm['ini_train_size']:, -1].shape)
        result = np.zeros([pred.shape[0], 2])
        result[:, 0] = pred
        result[:, 1] = data[exp_parm['ini_train_size']:, -1]
        np.savetxt(str(dataset_name)+'_egbdt_'+str(r_seed)+'.out', result , delimiter=',')
        
        
    print('************************************************')
    tqdm.write('Average acc,' + str(np.mean(aver_total_acc)))
    tqdm.write('Std acc,' + str(np.std(aver_total_acc)))
    
    tqdm.write('Average f1,' + str(np.mean(aver_total_f1)))
    tqdm.write('Std f1,' + str(np.std(aver_total_f1)))
              


if __name__ == '__main__': 
    
    path = '/Realworld Data/'
    num_run = 30
    datasets = ['usenet1', 'usenet2', 'weather', 'spam_corpus_x2_feature_selected', 'elecNorm']
    chunk_size = [40, 40, 365, 100, 100]
    
    # datasets = ['airline']
    # chunk_size = [100]
    
    # initial parameter
    for i in range(len(datasets)):
        
        print (datasets[i])
        print (chunk_size[i])
        
        eGBDT_parm = {
            'ini_train_size': chunk_size[i],
            'win_size': chunk_size[i],
            'max_tree': 10000,
            'num_ince_tree':25
        }
        
        GBDT_pram = {
            'max_iter': 200,
            'sample_rate': 0.8,
            'learn_rate': 0.01,
            'max_depth': 4
        }
        
        eGBDT_parm.update(GBDT_pram)
        dataset_name = str(datasets[i])
        exp_realworld(path, dataset_name, num_run, evaluation_eGBDT,
                      **eGBDT_parm)
    
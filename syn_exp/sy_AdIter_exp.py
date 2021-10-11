# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 11:05:17 2021

@author: Kun Wang
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
        self.sample_rate = sample_rate 
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
                
                
# load data
def load_arff(path, dataset_name, num_copy):
    if num_copy == -1:
        file_path = path + dataset_name + '/'+ dataset_name + '.arff'
        dataset = arff.load(open(file_path), encode_nominal=True)
    else:
        file_path = path + dataset_name + '/'+ dataset_name + str(num_copy) + '.arff'
        dataset = arff.load(open(file_path), encode_nominal=True)
    return np.array(dataset["data"])


# AdIter
def evaluation_eGBDT_ensemble(data, ini_train_size, win_size, max_tree,
                              max_num_inc_tree, gap_num_inc_tree, **GBDT_pram):

    #data = data[0:20000,:]
    
    x_train = data[0:ini_train_size, :-1]
    y_train = data[0:ini_train_size, -1]

    eGBDT_dict = {}
    for i in range(25, max_num_inc_tree, gap_num_inc_tree):
        eGBDT_dict[i] = GBDT(**GBDT_pram)
        eGBDT_dict[i].fit(x_train, y_train)

    kf = KFold(int((data.shape[0] - ini_train_size) / win_size))
    stream = data[ini_train_size:, :]
    pred = np.zeros(stream.shape[0])
    accuracy = []
    f1 = []
    

    for train_index, test_index in tqdm(kf.split(stream),
                                        total=kf.get_n_splits(),
                                        desc="#batch"):
        x_test = stream[test_index, :-1]
        y_test = stream[test_index, -1]

        y_pred_score_aver = np.zeros(x_test.shape[0])
        for ie in eGBDT_dict.keys():
            y_pred_score = eGBDT_dict[ie].predict(x_test)
            y_pred_score = np.squeeze(y_pred_score)
            y_pred_score_aver = y_pred_score_aver + y_pred_score
            y_pred_label = (y_pred_score >= 0.5)

        y_pred_score_aver = y_pred_score_aver / len(eGBDT_dict)
        y_pred_label = (y_pred_score_aver >= 0.5)

        accuracy.append(metrics.accuracy_score(y_test, y_pred_label.T))
        f1.append(metrics.f1_score(y_test,y_pred_label.T,average='macro'))
        
        pred[test_index] = y_pred_label

        for ie in eGBDT_dict.keys():

            eGBDT_dict[ie].best_tree_purning(y_test)

            if len(eGBDT_dict[ie].dtrees) < GBDT_pram['max_iter']:
                eGBDT_dict[ie] = GBDT(**GBDT_pram)
                eGBDT_dict[ie].fit(x_test, y_test)
            else:
                if len(eGBDT_dict[ie].dtrees) <= max_tree:
                    y_pred_score = np.squeeze(eGBDT_dict[ie].predict(x_test))
                    eGBDT_dict[ie].incremental_fit(x_test, y_test,
                                                   y_pred_score, ie)
    return accuracy, f1, pred


# Run Synthetic Experiment
def exp_synthetic(path, dataset_name, num_run, num_eval, exp_function,
                  **exp_parm):
    
    print(str(dataset_name))
    np.random.seed(0)
    batch_acc = np.zeros([num_run, num_eval])

    for num_copy in range(num_run):
        print(num_copy, '/', num_run)
        data = load_arff(path, dataset_name, num_copy)
        acc, f1,pred = exp_function(data, **exp_parm)
        batch_acc[num_copy] = acc

        print(
            'Total acc, ',
            metrics.accuracy_score(data[exp_parm['ini_train_size']:, -1],
                                   pred))
        
        result = np.zeros([pred.shape[0], 2])
        result[:, 0] = pred
        result[:, 1] = data[exp_parm['ini_train_size']:, -1]
        np.savetxt(str(dataset_name)+'_AdIter_'+str(num_copy)+'.out', result , delimiter=',')
    
    print("%4f" % (batch_acc.mean()))
    print("%4f" % (batch_acc.mean(axis=1).std()))
    
   

if __name__=='__main__':
    
    path = '/Synthetic Data/'
    num_run = 1
    num_eval = 99
    
    #initial parameter
    eGBDT_ensemble_parm = {
        'ini_train_size': 100,
        'win_size': 100,
        'max_tree': 10000,
        'max_num_inc_tree': 150,  # {25，50，75，100，125}，150 exit
        'gap_num_inc_tree': 25
    }
    
    GBDT_pram = {
        'max_iter': 200,
        'sample_rate': 0.8,
        'learn_rate': 0.01,
        'max_depth': 4,
    }
    
    eGBDT_ensemble_parm.update(GBDT_pram)
    dataset_name = 'SEAa'
    exp_synthetic(path, dataset_name, num_run, num_eval, evaluation_eGBDT_ensemble, **eGBDT_ensemble_parm)
    
    eGBDT_ensemble_parm.update(GBDT_pram)
    idataset_name = 'RTG'
    exp_synthetic(path, dataset_name, num_run, num_eval, evaluation_eGBDT_ensemble, **eGBDT_ensemble_parm)
    
    eGBDT_ensemble_parm.update(GBDT_pram)
    dataset_name = 'RBF'
    exp_synthetic(path, dataset_name, num_run, num_eval, evaluation_eGBDT_ensemble, **eGBDT_ensemble_parm)
    
    eGBDT_ensemble_parm.update(GBDT_pram)
    dataset_name = 'RBFr'
    exp_synthetic(path, dataset_name, num_run, num_eval, evaluation_eGBDT_ensemble, **eGBDT_ensemble_parm)
    
    eGBDT_ensemble_parm.update(GBDT_pram)
    dataset_name = 'AGRa'
    exp_synthetic(path, dataset_name, num_run, num_eval, evaluation_eGBDT_ensemble, **eGBDT_ensemble_parm)
    
    eGBDT_ensemble_parm.update(GBDT_pram)
    dataset_name = 'HYP'
    exp_synthetic(path, dataset_name, num_run, num_eval, evaluation_eGBDT_ensemble, **eGBDT_ensemble_parm)
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 12:45:25 2020

@author: kunwang
"""

# Imports

from skmultiflow.data import SEAGenerator
from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.meta import LearnPPNSEClassifier
from skmultiflow.meta import AccuracyWeightedEnsembleClassifier
from skmultiflow.meta import LeveragingBaggingClassifier
from skmultiflow.meta import OnlineBoostingClassifier
from skmultiflow.meta import OnlineRUSBoostClassifier
from skmultiflow.meta import OzaBaggingClassifier
from skmultiflow.evaluation import EvaluatePrequential
import numpy as np
import arff
import pandas as pd
from skmultiflow.data.data_stream import DataStream
from sklearn.metrics import confusion_matrix, precision_score, matthews_corrcoef, accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve


# load .arff dataset
def load_arff(path, dataset_name):
    file_path = path + dataset_name + '/'+ dataset_name + '.arff'
    dataset = arff.load(open(file_path), encode_nominal=True)
    return pd.DataFrame(dataset["data"])


def ARF_run (dataset_name, batch):
    data = load_arff(path, dataset_name)
    # data = data.head(300)
    #print (data.shape)
    # data transform
    stream = DataStream(data)
    #print(stream)

    # Setup variables to control loop and track performance
    n_samples = 0
    max_samples = data.shape[0]

    # Train the classifier with the samples provided by the data stream
    pred = np.empty(0)
    np.random.seed(0)
    
    model = AdaptiveRandomForestClassifier()
    while n_samples < max_samples and stream.has_more_samples():
        X, y = stream.next_sample(batch)
        y_pred = model.predict(X)
        pred = np.hstack((pred,y_pred))
        model.partial_fit(X, y,stream.target_values)
        n_samples += batch

    # evaluate
    data = data.values
    Y = data[:,-1]
    acc = accuracy_score(Y[batch:], pred[batch:])
    f1 = f1_score(Y[batch:], pred[batch:], average='macro')
    mcc = matthews_corrcoef(Y[batch:], pred[batch:])
    #print (Y[batch:].shape, pred[batch:].shape)
    print("acc:",acc)
    print("f1:",f1)
    print("mcc:",mcc)
    
    # save results
    result = np.zeros([pred[batch:].shape[0], 2])
    result[:, 0] = pred[batch:]
    result[:, 1] = Y[batch:]
    np.savetxt(dataset_name +'_'+ 'ARF.out', result, delimiter=',')
    
    
    
def NSE_run (dataset_name, batch):
    data = load_arff(path, dataset_name)
    # data = data.head(300)
    #print (data.shape)
    # data transform
    stream = DataStream(data)
    #print(stream)

    # Setup variables to control loop and track performance
    n_samples = 0
    max_samples = data.shape[0]

    # Train the classifier with the samples provided by the data stream
    pred = np.empty(0)
    np.random.seed(0)
    
    model = LearnPPNSEClassifier()
    while n_samples < max_samples and stream.has_more_samples():
        X, y = stream.next_sample(batch)
        y_pred = model.predict(X)
        pred = np.hstack((pred,y_pred))
        model.partial_fit(X, y,stream.target_values)
        n_samples += batch

    # evaluate
    data = data.values
    Y = data[:,-1]
    acc = accuracy_score(Y[batch:], pred[batch:])
    f1 = f1_score(Y[batch:], pred[batch:], average='macro')
    mcc = matthews_corrcoef(Y[batch:], pred[batch:])
    #print (Y[batch:].shape, pred[batch:].shape)
    print("acc:",acc)
    print("f1:",f1)
    print("mcc:",mcc)
    
    # save results
    result = np.zeros([pred[batch:].shape[0], 2])
    result[:, 0] = pred[batch:]
    result[:, 1] = Y[batch:]
    np.savetxt(dataset_name +'_'+ 'NSE.out', result, delimiter=',')
    
    
def AWE_run (dataset_name, batch):
    data = load_arff(path, dataset_name)
    # data = data.head(300)
    #print (data.shape)
    # data transform
    stream = DataStream(data)
    #print(stream)

    # Setup variables to control loop and track performance
    n_samples = 0
    max_samples = data.shape[0]

    # Train the classifier with the samples provided by the data stream
    pred = np.empty(0)
    np.random.seed(0)
    
    #model = OnlineAdaC2Classifier(n_estimators=24)
    model = AccuracyWeightedEnsembleClassifier()
    while n_samples < max_samples and stream.has_more_samples():
        X, y = stream.next_sample(batch)
        y_pred = model.predict(X)
        pred = np.hstack((pred,y_pred))
        model.partial_fit(X, y,stream.target_values)
        n_samples += batch

    # evaluate
    data = data.values
    Y = data[:,-1]
    acc = accuracy_score(Y[batch:], pred[batch:])
    f1 = f1_score(Y[batch:], pred[batch:], average='macro')
    mcc = matthews_corrcoef(Y[batch:], pred[batch:])
    #print (Y[batch:].shape, pred[batch:].shape)
    print("acc:",acc)
    print("f1:",f1)
    print("mcc:",mcc)
    
    # save results
    result = np.zeros([pred[batch:].shape[0], 2])
    result[:, 0] = pred[batch:]
    result[:, 1] = Y[batch:]
    np.savetxt(dataset_name +'_'+ 'AWE.out', result, delimiter=',')
    
    
    
def LEV_run (dataset_name, batch):
    data = load_arff(path, dataset_name)
    # data = data.head(300)
    #print (data.shape)
    # data transform
    stream = DataStream(data)
    #print(stream)

    # Setup variables to control loop and track performance
    n_samples = 0
    max_samples = data.shape[0]

    # Train the classifier with the samples provided by the data stream
    pred = np.empty(0)
    np.random.seed(0)
    
    model = LeveragingBaggingClassifier()
    while n_samples < max_samples and stream.has_more_samples():
        X, y = stream.next_sample(batch)
        y_pred = model.predict(X)
        pred = np.hstack((pred,y_pred))
        model.partial_fit(X, y,stream.target_values)
        n_samples += batch

    # evaluate
    data = data.values
    Y = data[:,-1]
    acc = accuracy_score(Y[batch:], pred[batch:])
    f1 = f1_score(Y[batch:], pred[batch:], average='macro')
    mcc = matthews_corrcoef(Y[batch:], pred[batch:])
    #print (Y[batch:].shape, pred[batch:].shape)
    print("acc:",acc)
    print("f1:",f1)
    print("mcc:",mcc)
    
    # save results
    result = np.zeros([pred[batch:].shape[0], 2])
    result[:, 0] = pred[batch:]
    result[:, 1] = Y[batch:]
    np.savetxt(dataset_name +'_'+ 'LEV.out', result, delimiter=',')
    
    
def OBC_run (dataset_name, batch):
    data = load_arff(path, dataset_name)
    # data = data.head(300)
    #print (data.shape)
    # data transform
    stream = DataStream(data)
    #print(stream)

    # Setup variables to control loop and track performance
    n_samples = 0
    max_samples = data.shape[0]

    # Train the classifier with the samples provided by the data stream
    pred = np.empty(0)
    np.random.seed(0)
    
    model1 = OnlineBoostingClassifier()
    while n_samples < max_samples and stream.has_more_samples():
        X, y = stream.next_sample(batch)
        y_pred = model1.predict(X)
        pred = np.hstack((pred,y_pred))
        model1.partial_fit(X, y,stream.target_values)
        n_samples += batch

    # evaluate
    data = data.values
    Y = data[:,-1]
    acc = accuracy_score(Y[batch:], pred[batch:])
    f1 = f1_score(Y[batch:], pred[batch:], average='macro')
    mcc = matthews_corrcoef(Y[batch:], pred[batch:])
    #print (Y[batch:].shape, pred[batch:].shape)
    print("acc:",acc)
    print("f1:",f1)
    print("mcc:",mcc)
    
    # save results
    result = np.zeros([pred[batch:].shape[0], 2])
    result[:, 0] = pred[batch:]
    result[:, 1] = Y[batch:]
    np.savetxt(dataset_name +'_'+ 'OBC.out', result, delimiter=',')
    

def RUS_run (dataset_name, batch):
    data = load_arff(path, dataset_name)
    # data = data.head(300)
    #print (data.shape)
    # data transform
    stream = DataStream(data)
    #print(stream)

    # Setup variables to control loop and track performance
    n_samples = 0
    max_samples = data.shape[0]

    # Train the classifier with the samples provided by the data stream
    pred = np.empty(0)
    np.random.seed(0)
    
    model = OnlineRUSBoostClassifier()
    while n_samples < max_samples and stream.has_more_samples():
        X, y = stream.next_sample(batch)
        y_pred = model.predict(X)
        pred = np.hstack((pred,y_pred))
        model.partial_fit(X, y,stream.target_values)
        n_samples += batch

    # evaluate
    data = data.values
    Y = data[:,-1]
    acc = accuracy_score(Y[batch:], pred[batch:])
    f1 = f1_score(Y[batch:], pred[batch:], average='macro')
    mcc = matthews_corrcoef(Y[batch:], pred[batch:])
    #print (Y[batch:].shape, pred[batch:].shape)
    print("acc:",acc)
    print("f1:",f1)
    print("mcc:",mcc)
    
    # save results
    result = np.zeros([pred[batch:].shape[0], 2])
    result[:, 0] = pred[batch:]
    result[:, 1] = Y[batch:]
    np.savetxt(dataset_name +'_'+ 'RUS.out', result, delimiter=',')
    
    
    
def OZA_run (dataset_name, batch):
    data = load_arff(path, dataset_name)
    # data = data.head(300)
    #print (data.shape)
    # data transform
    stream = DataStream(data)
    #print(stream)

    # Setup variables to control loop and track performance
    n_samples = 0
    max_samples = data.shape[0]

    # Train the classifier with the samples provided by the data stream
    pred = np.empty(0)
    np.random.seed(0)
    
    model = OzaBaggingClassifier()
    while n_samples < max_samples and stream.has_more_samples():
        X, y = stream.next_sample(batch)
        y_pred = model.predict(X)
        pred = np.hstack((pred,y_pred))
        model.partial_fit(X, y,stream.target_values)
        n_samples += batch

    # evaluate
    data = data.values
    Y = data[:,-1]
    acc = accuracy_score(Y[batch:], pred[batch:])
    f1 = f1_score(Y[batch:], pred[batch:], average='macro')
    mcc = matthews_corrcoef(Y[batch:], pred[batch:])
    #print (Y[batch:].shape, pred[batch:].shape)
    print("acc:",acc)
    print("f1:",f1)
    print("mcc:",mcc)
    
    # save results
    result = np.zeros([pred[batch:].shape[0], 2])
    result[:, 0] = pred[batch:]
    result[:, 1] = Y[batch:]
    np.savetxt(dataset_name +'_'+ 'OZA.out', result, delimiter=',')

    
    
    
if __name__ == '__main__':
    
    path = '/Realworld Data/'

    datasets = ['usenet1','usenet2','weather','spam_corpus_x2_feature_selected',
                'elecNorm','airline']
    
    batch = [40,40,365,100,100,100]
    
    for i in range (len(datasets)):
        dataset_name = datasets[i]
        batch_size = batch[i]
        
        print (dataset_name, batch_size,'ARF')
        ARF_run(dataset_name, batch_size)
              
        print (dataset_name, batch_size,'NSE')
        NSE_run(dataset_name, batch_size)
        
        print (dataset_name, batch_size,'AWE')
        AWE_run(dataset_name, batch_size)
        
        print (dataset_name, batch_size,'LEV')
        LEV_run(dataset_name, batch_size)
                
        print (dataset_name, batch_size,'OBC')
        OBC_run(dataset_name, batch_size)
                 
        print (dataset_name, batch_size,'RUS')
        RUS_run(dataset_name, batch_size ) 
        
        print (dataset_name, batch_size,'OZA')
        OZA_run(dataset_name, batch_size ) 

    
  
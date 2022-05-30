_EPSILON = 1e-08

import numpy as np
import pandas as pd
import random
import os,sys
import pickle


from sklearn.model_selection import train_test_split


from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest

class CoxPH:
    """ A parent class for all survival estimators from sksuv. Particular survival models will inherit from this class."""
    # methods
    def __init__(self, alpha=0.01):
        self.model = CoxPHSurvivalAnalysis(alpha=alpha)
        
    def fit(self,X,T,Y):
        # Put the data in the proper format # check data type first
        y = [(Y[i,0], T[i,0]) for i in range(len(Y))]
        y = np.array(y, dtype=[('status', 'bool'),('time','<f8')])
        # print(self.name)
        self.model.fit(X,y)

    def predict(self,X, time_horizons): 
        surv   = self.model.predict_survival_function(X)  #returns StepFunction object
        preds_ = np.zeros([np.shape(surv)[0], len(time_horizons)])
        for t, eval_time in enumerate(time_horizons):
            if eval_time > np.max(surv[0].x):  #all have the same maximum surv.x
                eval_time = np.max(surv[0].x)
            preds_[:, t] = np.asarray([(1. - surv[i](eval_time)) for i in range(len(surv))])  #return cif at self.median_tte
        return preds_
    
    
class RSForest:
    """ A parent class for all survival estimators from sksuv. Particular survival models will inherit from this class."""
    # methods
    def __init__(self, n_estimators=100):
        self.model = RandomSurvivalForest(n_estimators=n_estimators)
        
    def fit(self,X,T,Y):
        # Put the data in the proper format # check data type first
        y = [(Y[i,0], T[i,0]) for i in range(len(Y))]
        y = np.array(y, dtype=[('status', 'bool'),('time','<f8')])
        # print(self.name)
        self.model.fit(X,y)

    def predict(self,X, time_horizons): 
        surv   = self.model.predict_survival_function(X)  #returns StepFunction object
        preds_ = np.zeros([np.shape(surv)[0], len(time_horizons)])
        
        for t, eval_time in enumerate(time_horizons):                        
            if len(np.where(self.model.event_times_ <= eval_time)[0]) == 0:
                preds_[:, t] = 0.
            else:
                tmp_idx = np.where(self.model.event_times_ <= eval_time)[0][-1]
                preds_[:, t] = 1. - surv[:, tmp_idx]            
        return preds_
    

class LANDMARKING:
    def __init__(self, LANDMARKING_TIMES, LANDMARKING_MODE, parameters):
        self.LANDMARKING_MODE = LANDMARKING_MODE
        
        if self.LANDMARKING_MODE == 'CoxPH':
            if 'alpha' in parameters.keys():
                self.alpha = parameters['alpha']
            else:
                self.alpha = 0.001
        elif self.LANDMARKING_MODE == 'RSForest':
            if 'n_estimators' in parameters.keys():
                self.n_estimators = parameters['n_estimators']
            else:
                self.n_estimators = 100
        else:
            raise ValueError("Wrong survival model. Select {'CoxPH', 'RSForest'}")
        
        self.LANDMARKING_TIMES = LANDMARKING_TIMES 
        
    def train(self, xs_, xt_, time_, tte_, label_):        
        self.num_Event = len(np.unique(label_)) - 1
        
        self.LANDMARKING_MODELS = {}
        for e in range(self.num_Event):
            self.LANDMARKING_MODELS[e] = []
        
        for l, l_time in enumerate(self.LANDMARKING_TIMES):
            print('Training {}-th Landmarking Model at {} - #{}'.format(l+1, l_time, np.sum(tte_ > l_time)))

            condition1            = (np.sum(np.abs(xt_), axis=2, keepdims=True) > 0) #where there are measurements
            condition2            = (time_ <= l_time) #where before 
            landmarking_condition = (condition1 & condition2)[:,:,0]

            last_meas_idx = np.sum(landmarking_condition, axis=1) - 1
            
            x_dim_static       = np.shape(xs_)[-1]
            x_dim_timevarying  = np.shape(xt_)[-1]
            
            tmp_data      = np.zeros([np.shape(xt_)[0], x_dim_static+x_dim_timevarying-1])
            tmp_tte       = np.zeros(np.shape(tte_))
            tmp_label     = label_

            pat_idx = tte_[:,0] > l_time

            for i in range(np.shape(tmp_data)[0]):
                tmp_data[i, :x_dim_static] = xs_[i, :]
                tmp_data[i, x_dim_static:] = xt_[i, last_meas_idx[i], 1:]
                tmp_tte[i, 0]              = tte_[i, 0] - l_time    

            for e in range(self.num_Event):
                if self.LANDMARKING_MODE == 'CoxPH':
                    model = CoxPH(alpha=self.alpha)
                elif self.LANDMARKING_MODE == 'RSForest':
                    model = RSForest(n_estimators=self.n_estimators)
                    
                model.fit(tmp_data[pat_idx], tmp_tte[pat_idx], (tmp_label[pat_idx] == e+1).astype(int))

                if e == 0:
                    self.LANDMARKING_MODELS[e].append(model)
                else:
                    self.LANDMARKING_MODELS[e].append(model)
                    
                    
    def predict(self, xs_, xt_, time_, EVAL_TIMES):
        landmarking_index = -1 * np.ones([np.shape(xt_)[0]])

        for l in range(len(self.LANDMARKING_TIMES)):
            if l == 0:
                l_curr = self.LANDMARKING_TIMES[l]
                condition = (np.max(time_[:,:,0], axis=1) <= l_curr)

            elif l == (len(self.LANDMARKING_TIMES) - 1):
                l_prev = self.LANDMARKING_TIMES[l-1]
                condition = (np.max(time_[:,:,0], axis=1) > l_prev)

            else:
                l_prev = self.LANDMARKING_TIMES[l-1]
                l_curr = self.LANDMARKING_TIMES[l]
                condition = (np.max(time_[:,:,0], axis=1) > l_prev) & (np.max(time_[:,:,0], axis=1) <= l_curr)

            landmarking_index[condition] = l

        pred_ = np.zeros([np.shape(xt_)[0], self.num_Event, len(EVAL_TIMES)])

        for l in range(len(self.LANDMARKING_TIMES)):
            l_time = self.LANDMARKING_TIMES[l]

            for i in range(np.shape(xt_)[0]):
                if landmarking_index[i] == l:
                    condition1            = (np.sum(np.abs(xt_[i, :, :]), axis=-1) > 0) #where there are measurements
                    condition2            = (time_[i, :, 0] <= l_time) #where before 
                    last_meas_idx = np.sum(condition1 & condition2) - 1

                    tmp_data = np.concatenate([xs_[[i], :], xt_[[i], last_meas_idx, 1:]], axis=1)

                    for e in range(self.num_Event):
                        pred_[i, e, :] = self.LANDMARKING_MODELS[e][l].predict(tmp_data, EVAL_TIMES)
                        
        return pred_
    
    
    
    def predict_predtime(self, xs_, xt_, time_, pred_time_, EVAL_TIMES):
        landmarking_index = [l_idx for l_idx, l_time in enumerate(self.LANDMARKING_TIMES) if l_time <= pred_time_][-1]
        
        pred_ = np.zeros([np.shape(xt_)[0], self.num_Event, len(EVAL_TIMES)])

        l_time = self.LANDMARKING_TIMES[landmarking_index]

        for i in range(np.shape(xt_)[0]):
            condition1            = (np.sum(np.abs(xt_[i, :, :]), axis=-1) > 0) #where there are measurements
            condition2            = (time_[i, :, 0] <= l_time) #where before 
            last_meas_idx = np.sum(condition1 & condition2) - 1

            tmp_data = np.concatenate([xs_[[i], :], xt_[[i], last_meas_idx, 1:]], axis=1)

            for e in range(self.num_Event):
                pred_[i, e, :] = self.LANDMARKING_MODELS[e][landmarking_index].predict(tmp_data, EVAL_TIMES)

        return pred_
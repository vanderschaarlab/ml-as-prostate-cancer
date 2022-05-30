import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os

from sksurv.metrics import concordance_index_ipcw, concordance_index_censored, brier_score


def save_logging(dictionary, log_name):
    with open(log_name, 'w') as f:
        for key, value in dictionary.items():
            if 'active_fn' in key:
                if 'relu' in str(value):
                    value = 'relu'
                if 'elu' in str(value):
                    value = 'elu'  
                if 'tanh' in str(value):
                    value = 'tanh'
            if 'init' in key:
                if 'variance_scaling_initializer' in str(value):
                    value = 'xavier'
                
            f.write('%s:%s\n' % (key, value))

# this open can calls the saved hyperparameters
def load_logging(filename):
    data = dict()
    with open(filename) as f:
        def is_float(input):
            try:
                num = float(input)
            except ValueError:
                return False
            return True

        for line in f.readlines():
            if ':' in line:
                key,value = line.strip().split(':', 1)
                if value.isdigit():
                    data[key] = int(value)
                elif is_float(value):
                    data[key] = float(value)
                elif value == 'None':
                    data[key] = None
                else:
                    data[key] = value
            else:
                pass # deal with bad lines of text here    
    return data


def f_get_minibatch(mb_size, x1, x2, x3=None, x4=None, x5=None):
    idx = range(np.shape(x1)[0])
    idx = random.sample(idx, mb_size)

    x1_mb    = x1[idx].astype(float)
    x2_mb    = x2[idx].astype(float)
    
    if x3 is not None:
        x3_mb     = x3[idx].astype(float)
    
    if x4 is not None:
        x4_mb   = x4[idx].astype(float)
    
    if x5 is not None:
        x5_mb     = x5[idx].astype(float) 

    if x3 is None:
        return x1_mb, x2_mb
    else:
        if x4 is None:
            return x1_mb, x2_mb, x3_mb
        else:
            if x5 is None:
                return x1_mb, x2_mb, x3_mb, x4_mb
            else:
                return x1_mb, x2_mb, x3_mb, x4_mb, x5_mb
            



def f_get_prediction(model_, xs_, xt_, time_, EVAL_TIMES_, pred_time_ = None):

    if isinstance(EVAL_TIMES_, list) == False:
        EVAL_TIMES_ = [EVAL_TIMES_] #make it as a list

    if pred_time_ == None:
        tmp_idx = (time_ <= 1e8) & (np.sum(np.abs(xt_), axis=2, keepdims=True) > 0)
        pat_idx = np.sum(tmp_idx[:,:,0], axis=1) != 0

        new_time   = time_
        new_data_t = xt_

    else:
        tmp_idx = (time_ <= pred_time_) & (np.sum(np.abs(xt_), axis=2, keepdims=True) > 0)
        pat_idx = np.sum(tmp_idx[:,:,0], axis=1) != 0

        new_time    = np.zeros(np.shape(time_))
        new_data_t    = np.zeros(np.shape(xt_))

        new_time[tmp_idx] = time_[tmp_idx]
        new_data_t[np.tile(tmp_idx, [1,1,np.shape(xt_)[2]])] = xt_[np.tile(tmp_idx, [1,1,np.shape(xt_)[2]])]

    for e_idx, eval_time in enumerate(EVAL_TIMES_):
        if e_idx == 0:
            num_Event = np.shape(model_.get_risk(xs_, new_data_t, eval_time*np.ones([np.shape(xt_)[0], 1])))[-1]
            pred = np.zeros([np.shape(xt_)[0], num_Event, len(EVAL_TIMES_)])
        pred[:, :, e_idx] = model_.get_risk(xs_, new_data_t, eval_time*np.ones([np.shape(xt_)[0], 1]))

    pred[~pat_idx, :, :] = -1  #if no observation -> no prediction indicator

    return (pred, pat_idx)



def evaluate(model, PRED_TIMES, EVAL_TIMES, TR_TTE, TE_DATA):
    (tr_label, tr_tte)                                = TR_TTE
    (te_data_s, te_data_t, te_time, te_label, te_tte) = TE_DATA
    
    CINDEX = np.zeros([len(PRED_TIMES), len(EVAL_TIMES)])
    BRIERSCORE = np.zeros([len(PRED_TIMES), len(EVAL_TIMES)])

    for p_idx, pred_time in enumerate(PRED_TIMES):
        p_idx_te  = te_tte >= pred_time
        p_idx_tr  = tr_tte >= pred_time

        tmp_pred, _ = f_get_prediction(model, te_data_s, te_data_t, te_time, EVAL_TIMES, pred_time_= pred_time)
               
        tmp_y = tr_label[p_idx_tr]
        tmp_t = tr_tte[p_idx_tr] - pred_time

        tr_y_structured =  [(tmp_y[i], tmp_t[i]) for i in range(len(tmp_y))]
        tr_y_structured = np.array(tr_y_structured, dtype=[('status', 'bool'),('time','<f8')])

        tmp_y = te_label[p_idx_te]
        tmp_t = te_tte[p_idx_te] - pred_time

        te_y_structured =  [(tmp_y[i], tmp_t[i]) for i in range(len(tmp_y))]
        te_y_structured = np.array(te_y_structured, dtype=[('status', 'bool'),('time','<f8')])


        for e_idx, eval_time in enumerate(EVAL_TIMES):
            if np.sum((tmp_t<=eval_time) & (tmp_y==1)) > 0:
                CINDEX[p_idx, e_idx] = concordance_index_ipcw(tr_y_structured, te_y_structured, tmp_pred[p_idx_te][:, e_idx], tau=eval_time)[0]
                if np.max(tmp_t) > eval_time:
                    BRIERSCORE[p_idx, e_idx] = brier_score(tr_y_structured, te_y_structured, 1.- tmp_pred[p_idx_te][:, e_idx], times=eval_time)[1][0]

    return CINDEX, BRIERSCORE
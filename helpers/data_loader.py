import numpy as np
import pandas as pd
import random

import os,sys

from sklearn.preprocessing import MinMaxScaler

def import_dataset(datapath):
    df            = pd.read_csv(datapath)

    feat_list     = [feat for feat in list(df) if feat not in ['id', 'tte', 'times', 'label']]
    scaler        = MinMaxScaler()

    df[feat_list] = scaler.fit_transform(df[feat_list])

    id_list     = df['id'].unique()
    grouped     = df.groupby(by='id')

    tmp         = grouped.count()
    num_samples = len(tmp)
    max_length  = tmp.max()[0]

    feat_timevarying = []

    for feat in feat_list:
        for i in id_list:
            if len(grouped.get_group(i).value_counts(feat).values) != 1:
                feat_timevarying += [feat]
                break

    feat_static = [feat for feat in feat_list if feat not in feat_timevarying]

    data_xs       = np.zeros([num_samples, len(feat_static)])
    data_xs[:, :] = np.asarray(df.drop_duplicates(subset='id')[feat_static])

    data_y        = np.zeros([num_samples, 1])
    data_y[:, 0]  = np.asarray(df.drop_duplicates(subset='id')['label'])

    data_tte        = np.zeros([num_samples, 1])
    data_tte[:, 0]  = np.asarray(df.drop_duplicates(subset='id')['tte'])

    data_xt       = np.zeros([num_samples, max_length, len(feat_timevarying)+1]) #inlcuding deltas 
    data_time     = np.zeros([num_samples, max_length, 1])

    for i, pid in enumerate(id_list):
        tmp                        = grouped.get_group(pid)
        data_xt[i, 1:len(tmp), 0]  = np.asarray(tmp['times'].diff())[1:]
        data_xt[i, :len(tmp), 1:]  = np.asarray(tmp[feat_timevarying])
        data_time[i, :len(tmp), 0] = np.asarray(tmp['times']) 

    data_xt[:, :, 0] = data_xt[:, :, 0]/data_xt[:, :, 0].max() #min-max on delta's


    xt_bin_list, xt_con_list = [],[]

    for f_idx, feat in enumerate(feat_timevarying):
        if len(df[feat].unique()) == 2:
            xt_bin_list += [f_idx]
        else:
            xt_con_list += [f_idx]    
    
    feat_timevarying = ['delta'] + feat_timevarying
    
    return (data_xs, data_xt, data_time, data_y, data_tte), (feat_static, feat_timevarying), (xt_bin_list, xt_con_list)
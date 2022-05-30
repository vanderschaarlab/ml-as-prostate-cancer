import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os

from sklearn.model_selection import train_test_split

from helpers.data_loader import *
from helpers.utils_others import *

from models.model import *
from helpers.utils_others import *

import argparse

def init_arg():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/pbc2_cleaned.csv', help='path to data file', type=str)
    parser.add_argument('--save_path', default='./saved/', help='path to save models', type=str)

    parser.add_argument('--seed', default=1234, help='random seed', type=int)
    
    parser.add_argument('--mb_size', default=64, help='batchsize', type=int)
    
    parser.add_argument("--RNN_type", default='GRU', type=str, choices=['GRU', 'LSTM'])
    parser.add_argument('--num_layers_RNN', default=2, help='number of layers -- RNN', type=int)
    parser.add_argument('--h_dim_RNN', default=100, help='number of hidden nodes -- RNN', type=int)
    
    parser.add_argument('--num_layers_FC', default=3, help='number of layers -- FC', type=int)
    parser.add_argument('--h_dim_FC', default=100, help='number of hidden nodes -- FC', type=int)
    
    parser.add_argument("--lr_rate", default=1e-3, help='learning rate', type=float)
    parser.add_argument("--keep_prob", help='keep probability for dropout', default=0.6, type=float)
    parser.add_argument("--reg_scale", default=0., help='l1-regularization', type=float)

    parser.add_argument('--alpha', default=10.0, help='coefficient -- alpha', type=float)
    
    
    parser.add_argument('--pred_times', default=[0, 180, 365], help='list of prediction times', type=int)
    parser.add_argument('--eval_times', default=[180, 365], help='list of evaluation times', type=int)

    return parser.parse_args()


if __name__ == '__main__':    
    args             = init_arg()    

    data_path        = args.data_path
    save_path        = args.save_path

    PRED_TIMES       = args.pred_times
    EVAL_TIMES       = args.eval_times

    seed             = args.seed

    ##### import dataset
    (data_xs, data_xt, data_time, data_y, data_tte), \
    (feat_static, feat_timevarying), \
    (xt_bin_list, xt_con_list) = \
    import_dataset(data_path)


    x_dim_static      = len(feat_static)
    x_dim_timevarying = len(feat_timevarying) # this includes delta

    max_length                  = np.shape(data_time)[1]
    num_Event                   = len(np.unique(data_y)) - 1  #number of next outcome events

    data_y_new   = np.zeros([np.shape(data_y)[0], max_length, num_Event])
    data_tte_new = np.zeros([np.shape(data_y)[0], max_length, num_Event])

    seq_length = np.sum(np.sum(data_xt, axis=2) != 0, axis=1)

    for i in range(np.shape(data_y)[0]):
        data_y_new[i, :seq_length[i], :]   = data_y[i]
        data_tte_new[i, :seq_length[i], :] = data_tte[i]


    ##### HYPER-PARAMETERS
    iteration                   = 50000

    mb_size                     = args.mb_size
    keep_prob                   = args.keep_prob
    lr_rate                     = args.lr_rate

    alpha                       = args.alpha  #for stepahead-prediction loss


    ##### MAKE DICTIONARIES
    # INPUT DIMENSIONS
    input_dims                  = { 'x_dim_static'      : x_dim_static,
                                    'x_dim_timevarying' : x_dim_timevarying, #this includes delta
                                    'num_Event'         : num_Event,        
                                    'xt_con_list'       : xt_con_list,
                                    'xt_bin_list'       : xt_bin_list,
                                    'max_length'        : max_length }

    # NETWORK HYPER-PARMETERS
    network_settings            = { 'p_weibull'         : 1.0,
                                    'h_dim_RNN'         : args.h_dim_RNN,
                                    'h_dim_FC'          : args.h_dim_FC,
                                    'num_layers_RNN'    : args.num_layers_RNN,
                                    'num_layers_FC'     : args.num_layers_FC,
                                    'RNN_type'          : args.RNN_type,
                                    'FC_active_fn'      : tf.nn.relu,
                                    'RNN_active_fn'     : tf.nn.tanh,
                                    'initial_W'         : tf.contrib.layers.xavier_initializer(),
                                    'reg_scale'         : args.reg_scale}

    
    if not os.path.exists(save_path + '/models/'):
        os.makedirs(save_path + '/models/')

    if not os.path.exists(save_path + '/results/'):
        os.makedirs(save_path + '/results/')
        
    save_logging(network_settings, save_path + '/models/network_settings.txt')

    (tr_data_s,te_data_s, tr_data_t,te_data_t, tr_time,te_time, tr_tte,te_tte, tr_label,te_label, tr_tte_new,te_tte_new, tr_label_new,te_label_new) = train_test_split(
        data_xs, data_xt, data_time, data_tte, data_y, data_tte_new, data_y_new, test_size=0.2, random_state=seed
    ) 

    (tr_data_s,va_data_s, tr_data_t,va_data_t, tr_time,va_time, tr_tte,va_tte, tr_label,va_label, tr_tte_new,va_tte_new, tr_label_new,va_label_new) = train_test_split(
        tr_data_s, tr_data_t, tr_time, tr_tte, tr_label, tr_tte_new, tr_label_new, test_size=0.2, random_state=seed
    ) 


    tf.reset_default_graph()
    gpu_options = tf.GPUOptions()
    config      = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True

    sess        = tf.Session(config=config)
    model       = Model_DeepHit_Weibull(sess, "DDHL", input_dims, network_settings)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    min_loss       = 1e8
    max_flag       = 30
    avg_loss_tte   = 0   
    avg_loss_mle   = 0

    check_step     = 100
    stopflag       = 0


    print( "MAIN TRAINING ...")
    for itr in range(iteration):
        xs_mb, xt_mb, t_mb, tte_mb, m_mb = f_get_minibatch(
            mb_size, tr_data_s, tr_data_t, tr_time, tr_tte_new, tr_label_new
        )

        DATA       = (xs_mb, xt_mb, t_mb, tte_mb, m_mb)

        _, loss_tte, loss_mle = model.train(DATA, alpha, keep_prob, lr_rate)

        avg_loss_tte   += loss_tte/check_step
        avg_loss_mle   += loss_mle/check_step

        if (itr+1)%check_step == 0:
            stopflag += 1      

            va_loss_tte, va_loss_mle = model.get_cost((va_data_s, va_data_t, va_time, va_tte_new, va_label_new), alpha)

            print('|| Epoch:{0:05d} | TR_Loss_tte:{1:0.4f} |  TR_Loss_mle:{2:0.4f} || VA_Loss_tte:{3:0.4f} || VA_Loss_mle:{4:0.4f} ||'.format(
                itr + 1, avg_loss_tte, avg_loss_mle, va_loss_tte, va_loss_mle))

            avg_loss_tte  = 0   
            avg_loss_mle  = 0

            if min_loss > va_loss_tte:
                stopflag = 0
                min_loss = va_loss_tte

                saver.save(sess, save_path + '/models/model_tte')
                print('saved...')

                result1, result2 = evaluate(model, PRED_TIMES, EVAL_TIMES, (tr_label, tr_tte), (va_data_s, va_data_t, va_time, va_label, va_tte))
                print('validation: averaged c-index: {:0.4f}'.format(np.mean(result1)))

        if stopflag >= max_flag:
            print('model trained...')
            break

    saver.restore(sess, save_path + '/models/model_tte')


    result1, result2 = evaluate(model, PRED_TIMES, EVAL_TIMES, (tr_label, tr_tte), (te_data_s, te_data_t, te_time, te_label, te_tte))

    print('Testing Performance - C-Index:')
    print(result1)
    print('\n')
    print('Testing Performance - Brier Score:')
    print(result2)

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


ACTIVATION_FN    = {'relu': tf.nn.relu, 'elu': tf.nn.elu, 'tanh': tf.nn.tanh}
INTIALIZATION_FN = {'xavier': tf.contrib.layers.xavier_initializer()}



import argparse

def init_arg():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/pbc2_cleaned.csv', help='path to data file', type=str)
    parser.add_argument('--open_path', default='./saved/models/', help='path to saved DDHL model', type=str)
    parser.add_argument('--save_path', default='./saved/', help='path to save models', type=str)

    parser.add_argument('--seed', default=1234, help='random seed', type=int)    
    parser.add_argument('--mb_size', default=64, help='batchsize', type=int)
    
    parser.add_argument('--num_layers_S', default=3, help='number of layers -- Selector', type=int)
    parser.add_argument('--h_dim_S', default=100, help='number of hidden nodes -- Selector', type=int)
    
    parser.add_argument('--K', default=10, help='maximum number of clusters', type=int)
    
    parser.add_argument("--lr_rate1", default=1e-3, help='learning rate 1', type=float)
    parser.add_argument("--lr_rate2", default=1e-3, help='learning rate 2', type=float)
    parser.add_argument("--keep_prob", help='keep probability for dropout', default=0.6, type=float)
    parser.add_argument("--reg_scale", default=0., help='l1-regularization', type=float)

    parser.add_argument('--alpha', default=0.1, help='coefficient -- alpha', type=float)
    parser.add_argument('--beta', default=1.0, help='coefficient -- beta', type=float)
    parser.add_argument('--gamma', default=1.0, help='coefficient -- gamma', type=float)
        
    parser.add_argument('--pred_times', default=[0, 180, 365], help='list of prediction times', type=int)
    parser.add_argument('--eval_times', default=[180, 365], help='list of evaluation times', type=int)

    return parser.parse_args()


if __name__ == '__main__':    
    args             = init_arg()    

    data_path        = args.data_path
    save_path        = args.save_path
    open_path        = args.open_path
    
    if not os.path.exists(save_path + '/models/'):
        os.makedirs(save_path + '/models/')

    if not os.path.exists(save_path + '/results/'):
        os.makedirs(save_path + '/results/')

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

         
    ##### MAKE DICTIONARIES
    # INPUT DIMENSIONS
    input_dims                  = { 'x_dim_static'      : x_dim_static,
                                    'x_dim_timevarying' : x_dim_timevarying, #this includes delta
                                    'num_Event'         : num_Event,        
                                    'xt_con_list'       : xt_con_list,
                                    'xt_bin_list'       : xt_bin_list,
                                    'max_length'        : max_length }

    # NETWORK HYPER-PARMETERS
    network_settings = load_logging(open_path + '/network_settings.txt')
    network_settings['FC_active_fn']  = ACTIVATION_FN[network_settings['FC_active_fn']]
    network_settings['RNN_active_fn'] = ACTIVATION_FN[network_settings['RNN_active_fn']]
    network_settings['initial_W']     = INTIALIZATION_FN[network_settings['initial_W']]

    

    (tr_data_s,te_data_s, tr_data_t,te_data_t, tr_time,te_time, tr_tte,te_tte, tr_label,te_label, tr_tte_new,te_tte_new, tr_label_new,te_label_new) = train_test_split(
        data_xs, data_xt, data_time, data_tte, data_y, data_tte_new, data_y_new, test_size=0.2, random_state=seed
    ) 

    (tr_data_s,va_data_s, tr_data_t,va_data_t, tr_time,va_time, tr_tte,va_tte, tr_label,va_label, tr_tte_new,va_tte_new, tr_label_new,va_label_new) = train_test_split(
        tr_data_s, tr_data_t, tr_time, tr_tte, tr_label, tr_tte_new, tr_label_new, test_size=0.2, random_state=seed
    ) 

    ### Initialization
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions()
    config      = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True

    sess        = tf.Session(config=config)
    model       = Model_DeepHit_Weibull(sess, "DDHL", input_dims, network_settings)

    saver = tf.train.Saver()
    saver.restore(sess, open_path + 'model_tte')
    enc_vars   = sess.run(model.enc_vars)
    pred_vars  = sess.run(model.pred_vars)
    
    
    K                                   = args.K
    z_dim                               = network_settings['h_dim_RNN'] * network_settings['num_layers_RNN']

    network_settings['h_dim_s']         = args.h_dim_S
    network_settings['num_layers_s']    = args.num_layers_S
    input_dims['max_cluster']           = K
    
    
    tf.reset_default_graph()
    config      = tf.ConfigProto()
    sess        = tf.Session(config=config)
    model       = ACTPC_DeepHit_Weibull(sess, "DDHL_ACTPC", input_dims, network_settings)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer(), feed_dict={model.E:np.zeros([K, z_dim]).astype(float)})
    
    ### Initialize AC-TPC
    for n in range(len(enc_vars)):
        _ = sess.run(tf.assign(model.blackbox_enc_vars[n], enc_vars[n]))
        _ = sess.run(tf.assign(model.vars_encoder[n], enc_vars[n]))

    for n in range(len(pred_vars)):
        _ = sess.run(tf.assign(model.blackbox_pred_vars[n], pred_vars[n]))
        _ = sess.run(tf.assign(model.vars_predictor[n], pred_vars[n]))
    
    
    
    ##### HYPER-PARAMETERS
    mb_size                     = args.mb_size
    keep_prob                   = args.keep_prob
    lr_rate1                    = args.lr_rate1
    lr_rate2                    = args.lr_rate2
    
    
    print('=============================================')
    print('===== INITIALIZING EMBEDDING & SELECTOR =====')
    # K-means over the latent encodings
    e, s_init, tmp_z = initialize_embedding(model, tr_data_s, tr_data_t, K)
    e = np.arctanh(e)
    sess.run(model.EE.initializer, feed_dict={model.E:e}) #model.EE = tf.nn.tanh(model.E)

    # update selector wrt initial classes
    iteration  = 1000
    check_step = 500

    avg_loss_s = 0
    for itr in range(iteration):
        z_mb, s_mb = f_get_minibatch(mb_size, tmp_z, s_init)
        _, tmp_loss_s = model.train_selector(z_mb, s_mb, lr_rate1, k_prob=keep_prob)

        avg_loss_s += tmp_loss_s/check_step
        if (itr+1)%check_step == 0:
            print("ITR:{:04d} | Loss_s:{:.4f}".format(itr+1, avg_loss_s) )
            avg_loss_s = 0

    tmp_ybars = model.get_lambdas_yy(np.tanh(e))
    new_e     = np.copy(e)
    print('=============================================')
    
    alpha  = args.alpha   
    beta   = args.beta
    gamma  = args.gamma  
    
    M          = int(tr_data_s.shape[0]/mb_size) #for main algorithm
    

    print('=============================================')
    print('========== TRAINING MAIN ALGORITHM ==========')

    iteration     = 5000
    check_step    = 10

    avg_loss_c_L1 = 0
    avg_loss_a_L1 = 0
    avg_loss_a_L2 = 0
    avg_loss_a_L4 = 0
    avg_loss_e_L1 = 0 
    avg_loss_e_L3 = 0

    va_avg_loss_L1 = 0
    va_avg_loss_L2 = 0
    va_avg_loss_L3 = 0
    va_avg_loss_L4 = 0
    
    min_loss  = 1e8
    max_flag  = 30
    stop_flag = 0

    for itr in range(iteration):        
        e = np.copy(new_e)

        for _ in range(M):
            x_mb, y_mb = f_get_minibatch(mb_size, tr_data_s, tr_data_t)

            _, tmp_loss_c_L1  = model.train_critic(x_mb, y_mb, lr_rate1, keep_prob)
            avg_loss_c_L1    += tmp_loss_c_L1/(M*check_step)

            x_mb, y_mb = f_get_minibatch(mb_size, tr_data_s, tr_data_t)

            _, tmp_loss_a_L1, tmp_loss_a_L2, tmp_loss_a_L4 = model.train_actor(x_mb, y_mb, alpha, gamma, lr_rate2, keep_prob)
            avg_loss_a_L1 += tmp_loss_a_L1/(M*check_step)
            avg_loss_a_L2 += tmp_loss_a_L2/(M*check_step)
            avg_loss_a_L4 += tmp_loss_a_L4/(M*check_step)

        for _ in range(M):
            x_mb, y_mb = f_get_minibatch(mb_size, tr_data_s, tr_data_t)

            _, tmp_loss_e_L1, tmp_loss_e_L3 = model.train_embedding(x_mb, y_mb, beta, lr_rate1, keep_prob)
            avg_loss_e_L1  += tmp_loss_e_L1/(M*check_step)
            avg_loss_e_L3  += tmp_loss_e_L3/(M*check_step)


        x_mb, y_mb = f_get_minibatch(min(mb_size, len(va_data_s)), va_data_s, va_data_t)
        tmp_loss_L1, tmp_loss_L2, tmp_loss_L3, tmp_loss_L4 = model.get_losses(va_data_s, va_data_t)

        va_avg_loss_L1  += tmp_loss_L1/check_step
        va_avg_loss_L2  += tmp_loss_L2/check_step
        va_avg_loss_L3  += tmp_loss_L3/check_step
        va_avg_loss_L4  += tmp_loss_L4/check_step

        new_e = sess.run(model.embeddings)

        if (itr+1)%check_step == 0:
            stop_flag += 1
            tmp_ybars = model.get_lambdas_yy(new_e)

            pred_y, tmp_m = model.get_s_sample(tr_data_s, tr_data_t)
            pred_y = pred_y.reshape([-1, 1])[tmp_m.reshape([-1]) == 1]

            va_avg_loss = va_avg_loss_L1 + alpha*va_avg_loss_L2 + beta*va_avg_loss_L3 + gamma*va_avg_loss_L4
            
            print ("ITR {:04d}: K={} | L1_c={:.3f} L1_a={:.3f} L1_e={:.3f} L2={:.3f} L3={:.3f} L4={:.3f} || vL={:.3f}  vL1={:.3f} vL2={:.3f} vL3={:.3f} vL4={:.3f}".format(
                itr+1, len(np.unique(pred_y)), avg_loss_c_L1, avg_loss_a_L1, avg_loss_e_L1, avg_loss_a_L2, avg_loss_e_L3, avg_loss_a_L4,
                va_avg_loss, va_avg_loss_L1, va_avg_loss_L2, va_avg_loss_L3, va_avg_loss_L4
            ))            
            
            
            if min_loss > va_avg_loss:
                stop_flag = 0
                min_loss = va_avg_loss

                saver.save(sess, save_path + '/models/model_acptc')
                print('saved...')

            avg_loss_c_L1 = 0
            avg_loss_a_L1 = 0
            avg_loss_a_L2 = 0
            avg_loss_a_L4 = 0

            avg_loss_e_L1 = 0
            avg_loss_e_L3 = 0

            va_avg_loss_L1 = 0
            va_avg_loss_L2 = 0
            va_avg_loss_L3 = 0
            va_avg_loss_L4 = 0    
    
    
        if stop_flag >= max_flag:
            print('model trained...')
            break
            
    saver.restore(sess, save_path + '/models/model_acptc')
    
    ### Cluster assignment prediction
    _, tmp_pi, tmp_m2 = model.get_zbars_and_pis_m2(tr_data_s, tr_data_t)

    tmp = np.argmax(tmp_pi[tmp_m2 == 1], axis=1)
    tr_cluster = np.nan * np.ones([len(tr_data_s), max_length])
    tr_cluster[tmp_m2 == 1] = tmp


    ### Cluster assignment prediction
    _, tmp_pi, tmp_m2 = model.get_zbars_and_pis_m2(te_data_s, te_data_t)

    tmp = np.argmax(tmp_pi[tmp_m2 == 1], axis=1)
    te_cluster = np.nan * np.ones([len(te_data_s), max_length])
    te_cluster[tmp_m2 == 1] = tmp
    print('=============================================')

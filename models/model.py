_EPSILON = 1e-08

import numpy as np
import tensorflow as tf
import random

from tensorflow.contrib.layers import fully_connected as FC_Net
from tensorflow.python.ops.rnn import _transpose_batch_time


import models.utils_network as utils
# import utils_network as utils

##### USER-DEFINED FUNCTIONS
def log(x):
    return tf.log(x + _EPSILON)

def div(x, y):
    return tf.div(x, (y + _EPSILON))

def get_seq_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    tmp_length = tf.reduce_sum(used, 1)
    tmp_length = tf.cast(tmp_length, tf.int32)
    return tmp_length


class Model_DeepHit_Weibull:
    def __init__(self, sess, name, input_dims, network_settings):
        self.sess               = sess
        self.name               = name

        # INPUT DIMENSIONS
        self.x_dim_static       = input_dims['x_dim_static']
        self.x_dim_timevarying  = input_dims['x_dim_timevarying'] #this includes feature dim + 1 (dela)
        
        self.xt_con_list        = input_dims['xt_con_list']
        self.xt_bin_list        = input_dims['xt_bin_list']
        
        self.num_Event          = input_dims['num_Event'] #number of event types.
        self.max_length         = input_dims['max_length']
        
        self.xt_con_list        = input_dims['xt_con_list']
        self.xt_con_list        = input_dims['xt_con_list']
        
        # NETWORK HYPER-PARMETERS
        self.p_weibull          = network_settings['p_weibull']
        
        self.h_dim1             = network_settings['h_dim_RNN']
        self.h_dim2             = network_settings['h_dim_FC']
        self.num_layers_RNN     = network_settings['num_layers_RNN']
        self.num_layers_FC      = network_settings['num_layers_FC']

        self.RNN_type           = network_settings['RNN_type']

        self.FC_active_fn       = network_settings['FC_active_fn']
        self.RNN_active_fn      = network_settings['RNN_active_fn']
        self.initial_W          = network_settings['initial_W']
        self.reg_scale          = network_settings['reg_scale']
        
        self.reg_W              = tf.contrib.layers.l1_regularizer(scale=self.reg_scale)
        self.reg_W_out          = tf.contrib.layers.l1_regularizer(scale=self.reg_scale)


        self._build_net()


    def _build_net(self):
        with tf.variable_scope(self.name):
            #### PLACEHOLDER DECLARATION
            self.mb_size     = tf.placeholder(tf.int32, [], name='batch_size')

            self.lr_rate     = tf.placeholder(tf.float32)
            self.keep_prob   = tf.placeholder(tf.float32)  #keeping rate            
            
            self.a           = tf.placeholder(tf.float32, name='alpha') #step-ahead prediction loss          
  
            self.xs          = tf.placeholder(tf.float32, shape=[None, self.x_dim_static])
            self.xt          = tf.placeholder(tf.float32, shape=[None, self.max_length, self.x_dim_timevarying], name='input')
            self.time        = tf.placeholder(tf.float32, shape=[None, self.max_length, 1], name='time_stamp')
            
            self.label       = tf.placeholder(tf.int32,   shape=[None, self.max_length, 1], name='label_next_event')     #event/censoring label (censoring:0)
            self.tte         = tf.placeholder(tf.float32, shape=[None, self.max_length, 1], name='time_to_next_event')    
            
            self.tau         = tf.placeholder(tf.float32, shape=[None, 1])  #at evaluation time (required for risk prediction)
            self.M_onehot    = tf.one_hot(tf.reshape(self.label, [-1, self.max_length]), self.num_Event+1)[:, :, 1:]  # censoring removed
                        
            '''
                ##### CREATE MASK
                    - rnn_mask1 [None, 1] (to collect the last value - Lambda & z)
                    - rnn_mask2 (for the last value - get lambda)
                    - rnn_mask3 (for the last value - get lambda)
            '''
            seq_length = get_seq_length(self.xt)

            tmp_range = tf.expand_dims(tf.range(0, self.max_length, 1), axis=0)
            self.rnn_mask1 = tf.cast(tf.equal(tmp_range, tf.expand_dims(seq_length - 1, axis=1)), tf.float32)
            self.rnn_mask2 = tf.cast(tf.less_equal(tmp_range, tf.expand_dims(seq_length - 1, axis=1)), tf.float32)
            self.rnn_mask3 = tf.tile(tf.expand_dims(self.rnn_mask2, axis=2), [1,1,self.x_dim_timevarying]) #for hisotry (1...J-1)
            
            self.p = tf.constant(self.p_weibull, name="P")    
            
            def loss_LogLikelihood(loglambda_, M_onehot_, time_, label_, tte_):
                ##### LOSS1: LOGLIKELIHOOD
                tmp_lambda   = tf.exp(loglambda_)
                
                tmp_tau      = tte_ - time_ # this is because, tte_ can change at every time step (since we are focusing on the time-to-next-event)
                I1           = 1. - tf.cast(tf.equal(label_, 0), dtype = tf.float32) #indicator for "next-event available"
                
                tmp1         = tf.reduce_sum(M_onehot_ * log(self.p * tf.pow(tmp_lambda, self.p) * tf.pow(tmp_tau, self.p-1.)), axis=1)
                tmp2         = tf.reduce_sum(tf.pow(tf.reduce_sum(tmp_lambda, axis=1, keepdims=True), self.p) * tf.pow(tmp_tau, self.p), axis=1)
                
                loss         = - (tf.reshape(I1, [-1]) * tmp1 - tmp2)
                return loss

            # predictions on the next time-varying covariates
            '''
                continuous binary splitt!!!!!!! then use mse and cross entropy respectively!
            '''            
            def loss_RNN_Prediction(xt_, xhat_):
                tmp_x    = xt_[:, 1:, 1:]
                tmp_mask = self.rnn_mask3[:, 1:, 1:]
                
                con_loss = 0.
                bin_loss = 0.
                
                if len(self.xt_con_list) != 0:
                    mask_con  = tf.gather(tmp_mask, self.xt_con_list, axis=2)
                    tmp_x_con = tf.gather(tmp_x, self.xt_con_list, axis=2)
                    x_hat_con = tf.gather(xhat_, self.xt_con_list, axis=2)
                    
#                     con_loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(mask_con*(tmp_x_con - x_hat_con)**2, axis=2), axis=1))  
                    con_loss = tf.reduce_mean(div(1., tf.reduce_sum(self.rnn_mask2, axis=1)) * tf.reduce_sum(tf.reduce_sum(mask_con*(tmp_x_con - x_hat_con)**2, axis=2), axis=1))                       
                    
                if len(self.xt_bin_list) != 0:
                    mask_bin  = tf.gather(tmp_mask, self.xt_bin_list, axis=2)
                    tmp_x_bin = tf.gather(tmp_x, self.xt_bin_list, axis=2)
                    x_hat_bin = tf.gather(xhat_, self.xt_bin_list, axis=2)
                    
#                     bin_loss = - tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(mask_bin*(tmp_x_bin*log(x_hat_bin) + (1.-tmp_x_bin)*log(1.-x_hat_bin)), axis=2), axis=1))  
                    bin_loss = - tf.reduce_mean(div(1., tf.reduce_sum(self.rnn_mask2, axis=1)) * tf.reduce_sum(tf.reduce_sum(mask_bin*(tmp_x_bin*log(x_hat_bin) + (1.-tmp_x_bin)*log(1.-x_hat_bin)), axis=2), axis=1))  
                    
                return con_loss + bin_loss
            
            # estimator network (i.e., predictions on the log_lambda)           
            def prediction_network(h_, reuse=tf.AUTO_REUSE):
                # outputs log_lambda i.e., lambda = tf.exp(out)
                with tf.variable_scope('prediction_net', reuse=reuse):
                    out = utils.create_FCNet(h_, (self.num_layers_FC), self.h_dim2, self.FC_active_fn, self.num_Event, None, self.initial_W, self.reg_W, self.keep_prob)
#                     out = tf.nn.softplus(h_)
                return out

            '''
                ##### DEFINE LOOP FUNCTION FOR TEMPORAL ATTENTION
                    - loop_state[0]: attention power (i.e., e_{j})
                    - loop_state[1]: hidden states
            '''
            def loop_fn_mle(time, cell_output, cell_state, loop_state):

                emit_output = cell_output 

                if cell_output is None:  # time == 0
                    next_cell_state = cell.zero_state(self.mb_size, tf.float32)
                    next_loop_state = loop_state_ta
                else:
                    next_cell_state = cell_state
                    tmp_z = utils.create_concat_state(next_cell_state, self.num_layers_RNN, self.RNN_type)
                                        
                    tmp_loglambda   = prediction_network(tmp_z)
                    tmp_loss        = loss_LogLikelihood(
                        tmp_loglambda, 
                        tf.reshape(self.M_onehot[:, time-1, :], [-1,self.num_Event]), 
                        tf.reshape(self.time[:, time-1, 0], [-1,1]),
                        tf.reshape(self.label[:, time-1, 0], [-1,1]),
                        tf.reshape(self.tte[:, time-1, 0], [-1,1])
                    )
                    
                    
                    next_loop_state = (
                        loop_state[0].write(time-1, tmp_z),         # save all the hidden states continuous
                        loop_state[1].write(time-1, tmp_loglambda), # save all the loglambdas
                        loop_state[2].write(time-1, tmp_loss),  # save all the log_mle
                    )

#                 elements_finished = (time >= seq_length)
                elements_finished = (time >= self.max_length)

                #this gives the break-point (no more recurrence after the max_length)
                finished = tf.reduce_all(elements_finished)    
                next_input = tf.cond(finished, 
                                     lambda: tf.zeros([self.mb_size, self.x_dim_timevarying+self.x_dim_static], dtype=tf.float32),  # [xs, xt_hist]
                                     lambda: tf.concat([inputs_ta.read(time), self.xs], axis=1))
                return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

            inputs = self.xt #(this contains delta at the first column)
            
            '''
                ##### RNN NETWORK
                    - (INPUT)  inputs_ta: TensorArray with [max_length, mb_size, 1+self.x_dim_timevarying+self.x_dim_static]
                    - (OUTPUT) rnn_outputs_ta: TensorArray
                    - (OUTPUT) rnn_final_state: Tensor
                    - (OUTPUT) loop_state_ta: 2 TensorArrays (e values and hidden states)
            '''
            inputs_ta = tf.TensorArray(
                dtype=tf.float32, size=self.max_length
            ).unstack(_transpose_batch_time(inputs), name = 'Rnn_Input')

            cell = utils.create_rnn_cell(self.h_dim1, self.num_layers_RNN, self.keep_prob, 
                                         self.RNN_type, self.RNN_active_fn)

            #define the loop_state TensorArray for information from rnn time steps
            loop_state_ta = (
                tf.TensorArray(size=self.max_length, dtype=tf.float32),  #hidden states continuous(j=1,...,J)
                tf.TensorArray(size=self.max_length, dtype=tf.float32),  #lambdas (j=1,...,J)
                tf.TensorArray(size=self.max_length, dtype=tf.float32),  #loglikelihoods(j=1,...,J)
            )  
            
            rnn_outputs_ta, _, loop_state_ta = tf.nn.raw_rnn(cell, loop_fn_mle)
            
            self.rnn_states           = _transpose_batch_time(loop_state_ta[0].stack())
            self.ys                   = _transpose_batch_time(loop_state_ta[1].stack())
            self.ys                   = tf.exp(self.ys) #log_lambda -> lambda
            self.loglikelihoods       = _transpose_batch_time(loop_state_ta[2].stack())
                        
            
            self.z      = tf.reduce_sum(
                self.rnn_states * tf.tile(tf.expand_dims(self.rnn_mask1, axis=2), [1,1,self.h_dim1*self.num_layers_RNN]), axis=1
            )
            self.y      = tf.reduce_sum(
                self.ys * tf.tile(tf.expand_dims(self.rnn_mask1, axis=2), [1,1,self.num_Event]), axis=1
            )
            
            
            '''
                ##### STEP-AHEAD PREDICTIONS
                    - x_hat: predictions for j = 2,...,J [None, max_length, x_dim-1] (delta removed)
            '''
            rnn_outputs         = _transpose_batch_time(rnn_outputs_ta.stack())
            stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, self.h_dim1])
            
            self.x_hat  = utils.create_FCNet(stacked_rnn_outputs, 2, self.h_dim2, self.FC_active_fn, self.x_dim_timevarying-1, tf.nn.sigmoid, self.initial_W, self.reg_W, self.keep_prob)
            self.x_hat  = tf.reshape(self.x_hat, [-1, self.max_length, self.x_dim_timevarying-1])
            self.x_hat  = (self.x_hat * self.rnn_mask3[:, :, 1:])[:, :-1, :]
            
            
            
            '''
                ##### RISK PREDICTIONS
                    - risk:  risk predictions given the evaulation time (tau)
            '''
            self.risk     = div(self.y, tf.reduce_sum(self.y, axis=1, keepdims=True)) * (1. - tf.exp(- tf.pow(tf.reduce_sum(self.y, axis=1, keepdims=True) * self.tau, self.p)))

            '''
                ##### DEFINE MAXIMUM-LIKELIHOOD LOSS
                    - LOSS_1: loglikelihood
                    - LOSS_2: regularization - step-ahead predictions
            '''
#             self.loss_mle       = tf.reduce_mean(tf.reduce_sum(self.loglikelihoods * self.rnn_mask2, axis=1))            
#             self.loss_stepahead = loss_RNN_Prediction(self.xt, self.x_hat)

            self.loss_mle       = tf.reduce_mean(div(1., tf.reduce_sum(self.rnn_mask2, axis=1)) * tf.reduce_sum(self.loglikelihoods * self.rnn_mask2, axis=1))            
            self.loss_stepahead = loss_RNN_Prediction(self.xt, self.x_hat)
            
            self.LOSS_TTE       = self.loss_mle + self.a*self.loss_stepahead #+ tf.losses.get_regularization_loss()
                        
            self.global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self.pred_vars   = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/rnn/prediction_net') 
            self.enc_vars    = [tmp_var for tmp_var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/rnn/') if tmp_var not in self.pred_vars]
            self.recon_vars  = [tmp_var for tmp_var in self.global_vars if tmp_var not in self.pred_vars+self.enc_vars]
            
            self.solver_TTE   = tf.train.AdamOptimizer(learning_rate=self.lr_rate).minimize(self.LOSS_TTE, var_list=self.enc_vars+self.pred_vars+self.recon_vars)
            
            
    def train(self, DATA, a_, keep_prob, lr_train):
        (xs_, xt_, t_, tte_, m_)   = DATA
        return self.sess.run([self.solver_TTE, self.LOSS_TTE, self.loss_mle], 
                             feed_dict={self.xs:xs_, self.xt:xt_, self.time: t_, 
                                        self.tte:tte_, self.label:m_, 
                                        self.a:a_, self.mb_size: np.shape(xt_)[0], self.keep_prob:keep_prob, self.lr_rate:lr_train})

    
    def get_cost(self, DATA, a_):
        (xs_, xt_, t_, tte_, m_)   = DATA
        return self.sess.run([self.LOSS_TTE, self.loss_mle], 
                             feed_dict={self.xs:xs_, self.xt:xt_, self.time: t_, 
                                        self.tte:tte_, self.label:m_, 
                                        self.a:a_, self.mb_size: np.shape(xt_)[0], self.keep_prob:1.0})
    
    
    def get_risk(self, xs_, xt_, tau_):
        return self.sess.run(self.risk, 
                             feed_dict={self.xs:xs_, self.xt: xt_, self.tau: tau_,
                                        self.mb_size: np.shape(xt_)[0], self.keep_prob:1.0})
    
    
    def get_zs(self, xs_, xt_, keep_prob=1.0):
        return self.sess.run(self.rnn_states, 
                             feed_dict={self.xs:xs_, self.xt:xt_, 
                                        self.mb_size: np.shape(xt_)[0], self.keep_prob: keep_prob})
    
    def get_final_z(self, xs_, xt_, keep_prob=1.0):
        return self.sess.run(self.z, 
                             feed_dict={self.xs:xs_, self.xt:xt_, 
                                        self.mb_size: np.shape(xt_)[0], self.keep_prob: keep_prob})
    
    
    def get_lambdas(self, xs_, xt_, keep_prob=1.0):
        return self.sess.run(self.ys, 
                             feed_dict={self.xs:xs_, self.xt:xt_, 
                                        self.mb_size: np.shape(xt_)[0], self.keep_prob: keep_prob})
    
    def get_final_lambda(self, xs_, xt_, keep_prob=1.0):
        return self.sess.run(self.y, 
                             feed_dict={self.xs:xs_, self.xt:xt_, 
                                        self.mb_size: np.shape(xt_)[0], self.keep_prob: keep_prob})
       
    
    def get_xhat(self, xs_, xt_, keep_prob=1.0):
        return self.sess.run(self.x_hat, 
                             feed_dict={self.xs:xs_, self.xt:xt_, 
                                        self.mb_size: np.shape(xt_)[0], self.keep_prob: keep_prob})

    
    
    
    
    
###############################################################################
def Gaussian_CDF(x): #change from erf function
    return 0.5 * (1. + tf.math.erf(x / tf.math.sqrt(2.))) 

class ACTPC_DeepHit_Weibull:
    def __init__(self, sess, name, input_dims, network_settings):
        self.sess               = sess
        self.name               = name

        # INPUT DIMENSIONS
        self.x_dim_static       = input_dims['x_dim_static']
        self.x_dim_timevarying  = input_dims['x_dim_timevarying'] #this includes feature dim + 1 (dela)
                
        self.num_Event          = input_dims['num_Event'] #number of event types.
        self.max_length         = input_dims['max_length']
        
        self.K                  = input_dims['max_cluster']
        
        # NETWORK HYPER-PARMETERS
        self.p_weibull          = network_settings['p_weibull']
        
        self.h_dim1             = network_settings['h_dim_RNN']
        self.h_dim2             = network_settings['h_dim_FC']
        self.num_layers_RNN     = network_settings['num_layers_RNN']
        self.num_layers_FC      = network_settings['num_layers_FC']
        
        self.h_dim_s            = network_settings['h_dim_s']  #selector_net
        self.num_layers_s       = network_settings['num_layers_s'] #selector_net
        
        self.RNN_type           = network_settings['RNN_type']

        self.FC_active_fn       = network_settings['FC_active_fn']
        self.RNN_active_fn      = network_settings['RNN_active_fn']
        self.initial_W          = network_settings['initial_W']
        self.reg_scale          = network_settings['reg_scale']
        
        self.reg_W              = tf.contrib.layers.l1_regularizer(scale=self.reg_scale)
        self.reg_W_out          = tf.contrib.layers.l1_regularizer(scale=self.reg_scale)

        self.z_dim              = self.h_dim1 * self.num_layers_RNN
        
        self._build_net()


    def _build_net(self):
        with tf.variable_scope(self.name):
            #### PLACEHOLDER DECLARATION
            self.lr_rate1    = tf.placeholder(tf.float32)
            self.lr_rate2    = tf.placeholder(tf.float32)
            
            self.keep_prob   = tf.placeholder(tf.float32)  #keeping rate            
             
            self.xs          = tf.placeholder(tf.float32, shape=[None, self.x_dim_static])
            self.xt          = tf.placeholder(tf.float32, shape=[None, self.max_length, self.x_dim_timevarying], name='input')
            
            self.mb_size     = tf.placeholder(tf.int32, [], name='batch_size')
#             self.mb_size     = tf.shape(self.xs)[0]
            
            self.is_training = tf.placeholder(tf.bool, [], name='trainiin_indicator')            
            self.tau         = tf.placeholder(tf.float32, shape=[None, 1])  #at evaluation time (required for risk prediction)
            
            
            # LOSS PARAMETERS
            self.alpha      = tf.placeholder(tf.float32, name = 'alpha') #For sample-wise entropy
            self.beta       = tf.placeholder(tf.float32, name = 'beta')  #For prediction loss (i.e., mle)
            self.gamma      = tf.placeholder(tf.float32, name = 'gamma') #For batch-wise entropy
            self.delta      = tf.placeholder(tf.float32, name = 'delta') #For embedding
            
                        
            '''
                ##### CREATE MASK
                    - rnn_mask1 [None, 1] (to collect the last value - Lambda & z)
                    - rnn_mask2 (for the last value - get lambda)
                    - rnn_mask3 (for the last value - get lambda)
            '''
            seq_length = get_seq_length(self.xt)
            tmp_range = tf.expand_dims(tf.range(0, self.max_length, 1), axis=0)
            self.rnn_mask1 = tf.cast(tf.equal(tmp_range, tf.expand_dims(seq_length - 1, axis=1)), tf.float32)
            self.rnn_mask2 = tf.cast(tf.less_equal(tmp_range, tf.expand_dims(seq_length - 1, axis=1)), tf.float32)
            self.rnn_mask3 = tf.tile(tf.expand_dims(self.rnn_mask2, axis=2), [1,1,self.x_dim_timevarying]) #for hisotry (1...J-1)
            
            self.p = tf.constant(self.p_weibull, name="P")
            

            ### BLACKBOX MODEL
            with tf.variable_scope('blackbox'):
                inputs_ta = tf.TensorArray(
                    dtype=tf.float32, size=self.max_length
                ).unstack(_transpose_batch_time(self.xt), name = 'Rnn_Input'
                         )

                # estimator network (i.e., predictions on the log_lambda)           
                def prediction_network(h_, reuse=tf.AUTO_REUSE):
                    # outputs log_lambda i.e., lambda = tf.exp(out)
                    with tf.variable_scope('prediction_net', reuse=reuse):
                        out = utils.create_FCNet(h_, (self.num_layers_FC), self.h_dim2, self.FC_active_fn, self.num_Event, None, self.initial_W, self.reg_W, self.keep_prob)
    #                     out = tf.nn.softplus(h_)
                    return out

                '''
                    ##### DEFINE LOOP FUNCTION FOR TEMPORAL ATTENTION
                        - loop_state[0]: attention power (i.e., e_{j})
                        - loop_state[1]: hidden states
                '''
                def loop_fn_mle(time, cell_output, cell_state, loop_state):

                    emit_output = cell_output 

                    if cell_output is None:  # time == 0
                        next_cell_state = cell.zero_state(self.mb_size, tf.float32)
                        next_loop_state = loop_state_ta
                    else:
                        next_cell_state = cell_state
                        tmp_z = utils.create_concat_state(next_cell_state, self.num_layers_RNN, self.RNN_type)

                        tmp_loglambda   = prediction_network(tmp_z)                    

                        next_loop_state = (
                            loop_state[0].write(time-1, tmp_z),         # save all the hidden states continuous
                            loop_state[1].write(time-1, tmp_loglambda), # save all the loglambdas
                        )

    #                 elements_finished = (time >= seq_length)
                    elements_finished = (time >= self.max_length)

                    #this gives the break-point (no more recurrence after the max_length)
                    finished = tf.reduce_all(elements_finished)    
                    next_input = tf.cond(finished, 
                                         lambda: tf.zeros([self.mb_size, self.x_dim_timevarying+self.x_dim_static], dtype=tf.float32),  # [xs, xt_hist]
                                         lambda: tf.concat([inputs_ta.read(time), self.xs], axis=1))
                    return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

                
                '''
                    ##### RNN NETWORK
                        - (INPUT)  inputs_ta: TensorArray with [max_length, mb_size, 1+self.x_dim_timevarying+self.x_dim_static]
                        - (OUTPUT) rnn_outputs_ta: TensorArray
                        - (OUTPUT) rnn_final_state: Tensor
                        - (OUTPUT) loop_state_ta: 2 TensorArrays (e values and hidden states)
                '''
                cell = utils.create_rnn_cell(self.h_dim1, self.num_layers_RNN, self.keep_prob, 
                                             self.RNN_type, self.RNN_active_fn)

                #define the loop_state TensorArray for information from rnn time steps
                loop_state_ta = (
                    tf.TensorArray(size=self.max_length, dtype=tf.float32),  #hidden states continuous(j=1,...,J)
                    tf.TensorArray(size=self.max_length, dtype=tf.float32),  #lambdas (j=1,...,J)
                )  

                rnn_outputs_ta, _, loop_state_ta = tf.nn.raw_rnn(cell, loop_fn_mle)

#                 self.rnn_states           = _transpose_batch_time(loop_state_ta[0].stack())
                self.ys                   = _transpose_batch_time(loop_state_ta[1].stack())
                self.ys                   = tf.exp(self.ys) #log_lambda -> lambda                       

#                 self.z      = tf.reduce_sum(
#                     self.rnn_states * tf.tile(tf.expand_dims(self.rnn_mask1, axis=2), [1,1,self.h_dim1*self.num_layers_RNN]), axis=1
#                 )
                self.y      = tf.reduce_sum(
                    self.ys * tf.tile(tf.expand_dims(self.rnn_mask1, axis=2), [1,1,self.num_Event]), axis=1
                )

                self.risk     = div(self.y, tf.reduce_sum(self.y, axis=1, keepdims=True)) * (1. - tf.exp(- tf.pow(tf.reduce_sum(self.y, axis=1, keepdims=True) * self.tau, self.p)))
                
                
            ### ACTPC MODEL
            with tf.variable_scope('actpc'):
                # Embedding
                self.E          = tf.placeholder(tf.float32, [self.K, self.z_dim], name='embeddings_input')
                self.EE         = tf.Variable(self.E, name='embeddings_var')
                self.embeddings = tf.nn.tanh(self.EE)

                self.s          = tf.placeholder(tf.int32, [None], name='cluster_label')
                self.s_onehot   = tf.one_hot(self.s, self.K)


                inputs_ta_actpc = tf.TensorArray(
                    dtype=tf.float32, size=self.max_length
                ).unstack(_transpose_batch_time(self.xt), name = 'Rnn_Input'
                         )
                # estimator network (i.e., predictions on the log_lambda)           
                def prediction_network(h_, reuse=tf.AUTO_REUSE):
                    # outputs log_lambda i.e., lambda = tf.exp(out)
                    with tf.variable_scope('prediction_net', reuse=reuse):
                        out = utils.create_FCNet(h_, (self.num_layers_FC), self.h_dim2, self.FC_active_fn, self.num_Event, None, self.initial_W, self.reg_W, self.keep_prob)
    #                     out = tf.nn.softplus(h_)
                    return out
    
                ### DEFINE SELECTOR
                def selector(x_, o_dim_=self.K, num_layers_=self.num_layers_s, h_dim_=self.h_dim_s, activation_fn=self.FC_active_fn, reuse=tf.AUTO_REUSE):
                    out_fn = tf.nn.softmax
                    with tf.variable_scope('selector', reuse=reuse):
                        if num_layers_ == 1:
                            out =  tf.contrib.layers.fully_connected(inputs=x_, num_outputs=o_dim_, activation_fn=out_fn, scope='selector_out')
                        else: #num_layers > 1
                            for tmp_layer in range(num_layers_-1):
                                if tmp_layer == 0:
                                    net = x_
                                net = tf.contrib.layers.fully_connected(inputs=net, num_outputs=h_dim_, activation_fn=activation_fn, scope='selector_'+str(tmp_layer))
                                net = tf.nn.dropout(net, keep_prob=self.keep_prob)
                            out =  tf.contrib.layers.fully_connected(inputs=net, num_outputs=o_dim_, activation_fn=out_fn, scope='selector_out')  
                    return out

                '''
                    ##### DEFINE LOOP FUNCTION FOR TEMPORAL ATTENTION
                        - loop_state[0]: attention power (i.e., e_{j})
                        - loop_state[1]: hidden states
                '''
                def loop_fn_actpc(time, cell_output, cell_state, loop_state):

                    emit_output = cell_output 

                    if cell_output is None:  # time == 0
                        next_cell_state = cell.zero_state(self.mb_size, tf.float32)
                        next_loop_state = loop_state_ta_actpc
                    else:
                        next_cell_state = cell_state
                        tmp_z = utils.create_concat_state(next_cell_state, self.num_layers_RNN, self.RNN_type)
                        tmp_loglambda   = prediction_network(tmp_z)
                        tmp_pi          = selector(tmp_z, self.K, self.num_layers_s, self.h_dim_s, self.FC_active_fn)

                        next_loop_state = (
                            loop_state[0].write(time-1, tmp_z),         # save all the hidden states continuous
                            loop_state[1].write(time-1, tmp_loglambda), 
                            loop_state[2].write(time-1, tmp_pi) # save all the selector_net output (i.e., pi)
                        )

    #                 elements_finished = (time >= seq_length)
                    elements_finished = (time >= self.max_length)

                    #this gives the break-point (no more recurrence after the max_length)
                    finished = tf.reduce_all(elements_finished)    
                    next_input = tf.cond(finished, 
                                         lambda: tf.zeros([self.mb_size, self.x_dim_timevarying+self.x_dim_static], dtype=tf.float32),  # [xs, xt_hist]
                                         lambda: tf.concat([inputs_ta_actpc.read(time), self.xs], axis=1))
                    return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

                
                '''
                    ##### RNN NETWORK
                        - (INPUT)  inputs_ta: TensorArray with [max_length, mb_size, 1+self.x_dim_timevarying+self.x_dim_static]
                        - (OUTPUT) rnn_outputs_ta: TensorArray
                        - (OUTPUT) rnn_final_state: Tensor
                        - (OUTPUT) loop_state_ta: 2 TensorArrays (e values and hidden states)
                '''
                cell = utils.create_rnn_cell(self.h_dim1, self.num_layers_RNN, self.keep_prob, 
                                             self.RNN_type, self.RNN_active_fn)

                #define the loop_state TensorArray for information from rnn time steps
                loop_state_ta_actpc = (
                    tf.TensorArray(size=self.max_length, dtype=tf.float32),  #hidden states continuous(j=1,...,J)
                    tf.TensorArray(size=self.max_length, dtype=tf.float32),  #lambdas (j=1,...,J)
                    tf.TensorArray(size=self.max_length, dtype=tf.float32, clear_after_read=False)   #pis (j=1,...,J)
                )  

                rnn_outputs_ta, _, loop_state_ta_actpc = tf.nn.raw_rnn(cell, loop_fn_actpc)

                self.rnn_states           = _transpose_batch_time(loop_state_ta_actpc[0].stack())
                self.y_hats               = _transpose_batch_time(loop_state_ta_actpc[1].stack())
                self.pis                  = _transpose_batch_time(loop_state_ta_actpc[2].stack())
                
                self.y_hats               = tf.exp(self.y_hats) #log_lambda -> lambda                       


                ### SAMPLING PROCESS
                s_dist          = tf.distributions.Categorical(probs=tf.reshape(self.pis, [-1, self.K])) #define the categorical dist.
                s_sample        = s_dist.sample()

                mask_e          = tf.cast(tf.equal(tf.expand_dims(tf.range(0, self.K, 1), axis=0), tf.expand_dims(s_sample, axis=1)), tf.float32)
                z_bars          = tf.matmul(mask_e, self.embeddings)
                pi_sample       = tf.reduce_sum(mask_e * tf.reshape(log(self.pis), [-1, self.K]), axis=1)
                
                with tf.variable_scope('rnn'):
                    self.y_bars     = prediction_network(z_bars, reuse=True)
                    
                self.y_bars     = tf.reshape(self.y_bars, [-1, self.max_length, self.num_Event])
                self.y_bars     = tf.exp(self.y_bars) #log_lambda -> lambda                       

                self.z_bars     = tf.reshape(z_bars, [-1, self.max_length, self.z_dim])                
                self.pi_sample  = tf.reshape(pi_sample, [-1, self.max_length])
                self.s_sample   = tf.reshape(s_sample, [-1, self.max_length])
            
                
#                 self.z_hats     = tf.reduce_sum(
#                     self.rnn_states * tf.tile(tf.expand_dims(self.rnn_mask1, axis=2), [1,1,self.z_dim]), axis=1
#                 )
#                 self.z_bars     = tf.reduce_sum(
#                     self.z_bars * tf.tile(tf.expand_dims(self.rnn_mask1, axis=2), [1,1,self.z_dim]), axis=1
#                 )

                self.z_hats     = self.rnn_states * tf.tile(tf.expand_dims(self.rnn_mask2, axis=2), [1,1,self.z_dim])
                self.z_bars     = self.z_bars * tf.tile(tf.expand_dims(self.rnn_mask2, axis=2), [1,1,self.z_dim])

                self.z_hat      = tf.reduce_sum(
                    self.rnn_states * tf.tile(tf.expand_dims(self.rnn_mask1, axis=2), [1,1,self.z_dim]), axis=1
                )
                self.z_bar      = tf.reduce_sum(
                    self.z_bars * tf.tile(tf.expand_dims(self.rnn_mask1, axis=2), [1,1,self.z_dim]), axis=1
                )
                
                self.y_hat      = tf.reduce_sum(
                    self.y_hats * tf.tile(tf.expand_dims(self.rnn_mask1, axis=2), [1,1,self.num_Event]), axis=1
                )
                self.y_bar      = tf.reduce_sum(
                    self.y_bars * tf.tile(tf.expand_dims(self.rnn_mask1, axis=2), [1,1,self.num_Event]), axis=1
                )

                self.risk_hat   = div(self.y_hat, tf.reduce_sum(self.y_hat, axis=1, keepdims=True)) * (1. - tf.exp(- tf.pow(tf.reduce_sum(self.y_hat, axis=1, keepdims=True) * self.tau, self.p)))
                self.risk_bar   = div(self.y_bar, tf.reduce_sum(self.y_bar, axis=1, keepdims=True)) * (1. - tf.exp(- tf.pow(tf.reduce_sum(self.y_bar, axis=1, keepdims=True) * self.tau, self.p)))
                
                
                ### EMBEDDING TRAINING
                with tf.variable_scope('rnn'):
                    self.Ey   = prediction_network(self.embeddings, reuse=True)

            
            ### this is already a js divergence
            def js_distance_weibull(v1, v2): 
                return 0.5 *( (div(v1, v2)**self.p) + (div(v2, v1)**self.p) )
            
            
            #batch-wise entropy
            tmp_pis   = tf.tile(tf.expand_dims(self.rnn_mask2, axis=2), [1,1,self.K]) * self.pis
            mean_pis  = tf.reduce_sum(tf.reduce_sum(tmp_pis, axis=1), axis=0) / tf.reduce_sum(tf.reduce_sum(self.rnn_mask2, axis=1), axis=0, keepdims=True)
            
            
            ## LOSS_MLE: MLE prediction loss (for initalization)
            loss_hats       = tf.reshape( js_distance_weibull(self.ys, self.y_hats), [-1, self.max_length] )
            loss_bars       = tf.reshape( js_distance_weibull(self.ys, self.y_bars), [-1, self.max_length] )
            
            self.LOSS_MLE   = tf.reduce_mean(div(tf.reduce_sum(self.rnn_mask2 * loss_hats, axis=1), tf.reduce_sum(self.rnn_mask2, axis=1)))
                        
            ## LOSS1: predictive clustering loss
            self.LOSS_1     = tf.reduce_mean(div(tf.reduce_sum(self.rnn_mask2 * loss_bars, axis=1), tf.reduce_sum(self.rnn_mask2, axis=1)))
            self.LOSS_1_AC  = tf.reduce_mean(div(tf.reduce_sum(self.rnn_mask2 * self.pi_sample * loss_bars, axis=1), tf.reduce_sum(self.rnn_mask2, axis=1)))

            ## LOSS2: sample-wise entropy loss
            self.LOSS_2     = tf.reduce_mean(- div(tf.reduce_sum(self.rnn_mask2 * tf.reduce_sum(self.pis * log(self.pis), axis=2), axis=1), tf.reduce_sum(self.rnn_mask2, axis=1)))
                      
            ## LOSS4: average-wise entropy
            self.LOSS_4     = tf.reduce_sum(- mean_pis * log(mean_pis))
            
            
            self.vars_predictor  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= self.name+'/actpc/rnn/prediction_net')
            self.vars_selecter   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= self.name+'/actpc/rnn/selector')
            self.vars_embedding  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= self.name+'/actpc/embeddings_var')
            self.vars_encoder    = [vars_ for vars_ in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= self.name+'/actpc') 
                               if vars_ not in self.vars_predictor+self.vars_selecter+self.vars_embedding]
            
            ##### VARIABLES
            self.blackbox_vars        = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/blackbox')        
            self.blackbox_pred_vars   = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/blackbox/rnn/prediction_net') 
            self.blackbox_enc_vars    = [tmp_var for tmp_var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/blackbox/rnn/') if tmp_var not in self.blackbox_pred_vars]
            

            ## LOSS3: embedding separation loss (prevents embedding from collapsing)
            self.LOSS_3 = 0
            for i in range(self.K):
                for j in range(i+1, self.K):
                    self.LOSS_3 += - tf.reduce_sum(js_distance_weibull(self.Ey[i, :], self.Ey[j, :])) / ((self.K-1)*(self.K-2)) # negative because we want to increase this;
            
            LOSS_ACTOR = self.LOSS_1_AC + self.alpha*self.LOSS_2 + self.gamma*self.LOSS_4
            LOSS_EMBED = self.LOSS_1 + self.beta*self.LOSS_3
            
            ### DEFINE OPTIMIZATION SOLVERS
            self.solver_MLE           = tf.train.AdamOptimizer(self.lr_rate1).minimize(
                self.LOSS_MLE, var_list=self.vars_encoder+self.vars_predictor
            )
            self.solver_L1_critic     = tf.train.AdamOptimizer(self.lr_rate1).minimize(
                self.LOSS_1,
                var_list=self.vars_encoder+self.vars_predictor
            )
            self.solver_L1_actor      = tf.train.AdamOptimizer(self.lr_rate2).minimize(
                LOSS_ACTOR, 
                var_list=self.vars_encoder+self.vars_selecter
            )            
            self.solver_E             = tf.train.AdamOptimizer(self.lr_rate1).minimize(
                LOSS_EMBED, 
                var_list=self.vars_embedding
            )

            ### INITIALIZE SELECTOR
            self.zz     = tf.placeholder(tf.float32, [None, self.z_dim])
            with tf.variable_scope('actpc/rnn'):                
                self.yy    = prediction_network(self.zz, reuse=True)
                self.yy    = tf.exp(self.yy)
                self.s_out = selector(self.zz, self.K, self.num_layers_s, self.h_dim_s, self.FC_active_fn, reuse=True)
            
            ## LOSS_S: selector initialization (cross-entropy wrt initialized class)
            self.LOSS_S   = tf.reduce_mean(- tf.reduce_sum(self.s_onehot*log(self.s_out), axis=1))
            self.solver_S = tf.train.AdamOptimizer(self.lr_rate1).minimize(
                self.LOSS_S, var_list=self.vars_selecter
            )
                     
    
    ### TRAINING FUNCTIONS        
    def train_mle(self, xs_, xt_, lr_train, k_prob):
        return self.sess.run([self.solver_MLE, self.LOSS_MLE],
                             feed_dict={self.xs: xs_, self.y: xt_,
                                        self.mb_size:np.shape(xs_)[0], self.lr_rate1: lr_train, self.keep_prob: k_prob})
    
    def train_critic(self, xs_, xt_, lr_train, k_prob):
        return self.sess.run([self.solver_L1_critic, self.LOSS_1],
                             feed_dict={self.xs: xs_, self.xt: xt_, 
                                        self.mb_size:np.shape(xs_)[0], self.lr_rate1: lr_train, self.keep_prob: k_prob})
    
    def train_actor(self, xs_, xt_, alpha_, gamma_, lr_train, k_prob):
        return self.sess.run([self.solver_L1_actor, self.LOSS_1, self.LOSS_2, self.LOSS_4],
                             feed_dict={self.xs: xs_, self.xt: xt_,
                                        self.alpha: alpha_, self.gamma: gamma_,
                                        self.mb_size:np.shape(xs_)[0], self.lr_rate2: lr_train, self.keep_prob: k_prob})
    
    def train_selector(self, z_, s_, lr_train, k_prob):
        return self.sess.run([self.solver_S, self.LOSS_S],
                             feed_dict={self.zz: z_, self.s: s_, 
                                        self.lr_rate1: lr_train, self.keep_prob: k_prob})
    
    def train_embedding(self, xs_, xt_, beta_, lr_train, k_prob):   
        return self.sess.run([self.solver_E, self.LOSS_1, self.LOSS_3], 
                             feed_dict={self.xs:xs_, self.xt:xt_,
                                        self.beta:beta_,
                                        self.mb_size:np.shape(xs_)[0], 
                                        self.lr_rate1:lr_train, self.keep_prob:k_prob})

    def get_losses(self, xs_, xt_):   
        return self.sess.run([self.LOSS_1, self.LOSS_2, self.LOSS_3, self.LOSS_4], 
                             feed_dict={self.xs:xs_, self.xt:xt_,
                                        self.mb_size:np.shape(xs_)[0], 
                                        self.keep_prob:1.0})
    
    
    ### blackbox predictions
    def get_risk(self, xs_, xt_, tau_):
        return self.sess.run(self.risk, 
                             feed_dict={self.xs:xs_, self.xt: xt_, self.tau: tau_,
                                        self.mb_size: np.shape(xt_)[0], self.keep_prob:1.0})
        
    def get_lambdas(self, xs_, xt_):
        return self.sess.run(self.ys, 
                             feed_dict={self.xs:xs_, self.xt:xt_, 
                                        self.mb_size: np.shape(xt_)[0], self.keep_prob: 1.0})
    

    def get_final_lambda(self, xs_, xt_):
        return self.sess.run(self.y, 
                             feed_dict={self.xs:xs_, self.xt:xt_, 
                                        self.mb_size: np.shape(xt_)[0], self.keep_prob: 1.0})
    
    ### interpreter estimations
    def get_risk_hat(self, xs_, xt_, tau_):
        return self.sess.run(self.risk_tilde, 
                             feed_dict={self.xs:xs_, self.xt: xt_, self.tau: tau_,
                                        self.mb_size: np.shape(xt_)[0], self.keep_prob:1.0})
       
    def get_z_hats(self, xs_, xt_):
        return self.sess.run(self.z_hats, 
                             feed_dict={self.xs:xs_, self.xt:xt_, 
                                        self.mb_size: np.shape(xt_)[0], self.keep_prob: 1.0})
    
    def get_lambdas_hat(self, xs_, xt_):
        return self.sess.run(self.y_hats, 
                             feed_dict={self.xs:xs_, self.xt:xt_, 
                                        self.mb_size: np.shape(xt_)[0], self.keep_prob: 1.0})
        
    def get_final_lambda_hat(self, xs_, xt_):
        return self.sess.run(self.y_hat, 
                             feed_dict={self.xs:xs_, self.xt:xt_, 
                                        self.mb_size: np.shape(xt_)[0], self.keep_prob: 1.0})
    

    def get_risk_bar(self, xs_, xt_, tau_):
        return self.sess.run(self.risk_bar, 
                             feed_dict={self.xs:xs_, self.xt: xt_, self.tau: tau_,
                                        self.mb_size: np.shape(xt_)[0], self.keep_prob:1.0})
    
    def get_z_bars(self, xs_, xt_):
        return self.sess.run(self.z_bars, 
                             feed_dict={self.xs:xs_, self.xt:xt_, 
                                        self.mb_size: np.shape(xt_)[0], self.keep_prob: 1.0})
        
    def get_lambdas_bar(self, xs_, xt_):
        return self.sess.run(self.y_bars, 
                             feed_dict={self.xs:xs_, self.xt:xt_, 
                                        self.mb_size: np.shape(xt_)[0], self.keep_prob: 1.0})
        
    def get_final_lambda_bar(self, xs_, xt_):
        return self.sess.run(self.y_bar, 
                             feed_dict={self.xs:xs_, self.xt:xt_, 
                                        self.mb_size: np.shape(xt_)[0], self.keep_prob: 1.0})
    
    def get_lambdas_yy(self, z_):
        return self.sess.run(self.yy,
                             feed_dict={self.zz:z_, self.mb_size:np.shape(z_)[0], self.keep_prob:1.0})
        
    def get_zhats_and_pis_m2(self, xs_, xt_):
        return self.sess.run([self.z_hats, self.pis, self.rnn_mask2], 
                             feed_dict={self.xs:xs_, self.xt:xt_, self.mb_size:np.shape(xs_)[0], self.keep_prob:1.0})
    
    def get_s_sample(self, xs_, xt_):
        return self.sess.run([self.s_sample, self.rnn_mask2], 
                             feed_dict={self.xs:xs_, self.xt:xt_, self.mb_size:np.shape(xs_)[0], self.keep_prob:1.0})
    
    def get_zbars_and_pis_m1(self, xs_, xt_):
        return self.sess.run([self.z_bars, self.pis, self.rnn_mask1], 
                             feed_dict={self.xs:xs_, self.xt:xt_, self.mb_size:np.shape(xs_)[0], self.keep_prob:1.0})
    
    def get_zhats_and_pis_m1(self, xs_, xt_):
        return self.sess.run([self.z_hats, self.pis, self.rnn_mask1], 
                             feed_dict={self.xs:xs_, self.xt:xt_, self.mb_size:np.shape(xs_)[0], self.keep_prob:1.0})
    
    def get_zbars_and_pis_m2(self, xs_, xt_):
        return self.sess.run([self.z_bars, self.pis, self.rnn_mask2], 
                             feed_dict={self.xs:xs_, self.xt:xt_, self.mb_size:np.shape(xs_)[0], self.keep_prob:1.0})




### INITIALIZE EMBEDDING AND SELECTOR
from sklearn.cluster import MiniBatchKMeans, KMeans

def initialize_embedding(model, xs, xt, K):
    
    chunksize = 10
    for i in range(int(np.ceil(np.shape(xs)[0]/chunksize))):
        tmp_z_, _, tmp_m_   = model.get_zhats_and_pis_m2(xs[i*chunksize:(i+1)*chunksize], xt[i*chunksize:(i+1)*chunksize])
        tmp_y_              = model.get_lambdas_hat(xs[i*chunksize:(i+1)*chunksize], xt[i*chunksize:(i+1)*chunksize])
        
        if i == 0:
            tmp_z = np.copy(tmp_z_)
            tmp_y = np.copy(tmp_y_)
            tmp_m = np.copy(tmp_m_)
        else:
            tmp_z = np.concatenate([tmp_z, tmp_z_], axis=0)
            tmp_y = np.concatenate([tmp_y, tmp_y_], axis=0)
            tmp_m = np.concatenate([tmp_m, tmp_m_], axis=0)

    z_dim  = np.shape(tmp_z)[-1]
    y_dim  = np.shape(tmp_y)[-1]
    
    tmp_z  = tmp_z[tmp_m == 1]
    tmp_y  = tmp_y[tmp_m == 1]

    km     = KMeans(n_clusters = K, init='k-means++')
#     _      = km.fit(tmp_y)
#     tmp_ey = km.cluster_centers_
#     tmp_s  = km.predict(tmp_y)

#     tmp_e  = np.zeros([K, z_dim])
#     for k in range(K):
# #         tmp_e[k, :] = np.mean(tmp_z[tmp_s == k])
#         tmp_e[k,:] = tmp_z[np.argmin(np.sum(np.abs(tmp_y - tmp_ey[k, :]),axis=1)), :]

    _      = km.fit(tmp_z)
    tmp_ey = km.cluster_centers_
    tmp_s  = km.predict(tmp_z)

    tmp_e  = np.zeros([K, z_dim])
    for k in range(K):
        tmp_e[k, :] = np.mean(tmp_z[tmp_s == k], axis=0)
#         tmp_e[k,:] = tmp_z[np.argmin(np.sum(np.abs(tmp_y - tmp_ey[k, :]),axis=1)), :]

    return tmp_e, tmp_s, tmp_z
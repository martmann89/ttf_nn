import tensorflow as tf
import numpy as np
import keras.backend as K
keras = tf.keras
from config import Config
from typedefs_config import Mlp_cfg, Lstm_cfg, Tcn_cfg
from random import random

def mlp(cfg : Mlp_cfg):
    x_in = keras.layers.Input(shape=(cfg['time_steps_in'], cfg['n_vars']))
    x = x_in
    x = keras.layers.Flatten()(x)
    for i in range(cfg['n_layers']):
        x = keras.layers.Dense(cfg['units'], activation='relu',name=f"dense_layer_{i}")(x)
    x = keras.layers.Dense(cfg['time_steps_out']*len(cfg['quantiles']), kernel_regularizer=keras.regularizers.l2(cfg['l2_lambda']), name="dense_layer_out")(x)
    out_quantiles = tf.reshape(x, (-1, cfg['time_steps_out'], len(cfg['quantiles'])))
    model = keras.Model(inputs=[x_in], outputs=[out_quantiles])
    model.summary()
    return model
   
def tcn(cfg : Tcn_cfg):
    x_in = keras.layers.Input(shape=(cfg['time_steps_in'], cfg['n_vars']))
    x = x_in
    for d in cfg['dilations']:
        x = residual_block(x, dilation=d, n_filters=cfg['n_filters'], kernel_size=cfg['kernel_size'], l2=cfg['l2_lambda'])
        
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(cfg['time_steps_out']*len(cfg['quantiles']), kernel_regularizer=keras.regularizers.l2(cfg['l2_lambda']))(x)
    out_quantiles = tf.reshape(x, (-1, cfg['time_steps_out'], len(cfg['quantiles'])))
    #print(out_quantiles)
    model = keras.Model(inputs=[x_in], outputs=[out_quantiles], name=f'{random()}')
    model.summary()
    
    return model

def residual_block(x, dilation, n_filters, kernel_size, l2):
    x_in = x
    x = keras.layers.Conv1D(filters=n_filters, kernel_size=kernel_size,kernel_initializer='he_normal', dilation_rate=dilation, padding='causal', kernel_regularizer=keras.regularizers.l2(l2), name=f"conv1d_layer_1_{dilation}")(x)
    # x = keras.layers.BatchNormalization(axis=-1)(x)
    x = keras.layers.Activation(activation='relu', name=f"activation_1_{dilation}")(x)
    x = keras.layers.Conv1D(filters=n_filters, kernel_size=kernel_size, kernel_initializer='he_normal', dilation_rate=dilation,padding='causal',kernel_regularizer=keras.regularizers.l2(l2),  name=f"conv1d_layer_2_{dilation}")(x)
    # x = keras.layers.BatchNormalization(axis=-1)(x)
    
    x = keras.layers.Add(name=f'add_{random()}')([x, keras.layers.Conv1D(filters=n_filters,kernel_size=1,dilation_rate=1,kernel_regularizer=keras.regularizers.l2(l2), name=f"conv1d_layer_3_{dilation}")(x_in)])
    # x = x + keras.layers.Conv1D(filters=n_filters,kernel_size=1,dilation_rate=1,kernel_regularizer=keras.regularizers.l2(l2), name=f"conv1d_layer_3_{dilation}")(x_in)
    x = keras.layers.Activation(activation='relu',name=f"activation_2_{dilation}")(x)
    return x

def LSTM(cfg : Lstm_cfg):
    dropout = 0.0
    # Since we use return_sequences=True, we must specify batch shape explicitly
    x_in = keras.layers.Input(batch_shape=(cfg['batch_size'], cfg['time_steps_in'], cfg['n_vars']))
    
    x = x_in
    if cfg['n_layers'] > 1:
        for i in range(cfg['n_layers']-1):
            x = keras.layers.LSTM(cfg['units'],
                            stateful=True, 
                            return_sequences=True,
                            name=f'lstm_{i}'
                            )(x)
            x = keras.layers.Dropout(rate=dropout)(x)
            
    
    x = keras.layers.LSTM(cfg['units'], 
                    stateful=True,
                    name="lstm_final" 
                    )(x)
    
    x = keras.layers.Dense(12,kernel_initializer='he_normal', activation='relu', name=f'add_{random()}') (x)
    x = keras.layers.BatchNormalization(axis=-1)(x)
    x = keras.layers.Dense(cfg['time_steps_out']*len(cfg['quantiles']), activation='linear', name=f'add_{random()}')(x)
    
    out_quantiles = tf.reshape(x, (-1, cfg['time_steps_out'], len(cfg['quantiles'])))
   
    model = keras.Model(inputs=[x_in], outputs=[out_quantiles])
    model.summary()
    
    return model

def _quality_driven(labels,pred, soften=100, lamb=0.01, alpha=0.1):
    N_ = tf.cast(tf.size(labels),tf.float32) #labels.shape[0]

    alpha_ = tf.constant(alpha)
    lamb_ = tf.constant(lamb)

    k_u = tf.sigmoid( ((pred[:,:,-1] + pred[:,:,-1])/2 - labels[:,0]) * soften)
    k_l = tf.sigmoid( (labels[:,0] - (pred[:,:,0] + pred[:,:,0]) / 2 ) * soften)
    k_s = tf.multiply(k_u, k_l)

    k_uh = tf.maximum(0., tf.sign((pred[:,:,-1] + pred[:,:,-1])/2 - labels[:,0]))
    k_lh = tf.maximum(0., tf.sign(labels[:,0] - (pred[:,:,0] + pred[:,:,0])/2))
    k_h = tf.multiply(k_uh, k_lh)
    lhs_soft = tf.divide(tf.reduce_sum( tf.multiply(pred[:,:,-1] - pred[:,:,0],k_s) ), tf.reduce_sum(k_s) + 0.001) # add small noise in case 0
    lhs_hard = tf.divide(tf.reduce_sum(tf.multiply(pred[:,:,-1] - pred[:,:,0], k_h)), (tf.reduce_sum(k_h)+0.001))
    PICP_soft = tf.reduce_mean(k_s)
    PICP_hard = tf.reduce_mean(k_h)
    # penalty = 10000*tf.maximum(0., pred[:,:,0] - pred[:,:,-1])
    rhs_soft = lamb_ * N_ / (alpha_ * (1 - alpha_))* tf.square(tf.maximum(0., (1. - alpha_) - PICP_soft)) # + penalty
    rhs_hard = lamb_ * N_ / (alpha_ * (1 - alpha_))* tf.square(tf.maximum(0., (1. - alpha_) - PICP_hard)) # + penalty
    return lhs_hard + rhs_soft

def _pin_loss(labels, pred, quantiles):
    loss = []
    for i,q in enumerate(quantiles):
        error = tf.subtract(labels,pred[:,:,i])
        loss_q = tf.reduce_mean(tf.maximum(q*error,(q-1)*error))
        loss.append(loss_q)
    L = tf.convert_to_tensor(loss)
    total_loss = tf.reduce_mean(L)
    return total_loss

def pi_cov(y_true, y_pred):
    """ 
    Get average coverage of prediction intervals
    """
    coverage = tf.reduce_mean(
        tf.cast((y_true >= y_pred[:,:,0])&(y_true <= y_pred[:,:,-1]), tf.float32))
    return coverage


def pi_len(y_true, y_pred):
    """ 
    Get length of prediction intervals
    """
    avg_length = tf.reduce_mean(tf.abs(y_pred[:,:,-1] - y_pred[:,:,0]))
    #avg_length = avg_length/(tf.reduce_max(y_true) - tf.reduce_min(y_true))
    return  avg_length


class keras_model():
    
    def __init__(self, cfg : Config):
        
        self.cfg = cfg
        self.NN_cfg = cfg.model_cfg
        if cfg.training_cfg['model_type'] == 'lstm':
            self.model = LSTM(self.NN_cfg)
        elif cfg.training_cfg['model_type'] == 'tcn':
            self.model = tcn(self.NN_cfg)
        elif cfg.training_cfg['model_type'] == 'mlp':
            self.model = mlp(self.NN_cfg)
        else:
            raise ValueError("model_type must be 'lstm', 'mlp' or 'tcn'")
            
    def fit(self, train_x, train_y, val_x, val_y, epochs=100, patience=30, verbose=1, start_from_epoch=20, learning_rate=0.001,weights=None):
        self.model.summary()

        tf_train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y)).repeat().batch(self.NN_cfg['batch_size'])
        val_data = tf.data.Dataset.from_tensor_slices((val_x, val_y)).repeat().batch(self.NN_cfg['batch_size'])
        
        # Since we use repeat(), we must specify the number of times we draw a bach in an epoch
        TRAIN_STEPS = int(np.ceil(train_x.shape[0]/self.NN_cfg['batch_size']))
        VAL_STEPS = int(np.ceil(val_x.shape[0]/self.NN_cfg['batch_size']))
        
        if self.NN_cfg['loss_func'] == 'qd':
            loss =[lambda y_true, y_pred: _quality_driven(y_true, y_pred, soften=self.NN_cfg['eta'], lamb=self.NN_cfg['lamb'], alpha=self.NN_cfg['alpha'])]
        elif self.NN_cfg['loss_func'] == 'pb':
            loss = [lambda y_true, y_pred: _pin_loss(y_true, y_pred, self.NN_cfg['quantiles'])]
        else:
            raise ValueError("loss function must be 'pb' or 'qd' !")
        self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss= loss,
                metrics=[pi_cov, pi_len],
                run_eagerly=False)
        
        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            start_from_epoch=start_from_epoch,
            restore_best_weights=True,
            verbose=1
        )
        
        history = self.model.fit(tf_train_data,
                       validation_data=val_data,
                       epochs=epochs,
                       steps_per_epoch=TRAIN_STEPS,
                       validation_steps=VAL_STEPS,
                       callbacks=[es],
                       verbose=verbose)
        
        return history
    
    
    def transform(self, data_x):
        
        tf_data = tf.data.Dataset.from_tensor_slices(data_x).repeat().batch(self.NN_cfg['batch_size'])
        it = iter(tf_data)
        n_steps = int(np.ceil(data_x.shape[0]/self.NN_cfg['batch_size']))
        
        preds =[]
        for _ in range(n_steps):
            batch = next(it)
            preds.append(self.model(batch))
            
        preds = np.concatenate(preds, axis=0)
        preds = preds[:data_x.shape[0],:,:]
        
        return preds

def regression_model(cfg : Config):
    
    if cfg.training_cfg['model_type'] in ['lstm', 'tcn', 'mlp']:
        return keras_model(cfg)
    else:
        raise ValueError("model_type must be 'lstm', 'tcn', or 'mlp'")
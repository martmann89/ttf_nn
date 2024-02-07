import numpy as np
import matplotlib.pyplot as plt
from config import Config
import os
plt.rcParams["font.family"] = "serif"

def compute_coverage_len(y_test, y_lower, y_upper, scaler, verbose=False, eta=30, mu=0.95):
    """ 
    Compute average coverage and length of prediction intervals
    """
   
    in_the_range = np.sum((y_test.flatten() >= y_lower) & (y_test.flatten() <= y_upper))
    
    coverage = in_the_range / np.prod(y_test.shape)
    
    avg_length = np.mean(abs(y_upper - y_lower))
    y_lower = np.reshape(y_lower,[len(y_lower),1])
    y_upper = np.reshape(y_upper,[len(y_upper),1])
    avg_length_transformed = np.mean(abs(scaler.inverse_transform_y(y_upper) - scaler.inverse_transform_y(y_lower)))
    cwc = (1-avg_length)*np.exp(-eta*(coverage-mu)**2)
    mis = compute_mis(scaler.inverse_transform_y(y_test),scaler.inverse_transform_y(y_lower),scaler.inverse_transform_y(y_upper))
    if verbose==True:
        print(f"PI coverage: {coverage*100:.4f}%, PI avg. length: {avg_length:.4f}, CWC: {cwc:.4f}")
    else:
        return coverage, avg_length, avg_length_transformed, cwc, mis

def compute_mis(y_test, y_lower, y_upper):
    alpha = 0.05
    score = []
    for i in range(len(y_test)):
        diff = y_upper[i] - y_lower[i]
        pen_l = 2/alpha*(y_lower[i]-y_test[i]) * int(y_test[i] < y_lower[i])
        pen_u = 2/alpha*(y_test[i] - y_upper[i]) * int(y_test[i] > y_upper[i])
        score.append(diff + pen_l + pen_u)
    score_total = np.mean(score)
    return score_total        

def asym_nonconformity(label, low, high):
    """
    Compute the asymetric conformity score
    """
    error_high = label - high 
    error_low = low - label
    return error_low, error_high

def plot_PIs(true, pred_mean, PI_low=None, PI_hi=None, 
             conf_PI_low=None, conf_PI_hi=None, 
             x_lims=None, scaler=None, title=None,
             label_pi=None,save_path=None, text=None, PI_all_retrains=None, PI_all_conf_retrains = None):
    
    if scaler:
        true = scaler.inverse_transform_y(true)
        pred_mean = scaler.inverse_transform_y(pred_mean)
    true = true.flatten()
    pred_mean = pred_mean.flatten()
    
    plt.set_cmap("tab10")
    plt.cm.tab20(0)
    fig = plt.figure(figsize=(14, 3.5))
    plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.1)
    plt.plot(np.arange(true.shape[0]), true, label='True', color='k')
    ax = fig.axes
    minor_ticks = np.arange(0, len(true),1)
    ax[0].set_xticks(minor_ticks, minor=True)
    ax[0].grid(which='both', linewidth=0.3)
    plt.plot(pred_mean, label='Pred', color=plt.cm.tab10(1))
    
    if PI_all_retrains is not None:
        max = PI_all_retrains.shape[-1]
        for k in range(max):
            if scaler:
                pi_low = scaler.inverse_transform_y(PI_all_retrains[:,:,0,k])
                pi_hi = scaler.inverse_transform_y(PI_all_retrains[:,:,-1,k])
            pi_low = pi_low.flatten()
            pi_hi = pi_hi.flatten()
            plt.fill_between(np.arange(true.shape[0]), pi_low, pi_hi, alpha=1/(max - k), color='blue')
    
    elif PI_all_conf_retrains is not None:
        max = PI_all_conf_retrains.shape[-1]
        for k in range(max):
            if scaler:
                pi_low_conf = scaler.inverse_transform_y(PI_all_conf_retrains[:,:,0,k])
                pi_hi_conf = scaler.inverse_transform_y(PI_all_conf_retrains[:,:,-1,k])
            pi_low_conf = pi_low_conf.flatten()
            pi_hi_conf = pi_hi_conf.flatten()
            plt.fill_between(np.arange(true.shape[0]), pi_low_conf, pi_hi_conf, alpha=1/(max - k), color='red')

    elif conf_PI_low is not None:
        
        if scaler:
            conf_PI_low = scaler.inverse_transform_y(conf_PI_low)
            conf_PI_hi = scaler.inverse_transform_y(conf_PI_hi)
            PI_low = scaler.inverse_transform_y(PI_low)
            PI_hi = scaler.inverse_transform_y(PI_hi)
        conf_PI_hi = conf_PI_hi.flatten()
        conf_PI_low = conf_PI_low.flatten()
        PI_hi = PI_hi.flatten()
        PI_low = PI_low.flatten()    
        plt.fill_between(np.arange(true.shape[0]), conf_PI_low, conf_PI_hi, alpha=0.3, label='Conformalized')
        plt.plot(PI_low, label='original', color=plt.cm.tab10(0), linestyle='dashed')
        plt.plot(PI_hi, color=plt.cm.tab10(0), linestyle='dashed')
    
        
    if (conf_PI_low is None) and (PI_low is not None):
        if scaler:
            PI_low = scaler.inverse_transform_y(PI_low)
            PI_hi = scaler.inverse_transform_y(PI_hi)
            
        if label_pi is None:
            label_pi = 'PI'
        PI_hi = PI_hi.flatten()
        PI_low = PI_low.flatten()  
        plt.fill_between(np.arange(true.shape[0]), PI_low, PI_hi, alpha=0.3, label=label_pi)
        
    if x_lims is not None:
        plt.xlim(x_lims)
    plt.legend(loc='upper right')
   # plt.grid()
    
    if title is not None:
        plt.title(title)
    if text is not None:
        form_text = ""
        for key,value in text.items():
            if str(value) == value:
                form_text = '\n'.join([form_text, key + " : " + value])
            else:
                form_text = '\n'.join([form_text, key + " : " + str(round(value,4))])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        fig.text(0.01, 0.99, form_text, fontsize=7,
        verticalalignment='top', bbox=props)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    
def plot_history(history, alpha):
    
    hist_dict = history.history
    
    plt.set_cmap("tab10")
    plt.cm.tab20(0)
    fig, axs = plt.subplots(1,3, figsize=(9,2.5))
    
    for ax in axs:
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
    axs[0].semilogy(hist_dict['loss'], label='Training', color='k')
    axs[0].semilogy(hist_dict['val_loss'], label='Validierung', color='g')
    axs[0].set_xlabel("Epochen", size=22)
    axs[0].set_ylabel("Verlust", size=22)

    axs[0].legend(fontsize=22)
    
    axs[1].axhline(y=(1 - alpha), color='r', linestyle='-')
    axs[1].plot(hist_dict['pi_cov'], label='Training', color='k')
    axs[1].plot(hist_dict['val_pi_cov'], label='Validierung', color='g')
    axs[1].set_xlabel("Epochen", size=22)
    axs[1].set_ylabel("Abdeckung", size=22)
    axs[1].legend(loc='lower right',fontsize=22)
    
    axs[2].plot(hist_dict['pi_len'], label='Training', color='k')
    axs[2].plot(hist_dict['val_pi_len'], label='Validierung', color='g')
    axs[2].set_xlabel("Epochen", size=22)
    axs[2].set_ylabel("Intervalllänge", size=22)
    axs[2].legend(loc='lower right',fontsize=22)
    axs[2].legend(fontsize=22)
    
    plt.tight_layout()
    plt.show()
    
def plot_ts(dfs):
    plt.figure(figsize=(12, 3.5))
    for i in range(len(dfs)):
        plt.plot(np.array(dfs[i].index),dfs[i]['Passengers'])
        # plt.plot(np.array(dfs[i].index),dfs[i]['Base'])
        # plt.plot(np.array(dfs[i].index),dfs[i]['Oil'])
    plt.show()

def plot_with_date(df):
    plt.figure(figsize=(12, 3.5))
    plt.plot(df['Price'], color='k', linewidth=0.5)
    plt.rcParams['text.usetex'] = True
    #plt.gcf().autofmt_xdate()
    dates = df["Date"].to_numpy()
    dates = np.concatenate((dates[::320],[dates[-1]]))
    idx = df.index[::320]
    idx = np.concatenate((idx,[df.index[-1]]))
    plt.xticks(idx,dates,fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel('Date', fontsize=8)
    plt.ylabel('Price (Euro/MWh)', fontsize=8)
    plt.title("Dataset: TTF Front Month")
    pos = [1, 2, 4, 5, 7, 8]
    plt.violinplot(df['Price'], pos, points=20, widths=0.3,
                     showmeans=True, showextrema=True, showmedians=True)
    plt.show() 

def set_plot_text(cfg : Config, values : dict):
    NN_cfg =cfg.model_cfg
    text = {}
    text['batch_size'] = NN_cfg['batch_size']
    text['learning_rate'] = NN_cfg['learning_rate']
    text['l2_lambda'] = NN_cfg['l2_lambda']
    text['alpha'] = NN_cfg['alpha']
    text['loss_func'] = NN_cfg['loss_func']
    text['time_steps_in'] = NN_cfg['time_steps_in']
    text['time_steps_out'] = NN_cfg['time_steps_out']
    text['n_vars'] = NN_cfg['n_vars']

    model_type = cfg.training_cfg['model_type']

    if model_type == "mlp":
        text['units'] = NN_cfg['units']
        text['n_layers'] = NN_cfg['n_layers']
    elif model_type == "lstm":
        text['units'] = NN_cfg['units']
        text['n_layers'] = NN_cfg['n_layers']
    elif model_type == "tcn":
        text['kernel_size'] = NN_cfg['kernel_size']
        text['n_filters'] = NN_cfg['n_filters']
        text['dilations'] = ','.join(str(e)  for e in NN_cfg['dilations'])
    
    if NN_cfg['loss_func'] == "qd":
        text['eta_qd'] = NN_cfg["eta"]
        text['lamb_qd'] = NN_cfg["lamb"]

    text.update(values)

    return text

def get_next_model_number(path):

    dirs = sorted([int(x) for x in os.listdir(path)])
    if len(dirs) == 0:
        model_number = 1
    else:
        num = int(dirs[-1])
        model_number = num + 1
    return model_number


     


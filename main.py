import copy
import multiprocessing
from multiprocessing.pool import ThreadPool
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import utils.utils as utils
import utils.data_preprocessing as preprocessing
import numpy as np
import pandas as pd
from models.predict import predict
from utils.utils import set_plot_text
from config import Config

# config.py is where to define used model, its architecture and all other hyperparameters
# utils.datapreprocessing.py is for scaling and windowing
# utils.postprocessing should be run after this script for computing stats like PICP, MPIW and MIS

def compute_PI(cfg : Config):
    """
        Main function. Returns dictionary with all results so they can be plotted in main thread.
    """
    # add index for identification
    i = cfg.model_cfg["idx"]
    # path for saving
    base_path = cfg.model_cfg['base_path']
    # get nof_retrains
    if cfg.training_cfg['retrain_period'] == 0:
        nof_retrains = 1
    else:
        nof_retrains = cfg.data_cfg['nof_days_to_pred'] // cfg.training_cfg['retrain_period']
    # array for stacking predictions of every retrain
    PI_final = np.ndarray((cfg.data_cfg['nof_days_to_pred'] - cfg.model_cfg['time_steps_in'],1, len(cfg.model_cfg['quantiles'])))

    text = {}
    test_y_initial = None
    test_date_initial = None
    scaler_initial = None
    weights = None

    for k in range(nof_retrains):
        print(f"Starting thread {i} with retrain no. {k}")
        days_to_predict =  cfg.data_cfg['nof_days_to_pred'] - k * cfg.training_cfg['retrain_period']
        # get training, validation and test dataset
        train_df, val_df, test_df =  cfg.get_datasets(days_to_predict)
        # get windowed training, validation and test dataset
        train_data, val_x, val_y, test_x, test_y, Scaler, train_date, val_date, test_date = preprocessing.data_windowing(df=train_df, 
                                                                                        val_data=val_df,
                                                                                        test_data=test_df,
                                                                                        time_steps_in=cfg.model_cfg['time_steps_in'], 
                                                                                        time_steps_out=cfg.model_cfg['time_steps_out'], 
                                                                                        label_columns=cfg.data_cfg['label_column'])
        
        if k == 0:
            # save whole test data for final result
            test_y_initial = test_y
            test_date_initial = test_date
            scaler_initial = Scaler
        
        # get nof different features (in this thesis only 1)
        cfg.model_cfg['n_vars'] = test_x.shape[2] 
        # get prediction intervals and hist for nof epochs
        PI, hist = predict(train_data,val_x,val_y,test_x, test_y, cfg, weights)
        # stack PI of current retrain in final array
        PI_final[k*cfg.training_cfg['retrain_period']:,:,:] = PI
    # save results here
    csv_path = base_path + f'intervals_prevs_{cfg.model_cfg["time_steps_in"]}_run_{i}.csv'

    result_df = pd.DataFrame({
        'low' : scaler_initial.inverse_transform_y(PI_final[:,:,0]).flatten(),
        'high' : scaler_initial.inverse_transform_y(PI_final[:,:,-1]).flatten(),
        'mean' : scaler_initial.inverse_transform_y(PI_final[:,:,1]).flatten(),
        'actual' :scaler_initial.inverse_transform_y(test_y_initial).flatten(),
        'date' : test_date_initial.flatten()
    })

    result_df.to_csv(csv_path, sep=",")
    # compute stats
    coverage, avg_length, avg_length_trans, cwc, mis = utils.compute_coverage_len(test_y_initial, PI_final[:,:,0].flatten(), PI_final[:,:,-1].flatten(), scaler_initial, verbose=False)
    print(f"PI coverage: {coverage*100:.4f}%, PI avg. length: {avg_length:.4f}, PI avg. length transformed: {avg_length_trans:.4f}, CWC: {cwc:.4f}")

    text=set_plot_text(cfg,{"epochs" : len(hist.history['loss'])-cfg.model_cfg['patience'], "coverage" : coverage, "avg_length" : avg_length_trans, "mis" : mis, "cwc" : cwc})
    
    return {"coverage" : coverage,  "avg_length" : avg_length_trans, "PI" : PI_final,  "mis" : mis, "cwc" : cwc, "labels" : test_y_initial, "scaler" : scaler_initial, "text" : text, "idx" : i}

def makeConfigs(cfg : Config, nof_runs, base_path):
    """
        Makes one config for each run, so it can be run in parallel. Returns config array
    """
    configs = []
    for i in range(nof_runs):
        curr_cfg = Config(cfg.training_cfg["dataset"], cfg.training_cfg["model_type"]) 
        curr_cfg = copy.deepcopy(curr_cfg)
        NN_config = curr_cfg.model_cfg
        NN_config['idx'] = i
        NN_config['base_path'] = base_path
        configs.append(curr_cfg)
    return configs

# main thread
if __name__ == '__main__':
    # get config from config.py
    # here define dataset and the model config to use
    # before_shock, during_shock, after_shock
    # mlp, tcn, lstm
    cfg = Config("during_shock","mlp")
    nof_runs = 10
    results = []
    # store results here
    base_path = f'./predictions/{cfg.training_cfg["dataset"]}/{cfg.model_cfg["loss_func"]}/{cfg.training_cfg["model_type"]}/'
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    # get increasing number = next number in save folder
    model_number = utils.get_next_model_number(base_path)
    base_path += f'{model_number}/'
    os.makedirs(base_path)
    text = None
    n = multiprocessing.cpu_count()
    # make own config for each thread
    configs = makeConfigs(cfg, nof_runs, base_path)
    if len(configs) < n:
        n = len(configs)
    # multithreaded execution
    with ThreadPool(n) as pool:
        for res in pool.map(compute_PI,configs):
            results.append(res)
    # make plots and save stats file
    
    coverage_sum = 0
    avg_length_sum = 0
    mis_sum = 0
    cwc_sum = 0
    
    for res in results:
        text = res["text"]
        utils.plot_PIs(res["labels"], res["PI"][:,:,1],
                            res["PI"][:,:,0],  res["PI"][:,:,-1],
                            scaler=res["scaler"], title=f'Single CQR - {cfg.training_cfg["model_type"]}',save_path=base_path + f'final_{res["idx"]}',text=res["text"])
        coverage_sum += res["coverage"]
        avg_length_sum += res["avg_length"]
       
        mis_sum += res["mis"]
        cwc_sum += res["cwc"]
 
    text["coverage_sum"] = coverage_sum / nof_runs

    text["avg_length_sum"] = avg_length_sum / nof_runs

    text["mis_sum"] = mis_sum / nof_runs
    text["cwc_sum"] = cwc_sum / nof_runs

    stats_df = pd.DataFrame(text, index=[0])
    stats_df.to_csv(base_path + 'stats.csv', sep=";")
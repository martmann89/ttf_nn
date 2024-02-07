from multiprocessing.pool import ThreadPool
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import utils.data_loaders
from utils.utils import compute_coverage_len
from utils.utils import set_plot_text
import utils.data_preprocessing
from models.predict import predict
import numpy as np
import pandas as pd

from config import Config
import copy
import multiprocessing
import random

def compute_PI(cfg : Config):
    NN_cfg = cfg.model_cfg
    index = NN_cfg['idx']

    train_df, val_df, test_df =  cfg.get_datasets(cfg.data_cfg["nof_days_to_pred"])
    train_data, val_x, val_y, test_x, test_y, Scaler, train_date, val_date, test_date = utils.data_preprocessing.data_windowing(df=train_df, 
                                                                                        val_data=val_df,
                                                                                        test_data=test_df, 
                                                                                        time_steps_in=NN_cfg['time_steps_in'], 
                                                                                        time_steps_out=NN_cfg['time_steps_out'], 
                                                                                        label_columns=cfg.data_cfg["label_column"])


    NN_cfg['n_vars'] = test_x.shape[2] 
    #    NN_cfg['time_steps_out'] = test_y.shape[1]
   # csv_path_statistics = base_path + f'stats_{index}.csv'

    coverage_sum = 0
    avg_length_sum = 0
    avg_length_trans_sum = 0
    cwc_sum = 0
    text = {}
    weights = None

    nof_runs = 1
    for i in range(1,nof_runs+1):
        print(f'start fitting model no {index}, run no. {i}')
        PI, hist = predict(train_data,val_x,val_y,test_x, test_y, cfg, weights)
        coverage, avg_length, avg_length_trans, cwc, mis = compute_coverage_len(test_y, PI[:,:,0].flatten(), PI[:,:,-1].flatten(), Scaler, verbose=False)
        print(f"PI coverage: {coverage*100:.4f}%, PI avg. length: {avg_length:.4f}, PI avg. length transformed: {avg_length_trans:.4f}, CWC: {cwc:.4f}")
        
        coverage_sum += coverage
        avg_length_sum += avg_length
        avg_length_trans_sum += avg_length_trans 
        cwc_sum += cwc

    text=set_plot_text(cfg,{"epochs" : len(hist.history['loss'])-NN_cfg['patience'], "coverage" : coverage, "avg_length" : avg_length_trans, "idx" : index})
    text["coverage_sum"] = coverage_sum / nof_runs
    text["avg_length_sum"] = avg_length_trans_sum / nof_runs


    #stats_df = pd.DataFrame(text, index=[0])
    #stats_df.to_csv(csv_path_statistics, sep=";")

    return text

# make random filled configs for the different types of networks
def makeConfigs(cfg : Config):
    nof_configs = 5
    configs = []

    params = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    etas =[50,60,70,100,120,130,150]
    lambs = [0.01, 0.05, 0.075, 0.1, 0.5 , 0.75]
    dils = [[1],[1,2],[1,2,4]]
    lstm_units = [3,6,9,12]

    for i in range(0,nof_configs):
        curr_cfg = Config(cfg.training_cfg["dataset"], cfg.training_cfg["model_type"])
        curr_cfg = copy.deepcopy(curr_cfg)
        NN_config = curr_cfg.model_cfg
        NN_config['l2_lambda'] = params[random.randint(0,len(params)-1)]
        NN_config['learning_rate'] =  params[random.randint(0,len(params)-1)]
        NN_config['batch_size'] = 2**(random.randint(2,5))
        NN_config['time_steps_in'] = random.randint(1,4)

        if curr_cfg.training_cfg["model_type"] == 'mlp':
            NN_config['n_layers'] = random.randint(1,3)
            NN_config['units'] = 2**(random.randint(4,7))
            NN_config['eta'] = etas[random.randint(0,len(etas)-1)]
            NN_config['lamb'] = lambs[random.randint(0,len(lambs)-1)]
        elif curr_cfg.training_cfg["model_type"] == 'lstm':
            NN_config['units'] = lstm_units[random.randint(0,len(lstm_units)-1)]
            NN_config['n_layers'] = random.randint(1,3)
        elif curr_cfg.training_cfg["model_type"] == 'tcn':
            NN_config['eta'] = etas[random.randint(0,len(etas)-1)]
            NN_config['lamb'] = lambs[random.randint(0,len(lambs)-1)]
            NN_config['kernel_size'] = random.randint(3,7)
            NN_config['dilations'] = dils[random.randint(0,2)]
            NN_config['n_filters'] =  2**(random.randint(1,9))
        
        NN_config['idx'] = i
        configs.append(curr_cfg)
    return configs


if __name__ == '__main__':
    # choose dataset and model type
    cfg = Config("during_shock","mlp")
    # path for storing the results
    base_path = f'./hypertuning_predictions/{cfg.training_cfg["dataset"]}/{cfg.model_cfg["loss_func"]}/{cfg.training_cfg["model_type"]}/'
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    n = multiprocessing.cpu_count()
    configs = makeConfigs(cfg)
    results = []
    with ThreadPool(n) as pool:
        for res in pool.map(compute_PI,configs):
            results.append(res)
    # save results
    for res in results:
        stats_df = pd.DataFrame(res, index=[0])
        stats_df.to_csv(base_path + f'stats_{res["idx"]}.csv', sep=";")
    
    csv_files = []
    for f in os.listdir(f'{base_path}'):
        if os.path.isfile(os.path.join(f'{base_path}', f)):
            name,ext = os.path.splitext(os.path.join(f'{base_path}', f))
            if ext == ".csv":
                csv_files.append(f)
    results_dic = []
    # print results id with coverage > min_cov
    min_cov = 0.7
    for file in csv_files:
        name,ext = os.path.splitext(file)
        index = int(name.split("_")[1])
        df = pd.read_csv(f'{base_path}{file}', sep=";")
        results = pd.DataFrame(df, columns=['coverage_sum','avg_length_sum']).to_numpy()
        results = np.squeeze(results)
        if results[0] > min_cov:
            results_dic.append({"cov" : results[0], "len" : results[1], "idx" : index})
    for res in results_dic:
        print(res)
        print('\n')

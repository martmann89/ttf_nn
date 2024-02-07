import tensorflow as tf
import os
import numpy as np
import pandas as pd
from datetime import datetime
from typedefs_config import Cfg, Data_cfg
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def pp_electricity(version='new'):
    csv_path_elec = './datasets/PHELIX.csv'
    if version == 'new':
        save_path = './datasets/TTF_FM_new_PHELIX.csv'
        csv_path_gas = './datasets/TTF_FM_new.csv'
    else:
        csv_path_gas = './datasets/TTF_FM_old.csv'
        save_path = './datasets/TTF_FM_old_PHELIX.csv'
    data = pd.read_csv(csv_path_elec, sep=",")
    data2 =  pd.read_csv(csv_path_gas, sep=";")
    result_list = []
    for r in data.index:
        d = datetime.strptime(data.at[r,"Date"],"%d.%m.%Y").date()
        #print("phelix date = ", d.day,d.month,d.year)
        for r2 in data2.index:   
            d2 = datetime.strptime(data2.at[r2,"Date"],"%d.%m.%Y").date()
            #print("gas date = ", d2.day,d2.month,d2.year)
            if d == d2:
                print("phelix date = ", d.day,d.month,d.year)
                row = {"Date" : d, "Price" : data2.at[r2,"Price"], "Base" : data.at[r, "Base"]}
                result_list.append(row)
                break
    result = pd.DataFrame(data=result_list)
    result.to_csv(save_path,sep=";")

def pp_electricity_oil(version='old'):
    csv_path_oil = './datasets/brent_eu.csv'
    if version == 'new':
        save_path = './datasets/TTF_FM_new_PHELIX_brent.csv'
        csv_path_gas = './datasets/TTF_FM_new_PHELIX.csv'
    else:
        csv_path_gas = './datasets/TTF_FM_old_PHELIX.csv'
        save_path = './datasets/TTF_FM_old_PHELIX_brent.csv'

    data =  pd.read_csv(csv_path_gas, sep=";")
    data2 = pd.read_csv(csv_path_oil, sep=",")
    result_list = []
    for r in data.index:
        d = datetime.strptime(data.at[r,"Date"],"%Y-%m-%d").date()
        for r2 in data2.index:
            d2 = datetime.strptime(data2.at[r2,"DATE"],"%Y-%m-%d").date()
            if d == d2:
                oilData = data2.at[r2, "DCOILBRENTEU"]
                if  oilData == ".":
                    i1 = 1
                    i2 = 1
                    while data2.at[r2-i1, "DCOILBRENTEU"] == ".":
                        i1 += 1
                    while data2.at[r2+i2, "DCOILBRENTEU"] == ".":
                        i2 += 1
                    oilData = ( float(data2.at[r2-i1, "DCOILBRENTEU"]) + float(data2.at[r2+i2, "DCOILBRENTEU"]) ) / 2

                row = {"Date" : d, "Price" : data.at[r,"Price"], "Base" : data.at[r, "Base"], "Oil" : oilData}
                result_list.append(row)
    result = pd.DataFrame(data=result_list)
    result.to_csv(save_path,sep=";")


def get_gas_elec_oil_data(cfg : Data_cfg, test_days):
    split_perc = cfg['split_train_val']
    usage_perc = cfg['usage_perc']
    version = cfg['dataset']

    if version == "new":
        csv_path = './datasets/TTF_FM_new_PHELIX_brent.csv'
    else:
        csv_path = './datasets/TTF_FM_old_PHELIX_brent.csv'
    data = pd.read_csv(csv_path, sep=";")
    df = pd.DataFrame(data, columns=['Price','Base',"Oil"])
    index = range(len(df))
    df['hour_list'] = index
    df.set_index(df.pop('hour_list'))
    if test_days is not None:
        n = len(df)-test_days
        assert(split_perc[0] + split_perc[1] == 1)
        df_test = df[len(df) - test_days:]
    else:
        n = int(len(df)*usage_perc)
        df_test = df[int((split_perc[0] + split_perc[1])*n):n]
    df_train = df[0:int(split_perc[0]*n)]
    df_val = df[int(split_perc[0]*n):int((split_perc[0] + split_perc[1])*n)]
    
    return df_train, df_val, df_test

def get_gas_elec_data(cfg : Data_cfg, test_days):
    split_perc = cfg['split_train_val']
    usage_perc = cfg['usage_perc']
    version = cfg['dataset']

    if version == "new":
        csv_path = './datasets/TTF_FM_new_PHELIX.csv'
    else:
        csv_path = './datasets/TTF_FM_old_PHELIX.csv'
    data = pd.read_csv(csv_path, sep=";")
    elec_data = data["Base"].to_numpy()
    elec_data = savgol_filter(elec_data, 11, 3) # window size 51, polynomial order 3
    data["Base"] = elec_data
    df = pd.DataFrame(data, columns=['Price','Base','Date'])
    index = range(len(df))
    df['hour_list'] = index
    df.set_index(df.pop('hour_list'))
    if test_days is not None:
        n = len(df)-test_days
        assert(split_perc[0] + split_perc[1] == 1)
        df_test = df[len(df) - test_days:]
    else:
        n = int(len(df)*usage_perc)
        df_test = df[int((split_perc[0] + split_perc[1])*n):n]
    df_train = df[0:int(split_perc[0]*n)]
    df_val = df[int(split_perc[0]*n):int((split_perc[0] + split_perc[1])*n)]
    
    return df_train, df_val, df_test
    
def get_gas_data(cfg : Data_cfg, test_days):
    split_perc = cfg['split_train_val']
    usage_perc = cfg['usage_perc']
    version = cfg['dataset']

    if version == "new":
        csv_path = './datasets/TTF_FM_new.csv'
    else:
        csv_path = './datasets/TTF_FM_old.csv'
    data = pd.read_csv(csv_path, sep=";")
    df = pd.DataFrame(data, columns=['Price','Date'])
    index = range(len(df))
    df['hour_list'] = index
    df.set_index(df.pop('hour_list'))
    if test_days is not None:
        n = int(len(df)*usage_perc)-test_days
        assert(split_perc[0] + split_perc[1] == 1)
        df_test = df[n:n+test_days]
    else:
        n = int(len(df)*usage_perc)
        df_test = df[int((split_perc[0] + split_perc[1])*n):n]
    df_train = df[0:int(split_perc[0]*n)]
    df_val = df[int(split_perc[0]*n):int((split_perc[0] + split_perc[1])*n)]
    
    return df_train, df_val, df_test

def get_data_with_date():
    csv_path = './datasets/TTF_FM_new.csv'
    data = pd.read_csv(csv_path, sep=";")
    result_list=[]
    for r in data.index:
        d = datetime.strptime(data.at[r,"Date"],"%d.%m.%Y").date()
        d = str('{:02d}'.format(d.month)) + '-' + str(d.year)
        row = {"Date" : d, "Price" : data.at[r,"Price"]}
        result_list.append(row)
    result = pd.DataFrame(data=result_list)
    return result
    

def get_data(cfg : Cfg, test_days):
    if cfg["n_vars"] == 3:
        return get_gas_elec_oil_data(cfg, test_days)
    else:
        return get_gas_data(cfg, test_days)
    
def get_airline_data(cfg : Cfg, test_days):
    split_perc = [0.9,0.1]
    usage_perc = 1
    csv_path = './datasets/airline-passengers.csv'
    data = pd.read_csv(csv_path, sep=",")
    df = pd.DataFrame(data, columns=['Passengers'])
    index = range(len(df))
    df['hour_list'] = index
    df.set_index(df.pop('hour_list'))
    if test_days is not None:
        n = len(df)-test_days
        assert(split_perc[0] + split_perc[1] == 1)
        df_test = df[len(df) - test_days:]
    else:
        n = int(len(df)*usage_perc)
        df_test = df[int((split_perc[0] + split_perc[1])*n):n]
    df_train = df[0:int(split_perc[0]*n)+1]
    df_val = df[int(split_perc[0]*n):int((split_perc[0] + split_perc[1])*n)+1]
    return df_train, df_val, df_test
    
def get_test_data(cfg : Cfg, test_days):
    split_perc = cfg['split_train_val']
    usage_perc = cfg['usage_perc']
    version = cfg['dataset']
    if version == "new":
        csv_path = './datasets/TTF_FM_new.csv'
    else:
        csv_path = './datasets/TTF_FM_old.csv'
    data = pd.read_csv(csv_path, sep=";")
    df = pd.DataFrame(data, columns=['Price'])
    arr = df['Price'][2550:2855].to_numpy()
    f2 = interp1d(range(0,len(arr),4), arr[::4], kind='linear')
    interp = f2(range(len(arr))) + 10
    to_add = pd.DataFrame(interp, columns=['Price'])
    df = pd.concat([df.iloc[:2000],to_add,df[2000:]], ignore_index=True)
    index = range(len(df))
    df['hour_list'] = index
    df.set_index(df.pop('hour_list'))
    if test_days is not None:
        n = len(df)-test_days
        assert(split_perc[0] + split_perc[1] == 1)
        df_test = df[len(df) - test_days:]
    else:
        n = int(len(df)*usage_perc)
        df_test = df[int((split_perc[0] + split_perc[1])*n):n]
    df_train = df[0:int(split_perc[0]*n)]
    df_val = df[int(split_perc[0]*n):int((split_perc[0] + split_perc[1])*n)]
   
    return df_train, df_val, df_test

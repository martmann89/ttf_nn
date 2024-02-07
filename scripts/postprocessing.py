import numpy as np
import pandas as pd
import os
import math
from datetime import datetime

def get_cwc(PICP, NMPIW=1 , eta=50, mu=0.90):
    """
        Calculates CWC
    """
    is_not_covered = 1
    if PICP >= mu:
        is_not_covered = 0
    return NMPIW*(1 + is_not_covered * math.e**(-eta*(PICP - mu)))

def int_score(labels,y_lo,y_hi):
    """
        Calculates MIS
    """
    labels.flatten()
    y_lo.flatten()
    y_hi.flatten()
    alpha = 0.1
    score = []
    for i in range(len(labels)):
        diff = y_hi[i] - y_lo[i]
        pen_l = 2/alpha*(y_lo[i]-labels[i]) * int(labels[i] < y_lo[i])
        pen_u = 2/alpha*(labels[i] - y_hi[i]) * int(labels[i] > y_hi[i])
        score.append(diff + pen_l + pen_u)
    score_total = np.mean(score)
    return score_total        

def _pin_loss(labels, pred, alpha):
    """
        Calculates Loss_PB_lo / Loss_PB_hi
    """
    loss = []
    for i in range(len(labels)):
        error = labels[i] - pred[i]
        if error <= 0:
            loss.append(-error*(1-alpha))
        else:
            loss.append(error*alpha)
    
    return np.mean(loss)

alphas = [0.05,0.95]

# choose path with model id to evaluate
path = 'predictions/during_shock/qd/tcn/1/'
# choose dataset that was used
dataset = "during_shock"

if dataset == "during_shock":
    arr_len = 165
elif dataset == "before_shock" or dataset == "after_shock":
    arr_len = 60
else:
    raise ValueError("Dataset must be 'during_shock', 'before_shock', or 'after_shock'")
    



csv_files = []
csv_files_conf = []
for f in os.listdir(f'{path}'):
    if os.path.isfile(os.path.join(f'{path}', f)):
        name,ext = os.path.splitext(os.path.join(f'{path}', f))
        if ext == ".csv":
            ending = name.split("_")[-1]
            if ending == "conf":
                csv_files_conf.append(f)
            elif len(ending) == 1:
                csv_files.append(f)

score = []
cwc = []
MPIW_total = []
NMPIW_total = []
PICP_total = []
PL_upper = []
PL_lower = []

PI_low = np.ndarray((arr_len, len(csv_files)))
PI_hi = np.ndarray((arr_len, len(csv_files)))

for idx,file in enumerate(csv_files):

    df = pd.read_csv(f'{path}{file}', sep=",")
    results = pd.DataFrame(df, columns=['low','high','actual']).to_numpy()
    dates = pd.DataFrame(df, columns=['date'])
    # check for %d.%m.%Y date and format if so
    if  len(dates["date"].iloc[0].split(".")) > 1: 
        dates = [datetime.strptime(date, '%d.%m.%Y').strftime('%Y-%m-%d') for date in dates["date"]]
    else:
        dates = dates["date"].iloc[-arr_len:]
    
    y_low = results[-arr_len:,0]
    y_high = results[-arr_len:,1]
    test_y = results[-arr_len:,2]

    PI_low[:,idx] = y_low
    PI_hi[:,idx] = y_high

    in_the_range = np.sum((test_y >= y_low) & (test_y <= y_high))
    PICP = in_the_range / np.prod(test_y.shape)
    MPIW = np.mean(abs(results[:,1] - results[:,0]))

    if dataset == 'old':
        df = pd.read_csv('./datasets/TTF_FM_old.csv', sep=";")
    else:
        df = pd.read_csv('./datasets/TTF_FM_new.csv', sep=";")
    prices = pd.DataFrame(df, columns=["Price"]).to_numpy()

    y_max = max(prices)
    y_min = min(prices)

    NMPIW = MPIW/(y_max-y_min)
    single_cwc = get_cwc(PICP, NMPIW)
    single_score = int_score(test_y,y_low, y_high)
    MPIW_total.append(MPIW)
    NMPIW_total.append(NMPIW)
    PICP_total.append(PICP)
    cwc.append(single_cwc)
    score.append(single_score)
    single_PL_lower = _pin_loss(test_y,y_low,0.05)
    single_PL_upper = _pin_loss(test_y,y_high,0.95)
    PL_lower.append(single_PL_lower)
    PL_upper.append(single_PL_upper)
    
print(f'MPIW : Mean {np.mean(MPIW_total)}, std:  {np.std(MPIW_total)}')
print(f'NMPIW : {np.mean(NMPIW_total)}, std:  {np.std(NMPIW_total)}')
print(f'PICP : {np.mean(PICP_total)}, std:  {np.std(PICP_total)}')
print(f'Interval Score: {np.mean(score)}, std:  {np.std(score)}')
print(f'CWC : {np.mean(cwc)}, std:  {np.std(cwc)}')
print(f'PL lower : {np.mean(PL_lower)}')
print(f'PL upper : {np.mean(PL_upper)}')

mean_df = pd.DataFrame({"PI_low" :np.mean(PI_low,axis=1), "PI_hi" : np.mean(PI_hi,axis=1), "date" : dates, "actual" : test_y})
mean_df.to_csv(path + "mean.csv")


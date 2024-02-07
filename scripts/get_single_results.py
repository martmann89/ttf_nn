import os
import pandas as pd
import numpy as np

base_path = './predictions_retrain/new_before_break/pb/lstm/'

csv_files = []
for f in os.listdir(f'{base_path}'):
    p = os.path.join(f'{base_path}',f)
    for ff in os.listdir(p):
        name,ext = os.path.splitext(os.path.join(f'{p}', ff))
        if name.split("/")[-1] == "stats" and ext == ".csv":
            csv_files.append(os.path.join(f'{p}', ff))
results_dic = []
for file in csv_files:
    index = file.split("/")[-2]
    print(file, index)
    df = pd.read_csv(f'{file}', sep=";")
    results = pd.DataFrame(df, columns=['coverage_sum','avg_length_sum','mis_sum']).to_numpy()
    results = np.squeeze(results)
    
    results_dic.append({"cov" : results[0], "len" : results[1], "mis" : results[2], "idx" : index})
for res in results_dic:
    print(res)
    print('\n')
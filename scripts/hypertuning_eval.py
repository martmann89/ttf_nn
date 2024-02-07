import os
import pandas as pd
import numpy as np

base_path = f'hypertuning_predictions/during_shock/qd/mlp/'
csv_files = []
for f in os.listdir(f'{base_path}'):
    if os.path.isfile(os.path.join(f'{base_path}', f)):
        name,ext = os.path.splitext(os.path.join(f'{base_path}', f))
        if ext == ".csv":
            csv_files.append(f)
results_dic = []
# print results id with coverage > min_cov
min_cov = 0.6
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
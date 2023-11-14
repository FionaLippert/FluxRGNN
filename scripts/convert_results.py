import torch
import numpy as np
import os.path as osp
import pickle

results = {
    'MLP (raw)': 'outputs/2023-11-08/10-32-49/results',
    'MLP (log)': 'outputs/2023-11-08/11-19-47/results',
    'LSTM (raw)': 'outputs/2023-11-07/15-12-20/results',
    'LSTM (log)': 'outputs/2023-11-07/11-04-01/results'
}

for k, v in results.items():
    fp = osp.join(v, 'results', 'test_results.pickle')
    with open(fp, 'rb') as f:
        res = pickle.load(f)

    for ki, vi in res.items():
        data = torch.cat(vi, dim=0)
        res[ki] = data.cpu().numpy()

    fp = osp.join(v, 'results', 'test_results_np.pickle')
    with open(fp, 'wb') as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL) 

        

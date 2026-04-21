import os
import numpy as np
import pandas as pd

def load_st_dataset(dataset):
    if dataset == '':
        data = pd.read_csv('', encoding='gbk', header=None).to_numpy()
    elif dataset == '':
        data_path = os.path.join(r'')
        data = np.load(data_path)['data']
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data

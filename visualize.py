import pickle
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

data_type = 'ecg'
filename = 'chfdb_chf01_275.pkl'

train_path = Path('dataset', data_type, 'labeled', 'train', filename)
test_path = Path('dataset', data_type, 'labeled', 'test', filename)

with open(str(train_path), 'rb') as f:
    data = pickle.load(f)
    plt.plot(data)
    plt.show()

with open(str(test_path), 'rb') as f:
    data = pickle.load(f)
    plt.plot(data)
    plt.show()

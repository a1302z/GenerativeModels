import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '../')

class MNIST(Dataset):
    
    def __init__(self, path, train=True):
        self.train_path = path + '/train/train-images-idx3-ubyte'
        self.test_path = path + '/test/t10k-images-idx3-ubyte'
        
    
    def __len__(self):
        return 0 
    
    def __getitem__(self, idx):
        return None



if __name__ == '__main__':
    m = MNIST('../data/MNIST')
    print(__name__)
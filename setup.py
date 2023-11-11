import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import copy
import random
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

from sklearn.model_selection import train_test_split
from pandas_datareader import data as pdr
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
yf.pdr_override()


# check if running on CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

dev_cnt = torch.cuda.device_count()
cur_dev = torch.cuda.current_device()
dev = torch.cuda.device(cur_dev)
nm = torch.cuda.get_device_name(cur_dev)
print(dev_cnt)
print(cur_dev)
print(dev)
print(nm)

torch.set_printoptions(precision=5, sci_mode = False)
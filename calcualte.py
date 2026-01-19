def warn(*args, **kwargs):
    pass
import sys
import warnings
warnings.warn = warn
from model import HCL
from data_loader import *
import argparse
import numpy as np
import torch
import random
import faiss
import sklearn.metrics as skm
import torch_geometric
import copy
#seed:1 3
ddps=[0.6677,0.6753]
avg_auc = np.mean(ddps)
std_auc = np.std(ddps)
print(ddps)
print('[FINAL RESULT] AVG_AUC:{:.4f}+-{:.4f}'.format(avg_auc, std_auc))
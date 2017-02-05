import numpy as np
import pandas as pd
from discrete import *

__author__ = 'numerology'

df = pd.read_csv('Info_Dataset.csv')

df['d1'] = df['d1'].astype('category').cat.codes
df['d2'] = df['d2'].astype('category').cat.codes
df['d3'] = df['d3'].astype('category').cat.codes
df['d4'] = df['d4'].astype('category').cat.codes
df['d5'] = df['d5'].astype('category').cat.codes
#preprocess

levels = [0,0,0,0,0]
levels[0] = len(list(df['d1'].unique())) 
levels[1] = len(list(df['d2'].unique())) 
levels[2] = len(list(df['d3'].unique())) 
levels[3] = len(list(df['d4'].unique())) 
levels[4] = len(list(df['d5'].unique())) 

# pairwise independence
dm = df.as_matrix().reshape((10000,5))
print(dm.shape[0],dm.shape[1])
pair_score = g_square_dis(dm, 5, 1, [],levels)
print(pair_score)
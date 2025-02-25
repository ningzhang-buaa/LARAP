import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def smooth(values, weight):
    smoothed = np.array(values)
    for i in range(1, smoothed.shape[0]):
        smoothed[i] = smoothed[i-1] * weight + (1 - weight) * smoothed[i]
    return smoothed
weight = 0.99
df = pd.read_csv('/home/e509/zn/maple/data/peg_ins/02-04-peg-ins2/02-04-peg_ins2_2024_02_04_09_00_09_0000--s-29293/progress.csv')
r_mean = df['eval/Returns Mean'].to_numpy()
r_mean = smooth(r_mean, weight)
r_std = df['eval/Rewards Std'].to_numpy()
print(r_mean.shape[0])
print(r_mean[0])
# eval/Returns Mean
# eval/Returns Std
# eval/Rewards Mean
# eval/Rewards Std
MAX_EPOCH = 234
r_mean = r_mean[:MAX_EPOCH]
r_std = r_std[:MAX_EPOCH]
x = np.array(range(MAX_EPOCH))
plt.plot(x,r_mean, label='maple')

df = pd.read_csv('/home/e509/zn/maple-llm/data/peg_ins/03-09-peg-ins-llm/03-09-peg_ins-llm_2024_03_09_21_09_20_0000--s-30647/progress.csv')
r_mean = df['eval/Returns Mean'].to_numpy()
r_mean = smooth(r_mean, weight)
plt.plot(x,r_mean, label='maple-llm')
plt.legend()
plt.show()
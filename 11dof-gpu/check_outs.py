import numpy as np
import matplotlib.pyplot as mpl
import pandas as pd


# Get all the files in a folder
import os, sys

path = "./outs/gpu/"
dirs = os.listdir(path)

fileS = []
step_dict = {}
for file in dirs:
	num = file.split('_')[-1].split('.')[0] # To find out how many sub csv files exist here
	file = '_'.join(file.split('_')[:-1]) # To get the base name of the csv file without the sub number
	step_dict[file] = int(num)
	fileS.append(file)


error_series = []
error_combined = []
for fileName in fileS:
    # ground truth
    data_c = pd.read_csv("./outs/test_set2_modCut.csv", sep=',', header='infer')
    
    # vehicle simulations on GPU
    subs = step_dict[fileName]
    li = []
    for sub in range(subs+1):
        data_ = pd.read_csv(path + fileName + '_' + str(sub) + '.csv', sep=',', header='infer')
        li.append(data_)

    data_p = pd.concat(li, axis = 0, ignore_index = True)

    data_p = data_p.loc[:data_c.shape[0]-1,:]

    data_c = data_c.iloc[:,:9]

    error_series.append(np.sum(np.abs(data_p - data_c)))
    error_combined.append(np.sum(np.sum(np.abs(data_p - data_c))))


print(error_combined)

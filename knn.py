import numpy as np
#import pandas as pd
#from collections import Counter

'''
# only use standard library
def Normalization(data_lst):
    data_norm = data_lst.copy()
    #find min and max data
    data_min = min(data_norm)
    data_max = max(data_norm)
    for i in range(len(data_lst)):
        data_norm[i] = (data_lst[i] - data_min) / (data_max - data_min)
    return data_norm
'''
def Normalization(data_lst):
    data_norm = np.zeros_like(data_lst,dtype=float)
    min_val = np.min(data_lst)
    max_val = np.max(data_lst)
    data_norm = (data_lst - min_val) / (max_val - min_val)
    return data_norm


def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1 - x2)**2))


temp_data = [32.2,30.1,32.0,31.5,33.8,31.5,31.8,33.2,31.9]
humid_data = [74.0,70.0,76.0,75.0,71.0,73.0,73.5,72.0,73.2]

temp_data_norm = Normalization(temp_data)
humid_data_norm = Normalization(humid_data)

print(temp_data_norm)
print(humid_data_norm)

temp_data_norm = Normalization(temp_data)
humid_data_norm = Normalization(humid_data)

print(temp_data_norm)
print(humid_data_norm)

data_array = np.array(list(zip(temp_data_norm,humid_data_norm)))
print(data_array)
test_sample = data_array[len(data_array)-1]
print(test_sample)
data_distance = np.zeros_like(temp_data)
for i in range(len(data_array)):
    print("data_array:",data_array[i])
    data_distance[i] = euclidean_distance(data_array[i],test_sample)

print(data_distance)
print(np.sort(data_distance))
print(np.__version__)
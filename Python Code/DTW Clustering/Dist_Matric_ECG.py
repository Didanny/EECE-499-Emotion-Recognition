import numpy as np
import pandas as pd
import scipy.io
import os
from scipy.signal import savgol_filter
from dtaidistance import dtw
import seaborn as sns
from sklearn import preprocessing
from numpy import inf
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from matplotlib import pyplot as plt
import scipy.spatial.distance as ssd
from tqdm import tqdm

#load_path = 'D:\EECE499\Features\\'
load_path = '..\..\..\\'

Features = pd.read_excel(load_path + 'Features.xlsx')

Ratings = pd.read_excel(load_path + 'Ratings.xlsx')

presentations_ids = Features['presentation_id'].values
presentations_ids.shape

indices_dict = []
series_ecg_1 = []
series_ecg_2 = []

for i, presentation_id in enumerate(presentations_ids):
    print(presentation_id, end='\r')
    
    indices_dict.append(presentation_id)
    
    clip_id = presentation_id % 100
    user_id = int(presentation_id / 100)
    
    data_path = './../../ASCERTAIN_Raw/ECGData/Movie_P' + str(user_id).zfill(2) + '/ECG_Clip' + str(clip_id) + '.mat'
    #data_path = 'D:/EECE499/Raw/MyECGFunc/ASCERTAIN_Raw/GSRData/Movie_P' + str(user_id).zfill(2) + '/GSR_Clip' + str(clip_id) + '.mat'
    
    if os.path.isfile(data_path):
        
        signal = scipy.io.loadmat(data_path)
        ecg1 = signal['Data_ECG'][:,4]
        ecg2 = signal['Data_ECG'][:,4] 
        ecg1 = savgol_filter(ecg1, 501, 3)
        ecg2 = savgol_filter(ecg2, 501, 3)
        
        series_ecg_1.append(ecg1)
        series_ecg_2.append(ecg2)
        
dm_ecg_1 = dtw.distance_matrix_fast(series_ecg_1, show_progress=True)
np.save('dm_ecg_1', dm_ecg_1)

dm_ecg_2 = dtw.distance_matrix_fast(series_ecg_2, show_progress=True)
np.save('dm_ecg_1', dm_ecg_2)
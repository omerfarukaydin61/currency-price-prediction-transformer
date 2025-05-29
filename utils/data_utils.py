import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_scale_data(file_path):
    custom_df = pd.read_csv(file_path)
    data = custom_df[['sellPrice']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

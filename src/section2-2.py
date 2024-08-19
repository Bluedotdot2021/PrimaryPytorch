import torch
import csv
import numpy as np

file_path = "bike_sharing_dataset/hour-fixed.csv"
bikes_numpy = np.loadtxt(file_path, dtype=np.float32, delimiter=",", skiprows=1, converters={1: lambda x: float(x[8:10])})
bikes_tensor = torch.from_numpy(bikes_numpy)
print(bikes_tensor.shape, bikes_tensor.stride())
daily_bikes_tensor = bikes_tensor.view(-1, 24, bikes_tensor.shape[1])
print(daily_bikes_tensor.shape, daily_bikes_tensor.stride())

daily_bikes_tensor.transpose_(1,2)
weather_onehot = torch.zeros(daily_bikes_tensor.shape[0],4,24)
daily_weather = daily_bikes_tensor[:,9,:] - 1
daily_weather.unsqueeze_(2)
print(daily_weather.shape)
weather_onehot.scatter_(1, daily_weather.long(), 1.0)
print(weather_onehot.shape)
daily_bikes_tensor = torch.cat((daily_bikes_tensor, weather_onehot), 1)
print(daily_bikes_tensor.shape)

#temp = daily_bikes_tensor[:,10,:]
#temp_max = torch.max(temp)
#temp_min = torch.min(temp)
#daily_bikes_tensor[:,10,:] = (temp - temp_min) / (temp_max - temp_min)

temp = daily_bikes_tensor[:,10,:]
daily_bikes_tensor[:,10,:] = (temp - torch.mean(temp))/torch.std(temp)
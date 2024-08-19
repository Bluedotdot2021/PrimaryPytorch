
import torch
import csv
import numpy as np

file_path = "winequality-white.csv"
data_numpy = np.loadtxt(file_path, dtype=np.float32, delimiter=";", skiprows=1)
col_list = next(csv.reader(open(file_path), delimiter=";"))

data_tensor = torch.from_numpy(data_numpy)
features_tensor = data_tensor[:,:-1]
print(features_tensor, features_tensor.shape)
target_tensor = data_tensor[:,-1].long()
print(target_tensor, target_tensor.shape)

target_onehot = torch.zeros(target_tensor.shape[0], 10)
target_onehot.scatter_(1, target_tensor.unsqueeze(1), 1.0)
print(target_onehot)

bad_data = features_tensor[torch.le(target_tensor,3)]
mid_data = features_tensor[target_tensor.gt(3) & target_tensor.lt(7)]
good_data = features_tensor[torch.ge(target_tensor, 7)]

bad_mean = bad_data.mean(0)
mid_mean = mid_data.mean(0)
good_mean = good_data.mean(0)

for i, args in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):
    print("{:2} {:20} {:6.2f} {:6.2f} {:6.2f}".format(i, *args))

total_sulfur_threshold = mid_mean[6]
total_sulfur_data = features_tensor[:,6]
predicted_indexes = total_sulfur_data.lt(total_sulfur_threshold)
print(predicted_indexes.shape, predicted_indexes.sum())
actual_indexes = target_tensor.gt(5)
print(actual_indexes.shape, actual_indexes.sum())

correct_matches = torch.sum(predicted_indexes & actual_indexes).item()
total_predicted = predicted_indexes.sum().item()
total_actual = actual_indexes.sum().item()
precision = correct_matches / total_actual
recall = correct_matches / total_predicted
print(correct_matches, precision, recall)

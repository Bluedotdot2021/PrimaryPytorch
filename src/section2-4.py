
import torch
import os
import imageio.v3 as iio

img_arr = iio.imread("bobby.jpg")
print(type(img_arr), img_arr.shape)

img_tensor = torch.from_numpy(img_arr)
img_tensor.transpose_(0, 2).transpose_(1,2)
print(img_tensor.shape)

filenames = [name for name in os.listdir("image_cats") if os.path.splitext(name)[1] == '.png']
batch_tensor = torch.zeros(len(filenames), 3, 256, 256, dtype=torch.uint8)
for i, filename in enumerate(filenames):
    img_arr = iio.imread(os.path.join("image_cats", filename))
    batch_tensor[i] = torch.from_numpy(img_arr).transpose_(0,2)
print(batch_tensor.shape)

batch_tensor = batch_tensor.float()
batch_tensor /= 255.0
for c in range(batch_tensor.shape[1]): #channels
    mean = torch.mean(batch_tensor[:,c])
    std = torch.std(batch_tensor[:,c])
    batch_tensor[:,c] = (batch_tensor[:,c] - mean)/std
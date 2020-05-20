# -*- coding: utf-8 -*-
"""
Created on Thu May 21 01:41:21 2020

@author: 王式珩
"""

"""hw7_Network_Compression (Weight Quantization)"""
# use small model, which already performed Knowledge Distillation, for Quantization

"""Import Packages"""
import os
import sys
import torch
import pickle
import torch.nn as nn
import numpy as np
from PIL import Image
from glob import glob
import torchvision.models as models
import torchvision.transforms as transforms
from hw7_architecture_design import StudentNet # model

"""Path Specification"""
path_prefix        = './'
# in_model_filename  = os.path.join(path_prefix, 'model/student_model.bin')
# model_16b_filename = os.path.join(path_prefix, 'model/16_bit_model.bin')
# model_8b_filename  = os.path.join(path_prefix, 'model/model_8_bit.bin')
in_model_filename  = 'hw7_kaggle_predict_model.bin'
model_8b_filename  = 'hw7_kaggle_predict_model_8_bit.bin'

# region: 32-bit Tensor -> 16-bit
def encode16(params, fname):
    """
    Args:
      	params: model's state_dict
      	fname:  filename after compression
    """
    custom_dict = {}
    for (name, param) in params.items():
        param = np.float64(param.cpu().numpy())
        # some param. aren't ndarray, no compression needed
        if type(param) == np.ndarray:
            custom_dict[name] = np.float16(param)
        else:
            custom_dict[name] = param

    pickle.dump(custom_dict, open(fname, 'wb'))

def decode16(fname):
    """
    Args:
      	fname: filename after compression
    """
    params = pickle.load(open(fname, 'rb'))
    custom_dict = {}
    for (name, param) in params.items():
        param = torch.tensor(param)
        custom_dict[name] = param

    return custom_dict
#endregion

# region: 32-bit Tensor -> 8-bit (OPTIONAL)
# W' = round((W - min(W)) / (max(W) - min(W)) * (2^8 - 1))
def encode8(params, fname):
    custom_dict = {}
    for (name, param) in params.items():
        param = np.float64(param.cpu().numpy())
        if type(param) == np.ndarray:
            min_val = np.min(param)
            max_val = np.max(param)
            param = np.round((param - min_val) / (max_val - min_val) * 255)
            param = np.uint8(param)
            custom_dict[name] = (min_val, max_val, param)
        else:
            custom_dict[name] = param

    pickle.dump(custom_dict, open(fname, 'wb'))

def decode8(fname):
    params = pickle.load(open(fname, 'rb'))
    custom_dict = {}
    for (name, param) in params.items():
        if type(param) == tuple:
            min_val, max_val, param = param
            param = np.float64(param)
            param = (param / 255 * (max_val - min_val)) + min_val
            param = torch.tensor(param)
        else:
            param = torch.tensor(param)

        custom_dict[name] = param

    return custom_dict
# endregion


"""Main"""
print('=================== Weight Quantization ===================')
print(f'original cost: {os.stat(in_model_filename).st_size} bytes.')
params = torch.load(in_model_filename)

# 32b -> 16b
# encode
# encode16(params, model_16b_filename)
# print(f'16-bit cost: {os.stat(model_16b_filename).st_size} bytes')

# 32b -> 8b
# encode
encode8(params, model_8b_filename)
print(f'8-bit cost: {os.stat(model_8b_filename).st_size} bytes')

# -*- coding: utf-8 -*-
"""
Created on Thu May 21 01:02:42 2020

@author: 王式珩
"""
"""hw7_Network_Compression (Knowledge Distillation)"""

"""Import Packages"""
import numpy as np
import torch
import os
import re
import sys
import pickle
from glob import glob
from PIL import Image
import torchvision.transforms as transforms
from hw7_architecture_design import StudentNet 


dir_name = './food-11'
output = 'hw7_kaggle_predict_model_8_bit.csv'

model_filename = 'hw7_kaggle_predict_model_8_bit.bin'

#region: Data Processing
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, folderName, transform=None):
        self.transform = transform
        self.data = []
        self.label = []

        for img_path in sorted(glob(folderName + '/*.jpg')):
            try:
                # Get classIdx by parsing image path
                class_idx = int(re.findall(re.compile(r'\d+'), img_path)[1])
            except:
                # if inference mode (there's no answer), class_idx default 0
                class_idx = 0
 
            image = Image.open(img_path)
            # Get File Descriptor
            image_fp = image.fp
            image.load()
            # Close File Descriptor (or it'll reach OPEN_MAX)
            image_fp.close()

            self.data.append(image)
            self.label.append(class_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.label[idx]


trainTransform = transforms.Compose([
    transforms.RandomCrop(256, pad_if_needed=True, padding_mode='symmetric'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])
testTransform = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

def get_dataloader(mode='training', batch_size=32):

    assert mode in ['training', 'testing', 'validation']

    dataset = MyDataset(
        os.path.join(dir_name, mode),
        transform=trainTransform if mode == 'training' else testTransform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'training'))

    return dataloader
#endregion

# region: WQ decode
def decode16(fname):
    """
    Args:
      	fname: filename after compression
    """
    params = pickle.load(open(fname, 'rb'))
    custom_dict = {}
    for (name, param) in params.items():
        param = torch.tensor(param)
        custom_dict[name] = param

    return custom_dict

def decode8(fname):
    params = pickle.load(open(fname, 'rb'))
    custom_dict = {}
    for (name, param) in params.items():
        if type(param) == tuple:
            min_val, max_val, param = param
            param = np.float64(param)
            param = (param / 255 * (max_val - min_val)) + min_val
            param = torch.tensor(param)
        else:
            param = torch.tensor(param)

        custom_dict[name] = param

    return custom_dict

test_dataloader = get_dataloader('testing', batch_size=32)


student_net = StudentNet(base=16).cuda()
student_net.load_state_dict(decode8(model_filename))

student_net.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        test_pred = student_net(data[0].cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)


with open(output, 'w') as f:
    f.write('id,label\n')
    for i, y in  enumerate(prediction):
        f.write('{},{}\n'.format(i, y))
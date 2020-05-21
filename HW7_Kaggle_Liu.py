# -*- coding: utf-8 -*-
"""
Created on Wed May 20 20:17:22 2020

@author: 王式珩
"""

"""hw7_Network_Compression (Knowledge Distillation)"""

"""Import Packages"""
import numpy as np
import torch
import os
import sys
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torchvision.models as models
import re
from glob import glob
from PIL import Image
import torchvision.transforms as transforms
from hw7_Architecture_Design import StudentNet # model

"""Path Specification"""
path_prefix            = './'
data_dir               = os.path.join(path_prefix, 'food-11')
# teacher_model_filename = os.path.join(path_prefix, 'model/teacher_resnet18.bin')
# student_model_filename = os.path.join(path_prefix, 'model/student_model.bin')
#data_dir               = sys.argv[1]
teacher_model_filename = './teacher_resnet18.bin'
student_model_filename = './hw7_kaggle_predict_model_momentum_0.75.bin'

# region: Knowledge Distillation Loss
# Loss = alpha * T^2 * KL(Teacher's Logits / T || Student's Logits / T) + (1-alpha)(original Loss)
def loss_fn_kd(outputs, labels, teacher_outputs, T=20, alpha=0.5):
    # ordinary Cross Entropy
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha)
    # KL Divergence
    soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)
    return hard_loss + soft_loss
#endregion

# region: Data Processing
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, folderName, transform=None):
        self.transform = transform
        self.data = []
        self.label = []

        for img_path in glob(folderName + '/*.jpg'):
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
        os.path.join(data_dir, mode),
        transform=trainTransform if mode == 'training' else testTransform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'training'))

    return dataloader
#endregion

# region: Training
def run_epoch(dataloader, update=True, alpha=0.5):
    total_num, total_hit, total_loss = 0, 0, 0
    for now_step, batch_data in enumerate(dataloader):
        # initialize optimizer
        optimizer.zero_grad()
        inputs, hard_labels = batch_data
        inputs = inputs.cuda()
        hard_labels = torch.LongTensor(hard_labels).cuda()
        # TeacherNet no need to backprop -> torch.no_grad
        with torch.no_grad():
            soft_labels = teacher_net(inputs)

        if update:
            logits = student_net(inputs)
            # use the loss the combines soft label & hard label
            loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)
            loss.backward()
            optimizer.step()    
        else:
            with torch.no_grad():
                logits = student_net(inputs)
                loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)
            
        total_hit += torch.sum(torch.argmax(logits, dim=1) == hard_labels).item()
        total_num += len(inputs)

        total_loss += loss.item() * len(inputs)
    return total_loss / total_num, total_hit / total_num
#endregion

"""Main"""
print('=================== Knowledge Distillation ===================')
# training
# pre-processing
print('Pre-processing training data')
#train_dataloader = get_dataloader('training', batch_size=32)
#valid_dataloader = get_dataloader('validation', batch_size=32)
valid_dataloader = torch.load('valid_dataloader')
train_dataloader = torch.load('train_dataloader')
print('Loading models')
teacher_net = models.resnet18(pretrained=False, num_classes=11).cuda()
teacher_net.load_state_dict(torch.load(teacher_model_filename))
print(f'original cost: {os.stat(teacher_model_filename).st_size} bytes.')

student_net = StudentNet(base=16).cuda()
total = sum(p.numel() for p in student_net.parameters())
trainable = sum(p.numel() for p in student_net.parameters() if p.requires_grad)
print('\nparameter total:{}, trainable:{}\n'.format(total, trainable))

print("SGD optimizer")
optimizer = optim.SGD(student_net.parameters(), lr=1e-2, momentum=0.85)
scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
# print("Adam optimizer")
# optimizer = optim.Adam(student_net.parameters(), lr=1e-3)

print('Training')
teacher_net.eval() # TeacherNet is always Eval mode
now_best_acc = 0
for epoch in range(250):
    student_net.train()
    train_loss, train_acc = run_epoch(train_dataloader, update=True)
    student_net.eval()
    valid_loss, valid_acc = run_epoch(valid_dataloader, update=False)

    # save best model
    if valid_acc > now_best_acc:
        now_best_acc = valid_acc
        torch.save(student_net.state_dict(), student_model_filename)
    print('epoch {:>3d}: train loss: {:6.4f}, acc {:6.4f} valid loss: {:6.4f}, acc {:6.4f}'.format(
        epoch, train_loss, train_acc, valid_loss, valid_acc))

print(f'model size: {os.stat(student_model_filename).st_size} bytes')

# ----------------------------------------------------------------------------------

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


"""Path Specification"""
path_prefix        = './'
# in_model_filename  = os.path.join(path_prefix, 'model/student_model.bin')
# model_16b_filename = os.path.join(path_prefix, 'model/16_bit_model.bin')
# model_8b_filename  = os.path.join(path_prefix, 'model/model_8_bit.bin')
in_model_filename  = student_model_filename
model_8b_filename  = './hw7_kaggle_predict_model_momentum_0.75_8_bit.bin'

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

#--------------------------------------------------------------------------------------
#test_dataloader = get_dataloader('testing', batch_size=32)
test_dataloader = torch.load('testing_dataloader.pth')
print('Loading student model')
student_net = StudentNet(base=16).cuda()
student_net.load_state_dict(decode8(model_8b_filename))

print('Predicting')
student_net.eval()
prediction = []
predict_filename = 'hw7_kaggle_predict_SGD_momentum_0.75_liu.csv'
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        test_pred = student_net(data[0].cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)

print('Writing to csv file')
with open(predict_filename, 'w') as f:
    f.write('id,label\n')
    for i, y in  enumerate(prediction):
        f.write('{},{}\n'.format(i, y))

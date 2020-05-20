# -*- coding: utf-8 -*-
"""
Created on Wed May 20 14:44:33 2020

@author: 王式珩
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 19 16:13:51 2020

@author: 王式珩
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 19 00:38:08 2020

@author: 王式珩
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 18 21:07:34 2020

@author: 王式珩
"""

# -*- coding: utf-8 -*-
"""Copy of hw7_Knowledge_Distillation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-dOolz-OQ-_WYqBDrSKkUQJSSIAVU-KE

# Homework 7 - Network Compression (Knowledge Distillation)

> Author: Arvin Liu (b05902127@ntu.edu.tw)

若有任何問題，歡迎來信至助教信箱 ntu-ml-2020spring-ta@googlegroups.com
"""



"""# Readme


HW7的任務是模型壓縮 - Neural Network Compression。

Compression有很多種門派，在這裡我們會介紹上課出現過的其中四種，分別是:

* 知識蒸餾 Knowledge Distillation
* 網路剪枝 Network Pruning
* 用少量參數來做CNN Architecture Design
* 參數量化 Weight Quantization

在這個notebook中我們會介紹Knowledge Distillation，
而我們有提供已經學習好的大model方便大家做Knowledge Distillation。
而我們使用的小model是"Architecture Design"過的model。

* Architecute Design在同目錄中的hw7_Architecture_Design.ipynb。
* 下載pretrained大model(47.2M): https://drive.google.com/file/d/1B8ljdrxYXJsZv2vmTequdPOofp3VF3NN/view?usp=sharing
  * 請使用torchvision提供的ResNet18，把num_classes改成11後load進去即可。(後面有範例。)
"""

# Commented out IPython magic to ensure Python compatibility.
import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from hw7_Architecture_Design import *
# Load進我們的Model架構(在hw7_Architecture_Design.ipynb內)
#gdown --id '1lJS0ApIyi7eZ2b3GMyGxjPShI8jXM2UC' --output "hw7_Architecture_Design.ipynb"
# %run "hw7_Architecture_Design.ipynb"

"""Knowledge Distillation
===

<img src="https://i.imgur.com/H2aF7Rv.png=100x" width="500px">

簡單上來說就是讓已經做得很好的大model們去告訴小model"如何"學習。
而我們如何做到這件事情呢? 就是利用大model預測的logits給小model當作標準就可以了。

## 為甚麼這會work?
* 例如當data不是很乾淨的時候，對一般的model來說他是個noise，只會干擾學習。透過去學習其他大model預測的logits會比較好。
* label和label之間可能有關連，這可以引導小model去學習。例如數字8可能就和6,9,0有關係。
* 弱化已經學習不錯的target(?)，避免讓其gradient干擾其他還沒學好的task。


## 要怎麼實作?
* $Loss = \alpha T^2 \times KL(\frac{\text{Teacher's Logits}}{T} || \frac{\text{Student's Logits}}{T}) + (1-\alpha)(\text{原本的Loss})$


* 以下code為甚麼要對student使用log_softmax: https://github.com/peterliht/knowledge-distillation-pytorch/issues/2
* reference: [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
"""

def loss_fn_kd(outputs, labels, teacher_outputs, T=20, alpha=0.5):
    # 一般的Cross Entropy
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha)
    # 讓logits的log_softmax對目標機率(teacher的logits/T後softmax)做KL Divergence。
    soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)
    return hard_loss + soft_loss

"""# Data Processing

我們的Dataset使用的是跟Hw3 - CNN同樣的Dataset，因此這個區塊的Augmentation / Read Image大家參考或直接抄就好。

如果有不會的話可以回去看Hw3的colab。

需要注意的是如果要自己寫的話，Augment的方法最好使用我們的方法，避免輸入有差異導致Teacher Net預測不好。
"""

import re
import torch
from glob import glob
from PIL import Image
import torchvision.transforms as transforms
#from summary_writer import *
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
        f'./food-11/{mode}',
        transform=trainTransform if mode == 'training' else testTransform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'training'))

    return dataloader

"""# Pre-processing

我們已經提供TeacherNet的state_dict，其架構是torchvision提供的ResNet18。

至於StudentNet的架構則在hw7_Architecture_Design.ipynb中。

這裡我們使用的Optimizer為AdamW，沒有為甚麼，就純粹我想用。
"""

# get dataloader
train_dataloader = torch.load('train_dataloader')
valid_dataloader = torch.load('valid_dataloader')

#!gdown --id '1B8ljdrxYXJsZv2vmTequdPOofp3VF3NN' --output teacher_resnet18.bin

teacher_net = models.resnet18(pretrained=False, num_classes=11).cuda()
student_net = StudentNet(base=16).cuda()

teacher_net.load_state_dict(torch.load(f'./teacher_resnet18.bin'))
optimizer = optim.SGD(student_net.parameters(), lr=0.02, momentum=0.01)
# optimizer = optim.AdamW(student_net.parameters(), lr=1e-2)

"""# Start Training

* 剩下的步驟與你在做Hw3 - CNN的時候一樣。

## 小提醒

* torch.no_grad是指接下來的運算或該tensor不需要算gradient。
* model.eval()與model.train()差在於Batchnorm要不要紀錄，以及要不要做Dropout。
"""

def run_epoch(dataloader, update=True, alpha=0.5):
    total_num, total_hit, total_loss = 0, 0, 0
    for now_step, batch_data in enumerate(dataloader):
        # 清空 optimizer
        optimizer.zero_grad()
        # 處理 input
        inputs, hard_labels = batch_data
        inputs = inputs.cuda()
        hard_labels = torch.LongTensor(hard_labels).cuda()
        # 因為Teacher沒有要backprop，所以我們使用torch.no_grad
        # 告訴torch不要暫存中間值(去做backprop)以浪費記憶體空間。
        with torch.no_grad():
            soft_labels = teacher_net(inputs)

        if update:
            logits = student_net(inputs)
            # 使用我們之前所寫的融合soft label&hard label的loss。
            # T=20是原始論文的參數設定。
            loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)
            loss.backward()
            optimizer.step()    
        else:
            # 只是算validation acc的話，就開no_grad節省空間。
            with torch.no_grad():
                logits = student_net(inputs)
                loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)
            
        total_hit += torch.sum(torch.argmax(logits, dim=1) == hard_labels).item()
        total_num += len(inputs)

        total_loss += loss.item() * len(inputs)
    return total_loss / total_num, total_hit / total_num


# TeacherNet永遠都是Eval mode.
teacher_net.eval()
now_best_acc = 0
for epoch in range(300):
    if epoch % 50 == 0:
      for g in optimizer.param_groups:
        g['lr'] = g['lr']/2
    student_net.train()
    train_loss, train_acc = run_epoch(train_dataloader, update=True)
    student_net.eval()
    valid_loss, valid_acc = run_epoch(valid_dataloader, update=False)

    # 存下最好的model。
    
    if valid_acc > now_best_acc:
        now_best_acc = valid_acc
        torch.save(student_net.state_dict(), 'student_model_kaggle_SGD_300.bin')
    print('epoch {:>3d}: train loss: {:6.4f}, acc {:6.4f} valid loss: {:6.4f}, acc {:6.4f}'.format(
        epoch, train_loss, train_acc, valid_loss, valid_acc))
#summary(model, (3, 128, 128), 'student_model_from_teacher_fine.txt')
"""# Inference

同Hw3，請參考該作業:)。

# Q&A

有任何問題Network Compression的問題可以寄信到b05902127@ntu.edu.tw / ntu-ml-2020spring-ta@googlegroups.com。

時間允許的話我會更新在這裡。
"""
import os
import torch

print(f"\noriginal cost: {os.stat('student_model_kaggle_SGD_300.bin').st_size} bytes.")
params = torch.load('student_model_kaggle_SGD_300.bin')

"""# 32-bit Tensor -> 16-bit"""

import numpy as np
import pickle

def encode16(params, fname):
    '''將params壓縮成16-bit後輸出到fname。
    Args:
      params: model的state_dict。
      fname: 壓縮後輸出的檔名。
    '''

    custom_dict = {}
    for (name, param) in params.items():
        param = np.float64(param.cpu().numpy())
        # 有些東西不屬於ndarray，只是一個數字，這個時候我們就不用壓縮。
        if type(param) == np.ndarray:
            custom_dict[name] = np.float16(param)
        else:
            custom_dict[name] = param

    pickle.dump(custom_dict, open(fname, 'wb'))


def decode16(fname):
    '''從fname讀取各個params，將其從16-bit還原回torch.tensor後存進state_dict內。
    Args:
      fname: 壓縮後的檔名。
    '''

    params = pickle.load(open(fname, 'rb'))
    custom_dict = {}
    for (name, param) in params.items():
        param = torch.tensor(param)
        custom_dict[name] = param

    return custom_dict


encode16(params, '16_bit_model_SGD_300.pkl')
print(f"16-bit cost: {os.stat('16_bit_model_SGD_300.pkl').st_size} bytes.")

"""# 32-bit Tensor -> 8-bit (OPTIONAL)
這邊提供轉成8-bit的方法，僅供大家參考。
因為沒有8-bit的float，所以我們先對每個weight記錄最小值和最大值，進行min-max正規化後乘上$2^8-1$在四捨五入，就可以用np.uint8存取了。
$W' = round(\frac{W - \min(W)}{\max(W) - \min(W)} \times (2^8 - 1)$)
> 至於能不能轉成更低的形式，例如4-bit呢? 當然可以，待你實作。
"""

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

encode8(params, '8_bit_model_SGD_300.pkl')
print(f"8-bit cost: {os.stat('8_bit_model_SGD_300.pkl').st_size} bytes.")


import torch
from glob import glob

import torchvision.transforms as transforms
#my_mean = np.load('mean.npy')
#my_std = np.load('std.npy')
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



#testing_dataloader = get_dataloader('testing', batch_size=32)
#torch.save(testing_dataloader, 'testing_dataloader.pth')
testing_dataloader = torch.load('testing_dataloader.pth')
# workspace_dir = str(sys.argv[1])
# workspace_dir = './food-11/'
MODLE_PATH = '8_bit_model_SGD_300.pkl'
# #test_x = np.load('test_x.npy')

# print("Loading mean")
# my_mean = np.load("mean.npy")
# print("mean loaded")
# print("Loading std")
# my_std = np.load("std.npy")
# print("std loaded")

#training 時做 data augmentation


"""# Model"""


import torch






# Train

# with open(MODLE_PATH, 'rb') as f:
#     obj = f.read()
# weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
# model_best.load_state_dict(weights)
# infile = open(MODLE_PATH,'rb')
# model_best = pickle.load(infile)
# print(type(model_best))

model_best = StudentNet(base=16).cuda()
model_best.load_state_dict(decode8(MODLE_PATH))
print("Model(" + MODLE_PATH+ ") loaded")

"""# Testing
利用剛剛 train 好的 model 進行 prediction
"""

#test_set = ImgDataset(test_x, transform=test_transform)
#test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
# test_loader = torch.load('test_loader.pth')

model_best.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(testing_dataloader):
        #print('i is ', i)
        #print('data is ', data)
        test_pred = model_best(data[0].cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)
name = 'Hw7_output_SGD_300.csv' 
#將結果寫入 csv 檔
with open(name, 'w') as f:
    f.write('Id,Category\n')
    for i, y in  enumerate(prediction):
        f.write('{},{}\n'.format(i, y))
print("Prediction Done")
print(name + " saved")

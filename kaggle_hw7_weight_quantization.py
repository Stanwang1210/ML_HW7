# -*- coding: utf-8 -*-
"""Kaggle_hw7_Weight_Quantization.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PyPPhrL2bPMWjmcuzRaM8Xfd-1zRpYhq

# Homework 7 - Network Compression (Weight Quantization)

> Author: Arvin Liu (b05902127@ntu.edu.tw)

若有任何問題，歡迎來信至助教信箱 ntu-ml-2020spring-ta@googlegroups.com
"""



from google.colab import drive

drive.mount('/content/drive')

ls

cd 'drive'

ls

cd 'My Drive'

ls

cd 'HW7 Data'

ls

"""# Readme


HW7的任務是模型壓縮 - Neural Network Compression。

Compression有很多種門派，在這裡我們會介紹上課出現過的其中四種，分別是:

* 知識蒸餾 Knowledge Distillation
* 網路剪枝 Network Pruning
* 用少量參數來做CNN Architecture Design
* 參數量化 Weight Quantization

在這個notebook中我們會介紹非常簡單的Weight Quantization，
而我們有提供已經做完Knowledge Distillation的小model來做Quantization。

* Model架構 / Architecute Design在同目錄中的hw7_Architecture_Design.ipynb。
* 下載已經train好的小model(0.99M): https://drive.google.com/open?id=12wtIa0WVRcpboQzhgRUJOpcXe23tgWUL
  * 參數為 base=16, width_mult=1 (default)


## Weight Quantization
<img src="https://i.imgur.com/SMsaiAo.png" width="500px">

我們這邊會示範如何實作第一條: Using less bits to represent a value。

## 好的Quantization很重要。
這邊提供一些TA的數據供各位參考。

|bit|state_dict size|accuracy|
|-|-|-|
|32|1047430 Bytes|0.81315|
|16|522958 Bytes|0.81347|
|8|268472 Bytes|0.80791|
|7|268472 Bytes|0.80791|


## Byte Cost
根據[torch的官方手冊](https://pytorch.org/docs/stable/tensors.html)，我們知道torch.FloatTensor預設是32-bit，也就是佔了4byte的空間，而FloatTensor系列最低可以容忍的是16-bit。

為了方便操作，我們之後會將state_dict轉成numpy array做事。
因此我們可以先看看numpy有甚麼樣的type可以使用。
![](https://i.imgur.com/3N7tiEc.png)
而我們發現numpy最低有float16可以使用，因此我們可以直接靠轉型將32-bit的tensor轉換成16-bit的ndarray存起來。

# Read state_dict

下載我們已經train好的小model的state_dict進行測試。
"""

import os
import torch

print(f"\noriginal cost: {os.stat('student_model_from_teacher_fine.bin').st_size} bytes.")
params = torch.load('student_model_from_teacher_fine.bin')

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


encode16(params, '16_bit_model.pkl')
print(f"16-bit cost: {os.stat('16_bit_model.pkl').st_size} bytes.")

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

encode8(params, '8_bit_model.pkl')
decode8('8_bit_model.pkl')
print(f"8-bit cost: {os.stat('8_bit_model.pkl').st_size} bytes.")

"""# Q&A

有任何問題Network Compression的問題可以寄信到b05902127@ntu.edu.tw。

時間允許的話我會更新在這裡。
"""
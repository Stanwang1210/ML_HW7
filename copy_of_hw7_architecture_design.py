# -*- coding: utf-8 -*-
"""Copy of hw7_Architecture_Design.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1o4USoVcW4Ce5uM65oQ-0Y--r7KUDU-iT

# Homework 7 - Network Compression (Architecuture Design)

> Author: Arvin Liu (b05902127@ntu.edu.tw)

若有任何問題，歡迎來信至助教信箱 ntu-ml-2020spring-ta@googlegroups.com

# Readme

HW7的任務是模型壓縮 - Neural Network Compression。

Compression有很多種門派，在這裡我們會介紹上課出現過的其中四種，分別是:

* 知識蒸餾 Knowledge Distillation
* 網路剪枝 Network Pruning
* 用少量參數來做CNN Architecture Design
* 參數量化 Weight Quantization

在這個notebook中我們會介紹MobileNet v1的Architecture Design。

# Architecture Design

## Depthwise & Pointwise Convolution
![](https://i.imgur.com/FBgcA0s.png)
> 藍色為上下層Channel的關係，綠色則為該Receptive Field的擴張。
> (圖片引用自arxiv:1810.04231)

(a) 就是一般的Convolution Layer，所以他的Weight連接方式會跟Fully Connected一樣，只差在原本在FC是用數字相乘後相加，Convolution Layer是圖片卷積後相加。

(b) DW(Depthwise Convolution Layer)你可以想像成一張feature map各自過**一個filter**處理後，再用PW(Pointwise Convolution Layer)把所有feature map的單個pixel資訊合在一起(就是1個pixel的Fully Connected Layer)。

(c) GC(Group Convolution Layer)就是把feature map分組，讓他們自己過Convolution Layer後再重新Concat起來。算是一般的Convolution和Depthwise Convolution的折衷版。**所以說，Group Convolution的Group=Input Feautures數就會是Depthwise Convolution(因為每個Channel都各自獨立)，Group=1就會是一般的Convolution(因為就等於沒有Group)。**

<img src="https://i.imgur.com/Hqhg0Q9.png" width="500px">


## 實作細節
```python
# 一般的Convolution, weight大小 = in_chs * out_chs * kernel_size^2
nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding)

# Group Convolution, Group數目可以自行控制，表示要分成幾群。其中in_chs和out_chs必須要可以被groups整除。(不然沒辦法分群。)
nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, groups=groups)

# Depthwise Convolution, 輸入chs=輸出chs=Groups數目, weight大小 = in_chs * kernel_size^2
nn.Conv2d(in_chs, out_chs=in_chs, kernel_size, stride, padding, groups=in_chs)

# Pointwise Convolution, 也就是1 by 1 convolution, weight大小 = in_chs * out_chs
nn.Conv2d(in_chs, out_chs, 1)
```

# Model

* training的部分請參考Network Pruning、Knowledge Distillation，或直接只用Hw3的手把手即可。

> 註記: 這邊把各個Block多用一層Sequential包起來是因為Network Pruning的時候抓Layer比較方便。
"""

import torch.nn as nn
import torch.nn.functional as F
import torch

class StudentNet(nn.Module):
    '''
      在這個Net裡面，我們會使用Depthwise & Pointwise Convolution Layer來疊model。
      你會發現，將原本的Convolution Layer換成Dw & Pw後，Accuracy通常不會降很多。

      另外，取名為StudentNet是因為這個Model等會要做Knowledge Distillation。
    '''

    def __init__(self, base=16, width_mult=1):
        '''
          Args:
            base: 這個model一開始的ch數量，每過一層都會*2，直到base*16為止。
            width_mult: 為了之後的Network Pruning使用，在base*8 chs的Layer上會 * width_mult代表剪枝後的ch數量。        
        '''
        super(StudentNet, self).__init__()
        multiplier = [1, 2, 4, 8, 16, 16, 16, 16]

        # bandwidth: 每一層Layer所使用的ch數量
        bandwidth = [ base * m for m in multiplier]

        # 我們只Pruning第三層以後的Layer
        for i in range(3, 7):
            bandwidth[i] = int(bandwidth[i] * width_mult)

        self.cnn = nn.Sequential(
            # 第一層我們通常不會拆解Convolution Layer。
            nn.Sequential(
                nn.Conv2d(3, bandwidth[0], 3, 1, 1),
                nn.BatchNorm2d(bandwidth[0]),
                nn.ReLU6(),
                nn.MaxPool2d(2, 2, 0),
                
            ),
            # 接下來每一個Sequential Block都一樣，所以我們只講一個Block
            nn.Sequential(
                # Depthwise Convolution
                nn.Conv2d(bandwidth[0], bandwidth[0], 3, 1, 1, groups=bandwidth[0]),
                # Batch Normalization
                nn.BatchNorm2d(bandwidth[0]),
                # ReLU6 是限制Neuron最小只會到0，最大只會到6。 MobileNet系列都是使用ReLU6。
                # 使用ReLU6的原因是因為如果數字太大，會不好壓到float16 / or further qunatization，因此才給個限制。
                nn.ReLU6(),
                # Pointwise Convolution
                nn.Conv2d(bandwidth[0], bandwidth[1], 1),
                # 過完Pointwise Convolution不需要再做ReLU，經驗上Pointwise + ReLU效果都會變差。
                nn.MaxPool2d(2, 2, 0),
                
                # 每過完一個Block就Down Sampling
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[1], bandwidth[1], 3, 1, 1, groups=bandwidth[1]),
                nn.BatchNorm2d(bandwidth[1]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[1], bandwidth[2], 1),
                nn.MaxPool2d(2, 2, 0),
                
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[2], bandwidth[2], 3, 1, 1, groups=bandwidth[2]),
                nn.BatchNorm2d(bandwidth[2]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[2], bandwidth[3], 1),
                nn.MaxPool2d(2, 2, 0),
                
            ),

            # 到這邊為止因為圖片已經被Down Sample很多次了，所以就不做MaxPool
            nn.Sequential(
                nn.Conv2d(bandwidth[3], bandwidth[3], 3, 1, 1, groups=bandwidth[3]),
                nn.BatchNorm2d(bandwidth[3]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[3], bandwidth[4], 1),
                
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[4], bandwidth[4], 3, 1, 1, groups=bandwidth[4]),
                nn.BatchNorm2d(bandwidth[4]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[4], bandwidth[5], 1),
                
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[5], bandwidth[5], 3, 1, 1, groups=bandwidth[5]),
                nn.BatchNorm2d(bandwidth[5]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[5], bandwidth[6], 1),
                
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[6], bandwidth[6], 3, 1, 1, groups=bandwidth[6]),
                nn.BatchNorm2d(bandwidth[6]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[6], bandwidth[7], 1),
                
            ),

            # 這邊我們採用Global Average Pooling。
            # 如果輸入圖片大小不一樣的話，就會因為Global Average Pooling壓成一樣的形狀，這樣子接下來做FC就不會對不起來。
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            # 這邊我們直接Project到11維輸出答案。
            nn.Linear(bandwidth[7], 11),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

"""# Q&A

有任何問題Network Compression的問題可以寄信到b05902127@ntu.edu.tw。

我有空的話會更新在這裡。
"""

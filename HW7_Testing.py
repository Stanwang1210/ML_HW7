{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "HW7_Testing.ipynb",
      "provenance": [],
      "authorship_tag": "ABX9TyMqAWAx55FVgTa8UV1WGLDm",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Stanwang1210/ML_HW7/blob/master/HW7_Testing.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "KNkd7xYZIA-z",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        ""
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "2SIcUNY0nzmi",
        "colab_type": "code",
        "outputId": "0bbe2de0-103a-4d57-af3f-e9e4ac6e4bc4",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "from google.colab import drive\n",
        "\n",
        "drive.mount('/content/drive')"
      ],
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "E4jtnYbUqlCr",
        "colab_type": "code",
        "outputId": "b3dd8c51-744b-4bf5-ee91-e285071805f0",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "ls"
      ],
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "\u001b[0m\u001b[01;34mdrive\u001b[0m/  \u001b[01;34msample_data\u001b[0m/\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "yTcRbochqlK0",
        "colab_type": "code",
        "outputId": "85f960d9-c076-4e47-c4e2-cc5d7547167b",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "cd 'drive'"
      ],
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "/content/drive\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "aF_oKAq3qlOs",
        "colab_type": "code",
        "outputId": "33b62764-4a16-4c47-c479-08b248de72f0",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "ls"
      ],
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "\u001b[0m\u001b[01;34m'My Drive'\u001b[0m/\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "9HNh8P-yqlU4",
        "colab_type": "code",
        "outputId": "8c6c4897-a449-40b0-c3c9-5ee72e4640ac",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "cd 'My Drive'"
      ],
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "/content/drive/My Drive\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Y9cqDOhHqqnx",
        "colab_type": "code",
        "outputId": "3b0c3a73-a982-4bc2-b3cb-f4303c5c6735",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 207
        }
      },
      "source": [
        "ls"
      ],
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            " checkpoint.pth                       \u001b[0m\u001b[01;36m'HW5 Data'\u001b[0m@\n",
            " \u001b[01;34mClassroom\u001b[0m/                           \u001b[01;34m'HW7 Data'\u001b[0m/\n",
            "'CNN_model_generator.ipynb 的副本'    \u001b[01;34m'HW8 Data'\u001b[0m/\n",
            " cnn_model_generator_ipynb_的副本.py  \u001b[01;34m'HW9 Data'\u001b[0m/\n",
            "\u001b[01;34m'Colab Notebooks'\u001b[0m/                    'Sample file'\n",
            " copy_of_hw8_seq2seq.py               'Sample upload.txt'\n",
            " food-11.zip                          \u001b[01;34m'untitled folder.zip (Unzipped Files)'\u001b[0m/\n",
            " foo.txt                               \u001b[01;34m文件\u001b[0m/\n",
            "\u001b[01;34m'HW3 Data'\u001b[0m/                            \u001b[01;34m機率與統計2019\u001b[0m/\n",
            " \u001b[01;34mHW4\u001b[0m/                                  \u001b[01;34m電子影片\u001b[0m/\n",
            "\u001b[01;34m'HW4 Data'\u001b[0m/\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "xg0yQWW4qqtL",
        "colab_type": "code",
        "outputId": "2fe63cfb-4d6d-48a5-9d03-2e1d16f7dfc8",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "cd 'HW7 Data'"
      ],
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "/content/drive/My Drive/HW7 Data\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "sPutd3xxqsQJ",
        "colab_type": "code",
        "outputId": "b27185c2-aa62-4a19-a8d0-1024223e3094",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 86
        }
      },
      "source": [
        "ls"
      ],
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "16_bit_model.pkl  HW7_model.pt    student_model_from_teacher_fine.bin\n",
            "8_bit_model.pkl   Hw7_output.csv  testing_dataloader.pth\n",
            "\u001b[0m\u001b[01;34mfood-11\u001b[0m/          mean.npy        test_loader.pth\n",
            "food-11.zip       std.npy\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "PQefizA2QMni",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "!find . -name \"*.pyc\" -delete\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "iMPud-7ZtqNV",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "# # Download dataset\n",
        "# !gdown --id '19CzXudqN58R3D-1G8KeFWk8UDQwlb8is' --output food-11.zip\n",
        "# # Unzip the files\n",
        "# !unzip food-11.zip"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "1eI2TcvuzIMl",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import pickle"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "dSNTST3WF0ua",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import re\n",
        "import torch\n",
        "from glob import glob\n",
        "from PIL import Image\n",
        "import torchvision.transforms as transforms\n",
        "\n",
        "class MyDataset(torch.utils.data.Dataset):\n",
        "\n",
        "    def __init__(self, folderName, transform=None):\n",
        "        self.transform = transform\n",
        "        self.data = []\n",
        "        self.label = []\n",
        "\n",
        "        for img_path in sorted(glob(folderName + '/*.jpg')):\n",
        "            try:\n",
        "                # Get classIdx by parsing image path\n",
        "                class_idx = int(re.findall(re.compile(r'\\d+'), img_path)[1])\n",
        "            except:\n",
        "                # if inference mode (there's no answer), class_idx default 0\n",
        "                class_idx = 0\n",
        "\n",
        "            image = Image.open(img_path)\n",
        "            # Get File Descriptor\n",
        "            image_fp = image.fp\n",
        "            image.load()\n",
        "            # Close File Descriptor (or it'll reach OPEN_MAX)\n",
        "            image_fp.close()\n",
        "\n",
        "            self.data.append(image)\n",
        "            self.label.append(class_idx)\n",
        "\n",
        "    def __len__(self):\n",
        "        return len(self.data)\n",
        "\n",
        "    def __getitem__(self, idx):\n",
        "        if torch.is_tensor(idx):\n",
        "            idx = idx.tolist()\n",
        "        image = self.data[idx]\n",
        "        if self.transform:\n",
        "            image = self.transform(image)\n",
        "        return image, self.label[idx]\n",
        "\n",
        "\n",
        "trainTransform = transforms.Compose([\n",
        "    transforms.RandomCrop(256, pad_if_needed=True, padding_mode='symmetric'),\n",
        "    transforms.RandomHorizontalFlip(),\n",
        "    transforms.RandomRotation(15),\n",
        "    transforms.ToTensor(),\n",
        "])\n",
        "testTransform = transforms.Compose([\n",
        "    transforms.CenterCrop(256),\n",
        "    transforms.ToTensor(),\n",
        "])\n",
        "\n",
        "def get_dataloader(mode='training', batch_size=32):\n",
        "\n",
        "    assert mode in ['training', 'testing', 'validation']\n",
        "\n",
        "    dataset = MyDataset(\n",
        "        f'./food-11/{mode}',\n",
        "        transform=trainTransform if mode == 'training' else testTransform)\n",
        "\n",
        "    dataloader = torch.utils.data.DataLoader(\n",
        "        dataset,\n",
        "        batch_size=batch_size,\n",
        "        shuffle=(mode == 'training'))\n",
        "\n",
        "    return dataloader\n",
        "testing_dataloader = get_dataloader('testing', batch_size=32)\n",
        "torch.save(testing_dataloader, 'testing_dataloader.pth')\n",
        "# workspace_dir = str(sys.argv[1])\n",
        "# workspace_dir = './food-11/'\n",
        "# MODLE_PATH = '8_bit_model.pkl'\n",
        "# #test_x = np.load('test_x.npy')\n",
        "\n",
        "# print(\"Loading mean\")\n",
        "# my_mean = np.load(\"mean.npy\")\n",
        "# print(\"mean loaded\")\n",
        "# print(\"Loading std\")\n",
        "# my_std = np.load(\"std.npy\")\n",
        "# print(\"std loaded\")\n",
        "\n",
        "#training 時做 data augmentation\n",
        "\n",
        "\n",
        "\"\"\"# Model\"\"\"\n",
        "\n",
        "import torch.nn as nn\n",
        "import torch.nn.functional as F\n",
        "import torch\n",
        "\n",
        "class StudentNet(nn.Module):\n",
        "    '''\n",
        "      在這個Net裡面，我們會使用Depthwise & Pointwise Convolution Layer來疊model。\n",
        "      你會發現，將原本的Convolution Layer換成Dw & Pw後，Accuracy通常不會降很多。\n",
        "\n",
        "      另外，取名為StudentNet是因為這個Model等會要做Knowledge Distillation。\n",
        "    '''\n",
        "\n",
        "    def __init__(self, base=16, width_mult=1):\n",
        "        '''\n",
        "          Args:\n",
        "            base: 這個model一開始的ch數量，每過一層都會*2，直到base*16為止。\n",
        "            width_mult: 為了之後的Network Pruning使用，在base*8 chs的Layer上會 * width_mult代表剪枝後的ch數量。        \n",
        "        '''\n",
        "        super(StudentNet, self).__init__()\n",
        "        multiplier = [1, 2, 4, 8, 16, 16, 16, 16]\n",
        "\n",
        "        # bandwidth: 每一層Layer所使用的ch數量\n",
        "        bandwidth = [ base * m for m in multiplier]\n",
        "\n",
        "        # 我們只Pruning第三層以後的Layer\n",
        "        for i in range(3, 7):\n",
        "            bandwidth[i] = int(bandwidth[i] * width_mult)\n",
        "\n",
        "        self.cnn = nn.Sequential(\n",
        "            # 第一層我們通常不會拆解Convolution Layer。\n",
        "            nn.Sequential(\n",
        "                nn.Conv2d(3, bandwidth[0], 3, 1, 1),\n",
        "                nn.BatchNorm2d(bandwidth[0]),\n",
        "                nn.ReLU6(),\n",
        "                nn.MaxPool2d(2, 2, 0),\n",
        "            ),\n",
        "            # 接下來每一個Sequential Block都一樣，所以我們只講一個Block\n",
        "            nn.Sequential(\n",
        "                # Depthwise Convolution\n",
        "                nn.Conv2d(bandwidth[0], bandwidth[0], 3, 1, 1, groups=bandwidth[0]),\n",
        "                # Batch Normalization\n",
        "                nn.BatchNorm2d(bandwidth[0]),\n",
        "                # ReLU6 是限制Neuron最小只會到0，最大只會到6。 MobileNet系列都是使用ReLU6。\n",
        "                # 使用ReLU6的原因是因為如果數字太大，會不好壓到float16 / or further qunatization，因此才給個限制。\n",
        "                nn.ReLU6(),\n",
        "                # Pointwise Convolution\n",
        "                nn.Conv2d(bandwidth[0], bandwidth[1], 1),\n",
        "                # 過完Pointwise Convolution不需要再做ReLU，經驗上Pointwise + ReLU效果都會變差。\n",
        "                nn.MaxPool2d(2, 2, 0),\n",
        "                # 每過完一個Block就Down Sampling\n",
        "            ),\n",
        "\n",
        "            nn.Sequential(\n",
        "                nn.Conv2d(bandwidth[1], bandwidth[1], 3, 1, 1, groups=bandwidth[1]),\n",
        "                nn.BatchNorm2d(bandwidth[1]),\n",
        "                nn.ReLU6(),\n",
        "                nn.Conv2d(bandwidth[1], bandwidth[2], 1),\n",
        "                nn.MaxPool2d(2, 2, 0),\n",
        "            ),\n",
        "\n",
        "            nn.Sequential(\n",
        "                nn.Conv2d(bandwidth[2], bandwidth[2], 3, 1, 1, groups=bandwidth[2]),\n",
        "                nn.BatchNorm2d(bandwidth[2]),\n",
        "                nn.ReLU6(),\n",
        "                nn.Conv2d(bandwidth[2], bandwidth[3], 1),\n",
        "                nn.MaxPool2d(2, 2, 0),\n",
        "            ),\n",
        "\n",
        "            # 到這邊為止因為圖片已經被Down Sample很多次了，所以就不做MaxPool\n",
        "            nn.Sequential(\n",
        "                nn.Conv2d(bandwidth[3], bandwidth[3], 3, 1, 1, groups=bandwidth[3]),\n",
        "                nn.BatchNorm2d(bandwidth[3]),\n",
        "                nn.ReLU6(),\n",
        "                nn.Conv2d(bandwidth[3], bandwidth[4], 1),\n",
        "            ),\n",
        "\n",
        "            nn.Sequential(\n",
        "                nn.Conv2d(bandwidth[4], bandwidth[4], 3, 1, 1, groups=bandwidth[4]),\n",
        "                nn.BatchNorm2d(bandwidth[4]),\n",
        "                nn.ReLU6(),\n",
        "                nn.Conv2d(bandwidth[4], bandwidth[5], 1),\n",
        "            ),\n",
        "\n",
        "            nn.Sequential(\n",
        "                nn.Conv2d(bandwidth[5], bandwidth[5], 3, 1, 1, groups=bandwidth[5]),\n",
        "                nn.BatchNorm2d(bandwidth[5]),\n",
        "                nn.ReLU6(),\n",
        "                nn.Conv2d(bandwidth[5], bandwidth[6], 1),\n",
        "            ),\n",
        "\n",
        "            nn.Sequential(\n",
        "                nn.Conv2d(bandwidth[6], bandwidth[6], 3, 1, 1, groups=bandwidth[6]),\n",
        "                nn.BatchNorm2d(bandwidth[6]),\n",
        "                nn.ReLU6(),\n",
        "                nn.Conv2d(bandwidth[6], bandwidth[7], 1),\n",
        "            ),\n",
        "\n",
        "            # 這邊我們採用Global Average Pooling。\n",
        "            # 如果輸入圖片大小不一樣的話，就會因為Global Average Pooling壓成一樣的形狀，這樣子接下來做FC就不會對不起來。\n",
        "            nn.AdaptiveAvgPool2d((1, 1)),\n",
        "        )\n",
        "        self.fc = nn.Sequential(\n",
        "            # 這邊我們直接Project到11維輸出答案。\n",
        "            nn.Linear(bandwidth[7], 11),\n",
        "        )\n",
        "\n",
        "    def forward(self, x):\n",
        "        out = self.cnn(x)\n",
        "        out = out.view(out.size()[0], -1)\n",
        "        return self.fc(out)\n",
        "def decode8(fname):\n",
        "    params = pickle.load(open(fname, 'rb'))\n",
        "    custom_dict = {}\n",
        "    for (name, param) in params.items():\n",
        "        if type(param) == tuple:\n",
        "            min_val, max_val, param = param\n",
        "            param = np.float64(param)\n",
        "            param = (param / 255 * (max_val - min_val)) + min_val\n",
        "            param = torch.tensor(param)\n",
        "        else:\n",
        "            param = torch.tensor(param)\n",
        "\n",
        "        custom_dict[name] = param\n",
        "\n",
        "    return custom_dict\n",
        "\n",
        "\n",
        "\n",
        "# Train\n",
        "\n",
        "# with open(MODLE_PATH, 'rb') as f:\n",
        "#     obj = f.read()\n",
        "# weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}\n",
        "# model_best.load_state_dict(weights)\n",
        "# infile = open(MODLE_PATH,'rb')\n",
        "# model_best = pickle.load(infile)\n",
        "# print(type(model_best))\n",
        "\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ck8nm720zSdn",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "model_best = StudentNet(base=16).cuda()\n",
        "model_best.load_state_dict(decode8(MODLE_PATH))\n",
        "print(\"Model(\" + MODLE_PATH+ \") loaded\")\n",
        "\n",
        "\"\"\"# Testing\n",
        "利用剛剛 train 好的 model 進行 prediction\n",
        "\"\"\"\n",
        "\n",
        "#test_set = ImgDataset(test_x, transform=test_transform)\n",
        "#test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)\n",
        "# test_loader = torch.load('test_loader.pth')\n",
        "\n",
        "model_best.eval()\n",
        "prediction = []\n",
        "with torch.no_grad():\n",
        "    for i, data in enumerate(testing_dataloader):\n",
        "        test_pred = model_best(data)\n",
        "        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)\n",
        "        for y in test_label:\n",
        "            prediction.append(y)\n",
        "name = 'Hw7_output.csv' \n",
        "#將結果寫入 csv 檔\n",
        "with open(name, 'w') as f:\n",
        "    f.write('Id,Category\\n')\n",
        "    for i, y in  enumerate(prediction):\n",
        "        f.write('{},{}\\n'.format(i, y))\n",
        "print(\"Prediction Done\")\n",
        "print(name + \" saved\")"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "pzsAmmRUwqdA",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "from google.colab import files\n",
        "\n",
        "files.download(name)"
      ],
      "execution_count": 0,
      "outputs": []
    }
  ]
}
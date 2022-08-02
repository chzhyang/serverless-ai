import logging
import sys
import uuid
from pathlib import Path

from thrift import Thrift
from thrift.protocol import TBinaryProtocol
from thrift.transport import TSocket, TTransport

gen_py_dir = Path(__file__).resolve().parent.parent.parent / 'gen-py'
sys.path.append(str(gen_py_dir))
import argparse
import base64
import io
import json
import math
import os
import warnings
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from social_network import MediaFilterService
from tensorflow import keras
from torch import nn
from torch.autograd import Variable
from torchvision import models, transforms

logging.getLogger('PIL').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)


class ResNet(nn.Module):
    # ResNet for regression of 3 Euler angles.
    def __init__(self, block, layers, num_classes=1000, dropout=0.1):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        #
        self.dropout = nn.Dropout(dropout)
        self.fc_angles = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # add drop out
        x = self.dropout(x)

        x = self.fc_angles(x)
        return x


def load_filtered_state_dict(model, snapshot, ignore_layer=None, reverse=False, gpu=False):
    model_dict = model.state_dict()
    if reverse:
        #  snapshot keys have prefix 'module.'
        new_snapshot = dict()
        for k, v in snapshot.items():
            name = k[7:]  # remove `module.`
            new_snapshot[name] = v
        snapshot = new_snapshot
    else:
        snapshot = {k: v for k, v in snapshot.items() if k in model_dict}

    if ignore_layer:
        for l in ignore_layer:
            print("ignore_layer : {}".format(snapshot[l]))
            del snapshot[l]

    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


def load_base64_image(base64_str):
    img_str = base64.b64decode(base64_str)
    temp_buff = io.BytesIO()
    temp_buff.write(img_str)
    temp_buff.flush()
    image = Image.open(temp_buff).convert('RGB')
    return image


nsfw_model_path = Path(__file__).resolve().parent / 'data' / 'resnet50-19c8e357.pth'
model = ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 5)
saved_state_dict = torch.load(nsfw_model_path)
load_filtered_state_dict(model, saved_state_dict, ignore_layer=[], reverse=False)
model.share_memory()


def main():
    socket = TSocket.TSocket(host='localhost', port=9090)
    transport = TTransport.TFramedTransport(socket)
    protocol = TBinaryProtocol.TBinaryProtocol(transport)
    client = MediaFilterService.Client(protocol)

    image_path = Path(__file__).resolve().parent / 'data' / '4.jpg'
    base64_path = Path(__file__).resolve().parent / 'data' / '1.png'

    with open(image_path, 'rb') as f:
        base64_str = base64.b64encode(f.read()).decode('ascii')

    with open(base64_path, 'r') as f:
        base64_str = f.read()

    transformations = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])

    batch_size = 2
    imgs = torch.FloatTensor(batch_size, 3, 299, 299)

    imgs[0] = transformations(load_base64_image(base64_str))
    imgs[1] = transformations(Image.open(image_path).convert('RGB'))
    pred = model(imgs)
    _, pred_1 = pred.topk(1, 1, True, True)

    default_class = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']

    logging.info(default_class[pred_1.cpu().numpy()[0][0]])
    logging.info(default_class[pred_1.cpu().numpy()[1][0]])


if __name__ == '__main__':
    try:
        main()
    except Thrift.TException as tx:
        print('%s' % tx.message)

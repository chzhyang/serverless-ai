import base64
import io
import json
import logging
import math
import sys
import time
import warnings
from pathlib import Path

import torch
import torchvision
from PIL import Image
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
from thrift.transport import TSocket, TTransport
from torch import nn
from torchvision import transforms

gen_py_dir = Path(__file__).resolve().parent.parent.parent / 'gen-py'
sys.path.append(str(gen_py_dir))
from social_network import MediaFilterService

warnings.filterwarnings('ignore')
logging.getLogger('PIL').setLevel(logging.WARNING)
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


CLASS = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
TRANSFORM = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])
STATE_DICT_PATH = Path(__file__).resolve().parent / 'data' / 'resnet50-19c8e357.pth'
STATE_DICT = torch.load(STATE_DICT_PATH)
NSFW_MODEL = ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 5)
load_filtered_state_dict(NSFW_MODEL, STATE_DICT, ignore_layer=[], reverse=False)
# NSFW_MODEL.share_memory()
logging.info('NSFW_MODEL loaded')


class MediaFilterServiceHandler:
    def __init__(self):
        pass

    def _load_base64_image(self, base64_str):
        img_str = base64.b64decode(base64_str)
        temp_buff = io.BytesIO()
        temp_buff.write(img_str)
        temp_buff.flush()
        image = Image.open(temp_buff).convert('RGB')
        temp_buff.close()
        return image

    def _classify_base64(self, base64_images):
        images = torch.FloatTensor(len(base64_images), 3, 299, 299)
        for i, image in enumerate(base64_images):
            images[i] = TRANSFORM(self._load_base64_image(base64_str=image))

        filter_list = list()
        category_list = list()

        try:
            pred = NSFW_MODEL(images)
            _, pred_1 = pred.topk(1, 1, True, True)
            for i in range(0, len(base64_images)):
                category = CLASS[pred_1.cpu().numpy()[i][0]]
                category_list.append(category)
                flag = (category != 'porn' and category != 'hentai')
                filter_list.append(flag)
            logging.info('result: {}'.format(category_list))
        except Exception as e:
            logging.error('prediction failed: {}'.format(e))
            for _ in range(0, len(base64_images)):
                filter_list.append(False)
        return filter_list

    def MediaFilter(self, req_id, media_ids, media_types, media_data_list, carrier):
        start = time.time()
        filter_list = self._classify_base64(base64_images=media_data_list)
        end = time.time()
        logging.info('inference time = {0:.2f}s'.format(end - start))

        return filter_list


if __name__ == '__main__':
    host_addr = 'localhost'
    host_port = 9090

    service_config_path = Path(__file__).resolve().parent.parent.parent / 'config' / 'service-config.json'

    with Path(service_config_path).open(mode='r') as f:
        config_json_data = json.load(f)
        host_addr = config_json_data['media-filter-service']['addr']
        host_port = int(config_json_data['media-filter-service']['port'])

    print(host_addr, ' ', host_port)
    handler = MediaFilterServiceHandler()
    processor = MediaFilterService.Processor(handler)
    transport = TSocket.TServerSocket(host=host_addr, port=host_port)
    tfactory = TTransport.TFramedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    server = TServer.TThreadedServer(processor, transport, tfactory, pfactory)

    # Tensorflow is not compatible with TForkingServer
    # server = TServer.TForkingServer(processor, transport, tfactory, pfactory)

    logging.info('Starting the media-filter-service server...')
    server.serve()

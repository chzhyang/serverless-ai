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
import time
import warnings
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from social_network import MediaFilterService
from tensorflow import keras

tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

session = tf.compat.v1.Session()
tf.compat.v1.keras.backend.set_session(session)

IMAGE_DIM = 224
model_path = Path(__file__).resolve().parent / 'data' / 'nsfw_mobilenet_v2_140_224.h5'
model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)
model._make_predict_function()
graph = tf.compat.v1.get_default_graph()
graph.finalize()


def load_base64_image(base64_str, image_size):
    img_str = base64.b64decode(base64_str)
    temp_buff = io.BytesIO()
    temp_buff.write(img_str)
    temp_buff.flush()
    image = Image.open(temp_buff).convert("RGB")
    image = image.resize(size=image_size, resample=Image.NEAREST)
    temp_buff.close()
    image = keras.preprocessing.image.img_to_array(image)
    image /= 255
    return image


def classify_base64(base64_images, image_dim=IMAGE_DIM):
    images = []
    for img in base64_images:
        images.append(load_base64_image(base64_str=img,
                                        image_size=(image_dim, image_dim)))
    images = np.asarray(images)
    probs = classify_nd(images)
    return probs


def classify_nd(nd_images):
    with graph.as_default():
        tf.compat.v1.keras.backend.set_session(session)
        t = time.time()
        model_preds = model.predict(nd_images)
        print(time.time() - t)
    categories = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
    probs = []
    for _, single_preds in enumerate(model_preds):
        single_probs = {}
        for i, pred in enumerate(single_preds):
            single_probs[categories[i]] = float(pred)
        probs.append(single_probs)
    return probs


def main():
    socket = TSocket.TSocket(host='localhost', port=9090)
    transport = TTransport.TFramedTransport(socket)
    protocol = TBinaryProtocol.TBinaryProtocol(transport)
    client = MediaFilterService.Client(protocol)

    image_path = Path(__file__).resolve().parent / 'data' / '0cXfzlu.jpg'
    base64_path = Path(__file__).resolve().parent / 'data' / '1.png'

    with open(image_path, 'rb') as f:
        base64_str = base64.b64encode(f.read()).decode('ascii')

    # with open(base64_path, 'r') as f:
    #     base64_str = f.read()

    probs = classify_base64([base64_str], IMAGE_DIM)

    filter_list = list()
    category_list = list()
    for prob in probs:
        category = max(prob, key=prob.get)
        category_list.append(category)
        flag = (category != "porn" and category != "hentai")
        filter_list.append(flag)

    print(category_list)
    print(filter_list)
    print(json.dumps(probs, indent=2), '\n')


if __name__ == '__main__':
    try:
        main()
    except Thrift.TException as tx:
        print('%s' % tx.message)

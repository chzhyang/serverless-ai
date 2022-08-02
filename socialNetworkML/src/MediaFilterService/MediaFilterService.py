import base64
import io
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from tensorflow import keras
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
from thrift.transport import TSocket, TTransport

gen_py_dir = Path(__file__).resolve().parent.parent.parent / 'gen-py'
sys.path.append(str(gen_py_dir))
from social_network import MediaFilterService

tf.get_logger().setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')

tf.compat.v1.disable_eager_execution()

SESSION = tf.compat.v1.Session()
tf.compat.v1.keras.backend.set_session(SESSION)
IMAGE_DIM = 224
NSFW_MODEL_PATH = Path(__file__).resolve().parent / 'data' / 'nsfw_mobilenet2.224x224.h5'
NSFW_MODEL = tf.keras.models.load_model(NSFW_MODEL_PATH, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)
NSFW_MODEL._make_predict_function()
GRAPH = tf.compat.v1.get_default_graph()
GRAPH.finalize()
logging.info('NSFW_MODEL loaded')


class MediaFilterServiceHandler:
    def __init__(self):
        pass

    def _load_base64_image(self, base64_str):
        global IMAGE_DIM

        img_str = base64.b64decode(base64_str)
        temp_buff = io.BytesIO()
        temp_buff.write(img_str)
        temp_buff.flush()
        image = Image.open(temp_buff).convert('RGB')
        image = image.resize(size=(IMAGE_DIM, IMAGE_DIM),
                             resample=Image.NEAREST)
        temp_buff.close()
        image = keras.preprocessing.image.img_to_array(image)
        image /= 255
        return image

    def _classify_nd(self, nd_images):
        global NSFW_MODEL
        global SESSION

        # logging.info('inference started ...')
        with GRAPH.as_default():
            tf.compat.v1.keras.backend.set_session(SESSION)
            model_preds = NSFW_MODEL.predict(nd_images)
        categories = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
        probs = []
        for _, single_preds in enumerate(model_preds):
            single_probs = {}
            for i, pred in enumerate(single_preds):
                single_probs[categories[i]] = float(pred)
            probs.append(single_probs)
        return probs

    def _classify_base64(self, base64_images):
        # logging.info('loading images ...')
        images = []
        for img in base64_images:
            images.append(self._load_base64_image(base64_str=img))
        images = np.asarray(images)
        # logging.info('finish loading images ...')

        filter_list = list()
        category_list = list()

        try:
            probs = self._classify_nd(images)
            for prob in probs:
                category = max(prob, key=prob.get)
                category_list.append(category)
                flag = (category != 'porn' and category != 'hentai')
                filter_list.append(flag)
            logging.info('result: {}'.format(category_list))
        except Exception as e:
            # logging.error('prediction failed: {}'.format(e))
            for _ in range(0, len(base64_images)):
                filter_list.append(False)
        return filter_list

    def MediaFilter(self, req_id, media_ids, media_types, media_data_list, carrier):
        start = time.time()
        filter_list = self._classify_base64(base64_images=media_data_list)
        end = time.time()
        duration = end - start
        logging.info('inference time = {0:.1f}ms'.format(duration * 1000))
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

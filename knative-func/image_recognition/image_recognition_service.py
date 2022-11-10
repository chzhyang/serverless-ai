import imghdr
import json
import os
import subprocess
import time

import image_preprocess
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference

RESNET_IMAGE_SIZE = 224
INPUTS = 'input'
OUTPUTS = 'predict'
NUM_TOP_PREDICTIONS = 5

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
RESIZE_MIN = 256

class ImageRecognitionService():
  def __init__(self, model_path, lables_path):
    self.model_path = model_path
    self.lables_path = lables_path
    self._optimized_config()
    self.infer_graph, self.infer_sess = self._load_model()
    self.input_tensor = self.infer_graph.get_tensor_by_name('input:0')
    self.output_tensor = self.infer_graph.get_tensor_by_name('predict:0')
    self._cache_model()
    print("Ready for inference...", flush=True)

  def _optimized_config(self):
    num_cores = subprocess.getoutput('lscpu -b -p=Core,Socket | grep -v \'^#\' | sort -u | wc -l')
    
    os.environ["KMP_BLOCKTIME"] = "1"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
    os.environ["OMP_NUM_THREADS"]= num_cores
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(int(num_cores))

  def _load_model(self):
    """Load pretrained model and optimize it for inference"""
    print("Load model...", flush=True)
    infer_graph = tf.Graph()
    with infer_graph.as_default():
      graph_def = tf.compat.v1.GraphDef()
      with tf.compat.v1.gfile.FastGFile(self.model_path, 'rb') as input_file:
        input_graph_content = input_file.read()
        graph_def.ParseFromString(input_graph_content)
        output_graph = optimize_for_inference(graph_def, [INPUTS], [OUTPUTS], dtypes.float32.as_datatype_enum, False)
      tf.import_graph_def(output_graph, name='')
      infer_sess = tf.compat.v1.Session(graph=infer_graph)
    return infer_graph, infer_sess
    
  def _cache_model(self):
    """Use random data to warm up model inference"""
    print("Cache model...", flush=True)
    data_graph = tf.Graph()
    with data_graph.as_default():
      input_shape = [1, RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE, 3]
      images = tf.random.uniform(input_shape, 0.0, 255.0, dtype=tf.float32, name='synthetic_images')
    data_sess = tf.compat.v1.Session(graph=data_graph)
    image_np = data_sess.run(images)
    self.infer_sess.run(self.output_tensor, feed_dict={self.input_tensor: image_np})

  def _data_preprocess(self, data_location, batch_size, output_height, output_width, num_channels=3):
    """Read the given image, and process it to 3-D tensor that will be accepted by the session"""
    data_graph = tf.Graph()
    with data_graph.as_default():
      if imghdr.what(data_location) != "jpeg":
            raise ValueError("At this time, only JPEG images are supported, please try another image.")
      image_buffer = tf.io.read_file(data_location)
      # image_array = image_preprocess.preprocess_image(image_buffer, output_height, output_width, num_channels)
      # Decoding, cropping, and resizing the image
      image = tf.image.decode_jpeg(image_buffer, channels=num_channels)
      image = image_preprocess.image_resize(image, RESIZE_MIN)
      image = image_preprocess.central_crop(image, output_height, output_width)
      image.set_shape([output_height, output_width, num_channels])
      image = image_preprocess.mean_image_subtraction(image, CHANNEL_MEANS, num_channels)
      input_shape = [batch_size, output_height, output_width, num_channels]
      images = tf.reshape(image, input_shape)
    data_sess = tf.compat.v1.Session(graph=data_graph)
    image_np = data_sess.run(images)

    return image_np

  def _get_labels(self):
    """Get the set of possible labels for classification"""
    with open(self.lables_path, "r") as labels_file:
      labels = json.load(labels_file)

    return labels

  def _top_predictions(self, result, n):
    """Get the top n predictions given the array of softmax results"""
    probabilities = result
    # Get the ids of most probable labels. Reverse order to get greatest first.
    ids = np.argsort(probabilities)[::-1]
    
    return ids[:n]

  def _get_labels_for_ids(self, labels, ids, ids_are_one_indexed=False):
    """Get the human-readable labels for given ids"""
    return [labels[str(x + int(ids_are_one_indexed))] for x in ids]

  def _get_top_predictions(self, results, ids_are_one_indexed=False, preds_to_print=5):
    """Given an array of mode, graph_name, predicted_ID, print labels"""
    labels = self._get_labels()
    predictions = []
    for result in results:
      pred_ids = self._top_predictions(result, preds_to_print)
      pred_labels = self._get_labels_for_ids(labels, pred_ids, ids_are_one_indexed)
      predictions.append(pred_labels)

    return predictions

  def run_inference(self, data_location):
    """Run inference from given image, returns human-readable predictions"""
    start_time = time.time()
    image_np = self._data_preprocess(data_location, 1, RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE, 3)
    data_time = time.time() - start_time

    start_time = time.time()
    predictions = self.infer_sess.run(self.output_tensor, feed_dict={self.input_tensor: image_np})
    infer_time = (time.time() - start_time)

    predictions_labels = self._get_top_predictions(predictions, False, NUM_TOP_PREDICTIONS)

    return predictions_labels, data_time * 1000, infer_time * 1000


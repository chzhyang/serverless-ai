import imghdr
import json
import os
import subprocess
import time

import data_preprocess
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference

# Basic information of the pretrained Resnet50 model
RESNET_IMAGE_SIZE = 224
INPUTS = 'input_tensor'
INPUT_TENSOR = 'input_tensor:0'
OUTPUTS = 'softmax_tensor'
OUTPUT_TENSOR = 'softmax_tensor:0'
NUM_CHANNELS = 3

class ImageRecognitionService():
  """
  Class for image recognition inference with optimized TensorFlow

  Attributes
    ----------
    model_path: Path of pretained model
  
  """
  def __init__(self, model_path):
    """Config TensorFlow configuration settings, then load a pretrained model and cache it"""
    self.model_path = model_path
    self._optimized_config()
    self.infer_graph, self.infer_sess = self._load_model()
    self.input_tensor = self.infer_graph.get_tensor_by_name(INPUT_TENSOR)
    self.output_tensor = self.infer_graph.get_tensor_by_name(OUTPUT_TENSOR)
    self._cache_model()
    print("Ready for inference...", flush=True)

  def _optimized_config(self):
    """TensorFlow configuration settings"""
    # Get all physical cores
    num_cores = subprocess.getoutput('lscpu -b -p=Core,Socket | grep -v \'^#\' | sort -u | wc -l')

    # Environment variables
    os.environ["KMP_SETTINGS"] = "1"
    # Time(milliseconds) that a thread should wait, after completing the execution of a parallel region, before sleeping.
    os.environ["KMP_BLOCKTIME"] = "1"
    # Controls how threads are distributed and ultimately bound to specific processing units
    os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
    # Maximum number of threads available for the OpenMP runtime
    os.environ["OMP_NUM_THREADS"]= num_cores
    
    # TensorFlow runtime settings
    # Number of thread pools to use for a TensorFlow session
    tf.config.threading.set_inter_op_parallelism_threads(1)
    # Number of threads in each threadpool to use for a TensorFlow session
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
      
      # Optimize the model for inference
      output_graph = optimize_for_inference(graph_def, [INPUTS], [OUTPUTS], dtypes.float32.as_datatype_enum, False)
      tf.import_graph_def(output_graph, name='')
    infer_sess = tf.compat.v1.Session(graph=infer_graph)

    return infer_graph, infer_sess
    
  def _cache_model(self):
    """Use random data to warm up model inference session"""
    print("Cache model...", flush=True)
    data_graph = tf.Graph()
    with data_graph.as_default():
      input_shape = [1, RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE, NUM_CHANNELS]
      images = tf.random.uniform(input_shape, 0.0, 255.0, dtype=tf.float32, name='synthetic_images')
    data_sess = tf.compat.v1.Session(graph=data_graph)
    image_np = data_sess.run(images)
    self.infer_sess.run(self.output_tensor, feed_dict={self.input_tensor: image_np})

  def _data_preprocess(self, data_location, batch_size, output_height, output_width, num_channels):
    """Read the image, then process it to a 3-D tensor for tensorflow"""
    data_graph = tf.Graph()
    with data_graph.as_default():
      if imghdr.what(data_location) != "jpeg":
            raise ValueError("At this time, only JPEG images are supported, please try another image.")
      image_buffer = tf.io.read_file(data_location)
      image = data_preprocess.image_preprocess(image_buffer, output_height, output_width, num_channels)
      input_shape = [batch_size, output_height, output_width, num_channels]
      images = tf.reshape(image, input_shape)
      data_sess = tf.compat.v1.Session(graph=data_graph)
      image = data_sess.run(images)

    return image

  def _get_labels(self, lables_path):
    """Get the set of possible labels for classification"""
    with open(lables_path, "r") as labels_file:
      labels = json.load(labels_file)

    return labels

  def _top_predictions(self, result, n):
    """Get the top n predictions given the array of softmax results"""
    # Only care about the first example
    probabilities = result
    # Get the ids of most probable labels. Reverse order to get greatest first
    ids = np.argsort(probabilities)[::-1]
    
    return ids[:n]

  def _get_labels_for_ids(self, labels, ids, ids_are_one_indexed=False):
    """Get the human-readable labels for given ids"""
    return [labels[str(x + int(ids_are_one_indexed))] for x in ids]

  def _get_top_predictions(self, results, lables_path, ids_are_one_indexed=False, preds_to_print=5):
    """Given an array of mode, graph_name, predicted_ID, print labels"""
    labels = self._get_labels(lables_path)
    predictions = []
    for result in results:
      pred_ids = self._top_predictions(result, preds_to_print)
      pred_labels = self._get_labels_for_ids(labels, pred_ids, ids_are_one_indexed)
      predictions.append(pred_labels)

    return predictions

  def run_inference(self, data_location, lables_path, num_top_preds):
    """
    Inference given image, Returns human-readable predictions.

    Preprocess the image to tensor which can be accepted by tensorflow, 
    then run inference, and get human-readable predictions lastly.

    """
    image = self._data_preprocess(data_location, 1, RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE, NUM_CHANNELS)
    predictions = self.infer_sess.run(self.output_tensor, feed_dict={self.input_tensor: image})
    predictions_labels = self._get_top_predictions(predictions, lables_path, False, num_top_preds)

    return predictions_labels


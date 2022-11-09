import imghdr
import json
import os
import numpy as np
import requests
import os
import subprocess
import time

import tensorflow as tf
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from tensorflow.python.framework import dtypes

import imagenet_preprocessing

RESNET_IMAGE_SIZE = 224
INPUTS = 'input'
OUTPUTS = 'predict'
INPUT_PATH = "./data/"
LABELS_FILE = "./data/labellist.json"
RESNET_IMAGE_SIZE = 224

def optimized_config():
  # Get all physical cores
  num_physical_cores = subprocess.getoutput('lscpu -b -p=Core,Socket | grep -v \'^#\' | sort -u | wc -l')
  os.environ["KMP_BLOCKTIME"] = "1"
  os.environ["KMP_SETTINGS"] = "1"
  os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
  os.environ["OMP_NUM_THREADS"]= num_physical_cores
  
  tf.config.threading.set_inter_op_parallelism_threads(1)
  tf.config.threading.set_intra_op_parallelism_threads(int(num_physical_cores))

def load_model(input_graph):
  # Load model
  print("Load model...", flush=True)
  infer_graph = tf.Graph()
  with infer_graph.as_default():
    graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v1.gfile.FastGFile(input_graph, 'rb') as input_file:
      input_graph_content = input_file.read()
      graph_def.ParseFromString(input_graph_content)
      output_graph = optimize_for_inference(graph_def, [INPUTS], [OUTPUTS], dtypes.float32.as_datatype_enum, False)
    tf.import_graph_def(output_graph, name='')
  # Use random data to cache model
  print("Cache model...", flush=True)
  data_graph = tf.Graph()
  with data_graph.as_default():
    input_shape = [1, RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE, 3]
    images = tf.random.uniform(input_shape, 0.0, 255.0, dtype=tf.float32, name='synthetic_images')
  input_tensor = infer_graph.get_tensor_by_name('input:0')
  output_tensor = infer_graph.get_tensor_by_name('predict:0')

  data_sess = tf.compat.v1.Session(graph=data_graph)
  infer_sess = tf.compat.v1.Session(graph=infer_graph)
  
  image_np = data_sess.run(images)
  infer_sess.run(output_tensor, feed_dict={input_tensor: image_np})
  print("##########   Ready for inference   ##########", flush=True)
  return infer_graph, infer_sess

def download_image(img_url):
  """Download image from URL to default filepath"""
  if not os.path.exists(INPUT_PATH):
      os.makedirs(INPUT_PATH)
  img_name = img_url.split('/')[-1].split('.')[0]+'.jpg'
  img_filepath = os.path.join(INPUT_PATH, img_name)
  if not os.path.exists(img_filepath):
    img_data = requests.get(img_url)
    with open(img_filepath, 'wb') as f:
      f.write(img_data.content)
    print("Download image to ", img_filepath, flush=True)
  else:
    print("Image exists: ", img_filepath)
  return img_filepath

def data_preprocess(data_location, batch_size, output_height, output_width, num_channels=3):
  if imghdr.what(data_location) != "jpeg":
        raise ValueError("At this time, only JPEG images are supported, please try another image.")
  image_buffer = tf.io.read_file(data_location)
  image_array = imagenet_preprocessing.preprocess_image(image_buffer,None,output_height,output_width,num_channels,False)
  input_shape = [batch_size, RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE, 3]
  image_array_tensor = tf.reshape(image_array, input_shape)
  return image_array_tensor

def run_inference(data_location, infer_graph, infer_sess):
  """Run inference"""
  start_time = time.time()
  data_graph = tf.Graph()
  with data_graph.as_default():
    if (data_location):
      images = data_preprocess(data_location, 1, RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE, 3)
  data_sess = tf.compat.v1.Session(graph=data_graph)
  image_np = data_sess.run(images)
  data_time = time.time() - start_time

  input_tensor = infer_graph.get_tensor_by_name('input:0')
  output_tensor = infer_graph.get_tensor_by_name('predict:0')
  start_time = time.time()
  for i in range(1):
    predictions = infer_sess.run(output_tensor, feed_dict={input_tensor: image_np})
  infer_time = (time.time() - start_time)/2

  return predictions, data_time * 1000, infer_time * 1000

def get_top_predictions(results, ids_are_one_indexed=False, preds_to_print=5):
  """Given an array of mode, graph_name, predicted_ID, print labels"""
  labels = get_labels()

  predictions = []
  for result in results:
    pred_ids = top_predictions(result, preds_to_print)
    pred_labels = get_labels_for_ids(labels, pred_ids, ids_are_one_indexed)
    predictions.append(pred_labels)
  return predictions

def get_labels():
  """Get the set of possible labels for classification"""
  with open(LABELS_FILE, "r") as labels_file:
    labels = json.load(labels_file)

  return labels

def top_predictions(result, n):
  """Get the top n predictions given the array of softmax results"""
  # Only care about the first example.
  probabilities = result
  # Get the ids of most probable labels. Reverse order to get greatest first.
  ids = np.argsort(probabilities)[::-1]
  
  return ids[:n]

def get_labels_for_ids(labels, ids, ids_are_one_indexed=False):
  """Get the human-readable labels for given ids.
  Args:
    labels: dict, string-ID to label mapping from ImageNet.
    ids: list of ints, IDs to return labels for.
    ids_are_one_indexed: whether to increment passed IDs by 1 to account for
      the background category. See ArgParser `--ids_are_one_indexed`
      for details.
  Returns:
    list of category labels
  """
  return [labels[str(x + int(ids_are_one_indexed))] for x in ids]
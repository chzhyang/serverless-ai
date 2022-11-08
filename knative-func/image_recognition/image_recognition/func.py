import os
import subprocess
import time
import requests

from parliament import Context
from flask import Request, jsonify, make_response
import tensorflow as tf
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from tensorflow.python.framework import dtypes

import utils
import imagenet_preprocessing
import imghdr

MODEL_PATH = 'models/resnet50_fp32_pretrained_model.pb'
INPUTS = 'input'
OUTPUTS = 'predict'
RESNET_IMAGE_SIZE = 224

TEST_INPUT_DATA = './data/ILSVRC2012_test_00000181_1.JPEG'
INPUT_PATH = './data/'

# """Run standard ImageNet preprocessing on the passed image file.
  # Args:
  #   file_name: string, path to file containing a JPEG image
  #   output_height: int, final height of image
  #   output_width: int, final width of image
  #   num_channels: int, depth of input image
  # Returns:
  #   Float array representing processed image with shape
  #     [output_height, output_width, num_channels]
  # Raises:
  #   ValueError: if image is not a JPEG.
  # """
def optimize_graph_config():
  # Get all physical cores
  # num_physical_cores = subprocess.getoutput('lscpu -b -p=Core,Socket | grep -v \'^#\' | sort -u | wc -l')
  num_physical_cores = "56"
  ### log
  print("num_physical_cores = ",num_physical_cores)
  os.environ["KMP_BLOCKTIME"] = "1"
  os.environ["KMP_SETTINGS"] = "1"
  os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0" # when hyperthreading is enabled
  os.environ["OMP_NUM_THREADS"]= num_physical_cores
  tf.config.threading.set_inter_op_parallelism_threads(1)
  tf.config.threading.set_intra_op_parallelism_threads(int(num_physical_cores))

# def optimize_graph_config():
#   print("Optimize graph...")
#   data_config = tf.compat.v1.ConfigProto()
#   # TODO: read from config.json
#   data_config.intra_op_parallelism_threads = num_physical_cores
#   data_config.inter_op_parallelism_threads = 1
#   # data_config.intra_op_parallelism_threads = data_num_intra_threads
#   # data_config.inter_op_parallelism_threads = data_num_inter_threads
#   data_config.use_per_session_threads = 1

#   infer_config = tf.compat.v1.ConfigProto()
#   infer_config.intra_op_parallelism_threads = num_physical_cores
#   infer_config.inter_op_parallelism_threads = 1
#   infer_config.use_per_session_threads = 1

#   return data_config, infer_config

def data_preprocess(data_location, batch_size, output_height, output_width, num_channels=3):
  if imghdr.what(data_location) != "jpeg":
        raise ValueError("At this time, only JPEG images are supported, please try another image.")
  image_buffer = tf.io.read_file(data_location)
  image_array = imagenet_preprocessing.preprocess_image(image_buffer,None,output_height,output_width,num_channels,False)
  input_shape = [batch_size, RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE, 3]
  image_array_tensor = tf.reshape(image_array, input_shape)
  return image_array_tensor

def load_model(input_graph):
  print("Loading model...")
  infer_graph = tf.Graph()
  with infer_graph.as_default():
    graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v1.gfile.FastGFile(input_graph, 'rb') as input_file:
      input_graph_content = input_file.read()
      graph_def.ParseFromString(input_graph_content)
      output_graph = optimize_for_inference(graph_def, [INPUTS], [OUTPUTS], dtypes.float32.as_datatype_enum, False)
      # output_graph = graph_def
    tf.import_graph_def(output_graph, name='')
  return infer_graph

def cache_model(infer_graph):
  print("Cache model...")
  data_graph = tf.Graph()
  with data_graph.as_default():
    input_shape = [1, RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE, 3]
    images = data_preprocess(TEST_INPUT_DATA, 1, RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE, 3)
  input_tensor = infer_graph.get_tensor_by_name('input:0')
  output_tensor = infer_graph.get_tensor_by_name('predict:0')

  data_sess = tf.compat.v1.Session(graph=data_graph)
  infer_sess = tf.compat.v1.Session(graph=infer_graph)
  
  image_np = data_sess.run(images)
  predictions = infer_sess.run(output_tensor, feed_dict={input_tensor: image_np})
  predictions = infer_sess.run(output_tensor, feed_dict={input_tensor: image_np})
  predictions = infer_sess.run(output_tensor, feed_dict={input_tensor: image_np})
  predictions = infer_sess.run(output_tensor, feed_dict={input_tensor: image_np})
  predictions = infer_sess.run(output_tensor, feed_dict={input_tensor: image_np})
  predictions = infer_sess.run(output_tensor, feed_dict={input_tensor: image_np})

def run_inference(data_location, infer_graph):
  """run inference"""
  # data_config, infer_config = optimize_graph_config()
  print("Run inference...")
  print("Data processing...")
  data_graph = tf.Graph()
  with data_graph.as_default():
    if (data_location):
      images = data_preprocess(data_location, 1, RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE, 3)
  input_tensor = infer_graph.get_tensor_by_name('input:0')
  output_tensor = infer_graph.get_tensor_by_name('predict:0')

  data_sess = tf.compat.v1.Session(graph=data_graph)
  infer_sess = tf.compat.v1.Session(graph=infer_graph)
  
  image_np = data_sess.run(images)
  predictions = infer_sess.run(output_tensor, feed_dict={input_tensor: image_np})
  # print("Loading model...")
  # infer_graph = tf.Graph()
  # with infer_graph.as_default():
  #   graph_def = tf.compat.v1.GraphDef()
  #   with tf.compat.v1.gfile.FastGFile(input_graph, 'rb') as input_file:
  #     input_graph_content = input_file.read()
  #     graph_def.ParseFromString(input_graph_content)
  #     output_graph = optimize_for_inference(graph_def, [INPUTS], [OUTPUTS], dtypes.float32.as_datatype_enum, False)
  #   tf.import_graph_def(output_graph, name='')

  # Define input and output Tensors for detection_graph
  input_tensor = infer_graph.get_tensor_by_name('input:0')
  output_tensor = infer_graph.get_tensor_by_name('predict:0')

  data_sess = tf.compat.v1.Session(graph=data_graph)
  infer_sess = tf.compat.v1.Session(graph=infer_graph)
  
  start_time = time.time()
  image_np = data_sess.run(images)
  predictions = infer_sess.run(output_tensor, feed_dict={input_tensor: image_np})
  total_time = time.time() - start_time

  return predictions, total_time * 1000

def request_handler(req: Request, infer_graph) -> str:
  if req.method == "GET":
    # data = {}
    # graph = image_classifier(1,MODEL_NAME,MODEL_PATH,TEST_INPUT_DATA,1,36)
    predictions, latency= run_inference(TEST_INPUT_DATA, infer_graph)
    predictions_lables = utils.get_top_predictions(predictions, False, 5)
    data = {
      "top_predictions" : predictions_lables, 
      "inference_latency(ms)" : latency
    }
    # headers = { "content-type": "application/json" }
    return jsonify(data)
  elif req.method == "POST":
    img_url = req.form.get('imgURL')
    print("image url: ", img_url)
    # Download image from URL
    if not os.path.exists(INPUT_PATH):
      os.makedirs(INPUT_PATH)
    input_name = img_url.split('/')[-1].split('.')[0]+'.jpg'
    input_filepath = os.path.join(INPUT_PATH, input_name)
    if not os.path.exists(input_filepath):
      input_data = requests.get(img_url)
      with open(input_filepath, 'wb') as f:
        f.write(input_data.content)
    # Run inference
    predictions, latency = run_inference(input_filepath, infer_graph)
    predictions_lables = utils.get_top_predictions(predictions, False, 5)
    result = {
      "top5_predictions" : predictions_lables, 
      "inference_latency(ms)" : latency
    }
    # headers = { "content-type": "application/json" }
    return jsonify(result)

def main(context: Context):
  """
  Image classifier with optimized TensorFlow graph
  """
  optimize_graph_config()
  infer_graph = load_model(MODEL_PATH)
  cache_model(infer_graph)
  if 'request' in context.keys():
    return request_handler(context.request, infer_graph)
  else:
    # test
    print("Empty request", flush=True)
    img_url = "https://github.com/chzhyang/faas-workloads/blob/49bceeb337dab59e1523fb7316469904abd9cf4f/tensorflow/image_recognition/tensorflow_image_classification/data/ILSVRC2012_test_00000181.JPEG"
    print("image url: ", img_url)
    # Download image from URL
    if not os.path.exists(INPUT_PATH):
      os.makedirs(INPUT_PATH)
    input_name = img_url.split('/')[-1]
    input_filepath = os.path.join(INPUT_PATH, input_name)
    if not os.path.exists(input_filepath):
      input_data = requests.get(img_url)
      with open(input_filepath, 'wb') as f:
        f.write(input_data.content)
    print("input_filepath:",input_filepath)
    # Run inference
    predictions, latency = run_inference(input_filepath, infer_graph)
    predictions_lables = utils.get_top_predictions(predictions, False, 5)
    result = {
      "top5_predictions" : predictions_lables, 
      "inference_latency(ms)" : latency
    }
    print("inference_latency(ms)", latency)
    # headers = { "content-type": "application/json" }
    return jsonify(result)
    # test end
    # print("Empty request", flush=True)
    # return "{}", 400
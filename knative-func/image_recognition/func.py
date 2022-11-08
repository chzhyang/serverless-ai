import os
import subprocess
import time
import requests
import imghdr

from parliament import Context
from flask import Request, jsonify

import tensorflow as tf
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from tensorflow.python.framework import dtypes

import utils
import imagenet_preprocessing


MODEL_PATH = 'models/resnet50_fp32_pretrained_model.pb'
INPUTS = 'input'
OUTPUTS = 'predict'
RESNET_IMAGE_SIZE = 224

TEST_IMAGE = './data/test.JPEG'

# """Run standard ImageNet preprocessing on image file.
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
def optimize_config():
  # Get all physical cores
  num_physical_cores = subprocess.getoutput('lscpu -b -p=Core,Socket | grep -v \'^#\' | sort -u | wc -l')
  os.environ["KMP_BLOCKTIME"] = "1"
  os.environ["KMP_SETTINGS"] = "1"
  os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
  os.environ["OMP_NUM_THREADS"]= num_physical_cores
  tf.config.threading.set_inter_op_parallelism_threads(1)
  tf.config.threading.set_intra_op_parallelism_threads(int(num_physical_cores))

def data_preprocess(data_location, batch_size, output_height, output_width, num_channels=3):
  if imghdr.what(data_location) != "jpeg":
        raise ValueError("At this time, only JPEG images are supported, please try another image.")
  image_buffer = tf.io.read_file(data_location)
  image_array = imagenet_preprocessing.preprocess_image(image_buffer,None,output_height,output_width,num_channels,False)
  input_shape = [batch_size, RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE, 3]
  image_array_tensor = tf.reshape(image_array, input_shape)
  return image_array_tensor

def load_model(input_graph):
  # Load model
  infer_graph = tf.Graph()
  with infer_graph.as_default():
    graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v1.gfile.FastGFile(input_graph, 'rb') as input_file:
      input_graph_content = input_file.read()
      graph_def.ParseFromString(input_graph_content)
      output_graph = optimize_for_inference(graph_def, [INPUTS], [OUTPUTS], dtypes.float32.as_datatype_enum, False)
    tf.import_graph_def(output_graph, name='')
  # Cache model
  data_graph = tf.Graph()
  with data_graph.as_default():
    input_shape = [1, RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE, 3]
    images = tf.random.uniform(input_shape, 0.0, 255.0, dtype=tf.float32, name='synthetic_images')
    # images = data_preprocess(TEST_INPUT_DATA, 1, RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE, 3)
  input_tensor = infer_graph.get_tensor_by_name('input:0')
  output_tensor = infer_graph.get_tensor_by_name('predict:0')

  data_sess = tf.compat.v1.Session(graph=data_graph)
  infer_sess = tf.compat.v1.Session(graph=infer_graph)
  
  image_np = data_sess.run(images)
  for i in range(1):
    infer_sess.run(output_tensor, feed_dict={input_tensor: image_np})
  return infer_graph, infer_sess

def run_inference(data_location, infer_graph, infer_sess):
  """Run inference"""
  data_graph = tf.Graph()
  with data_graph.as_default():
    if (data_location):
      images = data_preprocess(data_location, 1, RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE, 3)
  input_tensor = infer_graph.get_tensor_by_name('input:0')
  output_tensor = infer_graph.get_tensor_by_name('predict:0')
  data_sess = tf.compat.v1.Session(graph=data_graph)
  image_np = data_sess.run(images)

  start_time = time.time()
  for i in range(2):
    predictions = infer_sess.run(output_tensor, feed_dict={input_tensor: image_np})
  total_time = (time.time() - start_time)/2

  return predictions, total_time * 1000

def request_handler(req: Request, infer_graph, infer_sess) -> str:
  """Handle the request"""
  if req.method == "GET":
    predictions, latency= run_inference(TEST_IMAGE, infer_graph, infer_sess)
    predictions_lables = utils.get_top_predictions(predictions, False, 5)
    result = {
      "top5_predictions" : predictions_lables, 
      "inference_latency(ms)" : latency
    }
    # headers = { "content-type": "application/json" }
    return jsonify(result)
  elif req.method == "POST":
    img_url = req.form.get('imgURL')
    img_filepath = utils.download_image(img_url)
    predictions, latency = run_inference(img_filepath, infer_graph, infer_sess)
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
  optimize_config()
  infer_graph, infer_sess = load_model(MODEL_PATH)
  if 'request' in context.keys():
    return request_handler(context.request, infer_graph, infer_sess)
  else:
    # performance test
    img_url = "https://raw.githubusercontent.com/chzhyang/faas-workloads/main/tensorflow/image_recognition/tensorflow_image_classification/data/ILSVRC2012_test_00000181.JPEG"
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
    predictions, latency = run_inference(input_filepath, infer_graph, infer_sess)
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
    # print("Empty request", flush=True)
    # return "{}", 400
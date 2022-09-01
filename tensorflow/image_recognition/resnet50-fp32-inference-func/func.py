from parliament import Context
from flask import Request
import json
from distutils.command.config import config
import time
import tensorflow as tf
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from tensorflow.python.framework import dtypes

import imagenet_preprocessing  # pylint: disable=g-bad-import-order
import imghdr
import datasets
import numpy as np
import os

LABELS_FILE = "labellist.json"
MODEL_NAME = 'resnet50'
MODLE_PATH = 'resnet50_fp32_pretrained_model.pb'
INPUTS = 'input'
OUTPUTS = 'predict'
RESNET_IMAGE_SIZE = 224

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

class image_classifier_optimized_graph:
  """Evaluate image classifier with optimized TensorFlow graph"""
  batch_size = 1
  model_name = MODEL_NAME
  input_graph = ""
  data_location = ""
  results_file_path = "" # need define+time
  # optimize options
  num_inter_threads = 1 
  num_intra_threads = 36 # physical cores
  data_num_inter_threads = 32
  data_num_intra_threads = 14
  num_cores = 28

  def __init__(self, 
               batch_size, 
               model_name, 
               input_graph, 
               data_location, 
               num_inter_threads=1, 
               num_intra_threads=36):
    self.batch_size = batch_size
    self.model_name = model_name
    self.input_graph = input_graph
    self.data_location = data_location
    self.num_inter_threads = num_inter_threads
    self.num_intra_threads = num_intra_threads 
    self.calibrate = False


  # Write out the file name, expected label, and top prediction
  def write_results_output(self, predictions, filenames, labels):
    top_predictions = np.argmax(predictions, 1)
    with open(self.results_file_path, "a") as fp:
      for filename, expected_label, top_prediction in zip(filenames, labels, top_predictions):
        fp.write("{},{},{}\n".format(filename, expected_label, top_prediction))

  def optimize_graph(self):
    print("Optimize graph")
    data_config = tf.compat.v1.ConfigProto()
    data_config.intra_op_parallelism_threads = self.data_num_intra_threads
    data_config.inter_op_parallelism_threads = self.data_num_inter_threads
    data_config.use_per_session_threads = 1

    infer_config = tf.compat.v1.ConfigProto()
    infer_config.intra_op_parallelism_threads = self.num_intra_threads
    infer_config.inter_op_parallelism_threads = self.num_inter_threads
    infer_config.use_per_session_threads = 1

    return data_config, infer_config

  def data_preprocess(self, file_name, batch_size, output_height=224, output_width=224,
                      num_channels=3):
    if imghdr.what(self.data_location) != "jpeg":
          raise ValueError("At this time, only JPEG images are supported. "
                        "Please try another image.")
    image_buffer = tf.io.read_file(file_name)
    image_array = imagenet_preprocessing.preprocess_image(
        image_buffer=image_buffer,
        bbox=None,
        output_height=output_height,
        output_width=output_width,
        num_channels=num_channels,
        is_training=False)
    # todo: shape, batchsize
    return [image_array]

  def run(self):
    """run inference with optimized graph"""
    data_config, infer_config = self.optimize_graph()

    print("Data preprocess")
    data_graph = tf.Graph()
    with data_graph.as_default():
      if (self.data_location):
        print("Inference with real data.")
        images = self.data_preprocess(self.data_location, RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE, 3)
      else:
        print("Inference with dummy data.")
        input_shape = [self.batch_size, RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE, 3]
        # images = np.random.random_sample(input_shape).astype(np.float32)
        images = tf.random.uniform(input_shape, 0.0, 255.0, dtype=tf.float32, name='synthetic_images')

    print("Run inference")
    infer_graph = tf.Graph()
    with infer_graph.as_default():
      graph_def = tf.compat.v1.GraphDef()
      with tf.compat.v1.gfile.FastGFile(self.input_graph, 'rb') as input_file:
        input_graph_content = input_file.read()
        graph_def.ParseFromString(input_graph_content)

      output_graph = optimize_for_inference(graph_def, [INPUTS], 
                              [OUTPUTS], dtypes.float32.as_datatype_enum, False)
      tf.import_graph_def(output_graph, name='')

    # Define input and output Tensors for detection_graph
    input_tensor = infer_graph.get_tensor_by_name('input:0')
    output_tensor = infer_graph.get_tensor_by_name('predict:0')

    data_sess = tf.compat.v1.Session(graph=data_graph,  config=data_config)
    infer_sess = tf.compat.v1.Session(graph=infer_graph, config=infer_config)

    total_time = 0
    
    data_load_start = time.time()
    image_np = data_sess.run(images)
    data_load_time = time.time() - data_load_start

    start_time = time.time()
    predictions = infer_sess.run(output_tensor, feed_dict={input_tensor: image_np})
    time_consume = time.time() - start_time

    total_time = time_consume
    # only add data loading time for real data, not for dummy data
    if self.data_location:
      total_time += data_load_time
    return predictions, total_time

def inference(req: Request) -> str:
    if req.method == "GET":
        inference = image_classifier_optimized_graph(1,MODEL_NAME,"resnet50_fp32_pretrained_model.pb","","./",1,False)
        prediction, inference_latency = inference.run()
        return {'prediction': prediction, 'inference_latency': inference_latency}
    elif req.method == "POST":
        print("request form: ", req.form)
        print("request url: ", req.form.get('url'))
        input_url = req.form.get('url')
        inference = image_classifier_optimized_graph(1,MODEL_NAME,"resnet50_fp32_pretrained_model.pb",input_url,"./",1,False)
        prediction, inference_latency = inference.run()
        return {'prediction': prediction, 'inference_latency': inference_latency}
  
def get_top_predictions(results, ids_are_one_indexed=False, preds_to_print=5):
  """Given an array of mode, graph_name, predicted_ID, print labels."""
  labels = get_labels()

  print("Predictions:")
  predictions = []
  for result in results:
    pred_ids = top_predictions(result, preds_to_print)
    pred_labels = get_labels_for_ids(labels, pred_ids, ids_are_one_indexed)
    predictions.append(pred_labels)
  return predictions

def get_labels():
  """Get the set of possible labels for classification."""
  with open(LABELS_FILE, "r") as labels_file:
    labels = json.load(labels_file)

  return labels

def top_predictions(result, n):
  """Get the top n predictions given the array of softmax results."""
  # We only care about the first example.
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
  for x in ids:
    print(x, "=", labels[str(x + int(ids_are_one_indexed))])
  return [labels[str(x + int(ids_are_one_indexed))] for x in ids]

def main(context: Context):
    """ 
    Function template
    The context parameter contains the Flask request object and any
    CloudEvent received with the request.
    """
    print(tf.__version__)
    # Add your business logic here
    print("Received request")

    if 'request' in context.keys():
        ret = inference(context.request)
        print(ret, flush=True)
        return inference(context.request), 200
    else:
        # print("Empty request", flush=True)
        # return "{}", 200
        inference = image_classifier_optimized_graph(
                                                    1,
                                                    MODEL_NAME,
                                                    MODLE_PATH,
                                                    "data/ILSVRC2012_test_00000181.JPEG",
                                                    1,
                                                    36)
        predictions, inference_latency = inference.run()
        predictions_lables = get_top_predictions(predictions, False, 5)

        return {'top_predictions': predictions_lables, 'inference_latency': inference_latency}, 200
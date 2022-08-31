from parliament import Context
from flask import Request
import json
from distutils.command.config import config
import time
import tensorflow as tf
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from tensorflow.python.framework import dtypes

import datasets
import numpy as np
import os

MODEL_NAME = 'resnet50'
INPUTS = 'input'
OUTPUTS = 'predict'
RESNET_IMAGE_SIZE = 224

# tf.keras.datasets.

class image_classifier_optimized_graph:
  """Evaluate image classifier with optimized TensorFlow graph"""
  batch_size = -1
  model_name = MODEL_NAME
  input_graph = ""
  data_location = ""
  results_file_path = "" # need define+time
  num_inter_threads = 1 
  num_intra_threads = 36 # physical cores
  data_num_inter_threads = 32
  data_num_intra_threads = 14
  num_cores = 28
  warmup_steps = 10
  steps = warmup_steps + 1

  def __init__(self, 
               batch_size, 
               model_name, 
               input_graph, 
               data_location, 
               warmup_steps,
               steps,
               num_inter_threads=1, 
               num_intra_threads=36):
    self.batch_size = batch_size
    self.model_name = model_name
    self.input_graph = input_graph
    self.data_location = data_location
    self.warmup_steps = warmup_steps
    self.steps = steps
    self.num_inter_threads = num_inter_threads
    self.num_intra_threads = num_intra_threads 
    self.calibrate = False

    self.validate_args()

  def write_results_output(self, predictions, filenames, labels):
    # If a results_file_path is provided, write the predictions to the file
    if self.results_file_path:
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
    
  def run(self):
    """run inference with optimized graph"""
    data_config, infer_config = self.optimize_graph()

    print("Data preprocess")
    data_graph = tf.Graph()
    with data_graph.as_default():
      if (self.data_location):
        print("Inference with real data.")
        if self.calibrate:
            subset = 'calibration'
        else:
            subset = 'validation'
        dataset = datasets.ImagenetData(self.data_location)
        preprocessor = dataset.get_image_preprocessor()(
            RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE, self.batch_size,
            num_cores=self.num_cores,
            resize_method='crop')

        images, labels, filenames = preprocessor.minibatch(dataset, subset=subset)

        # If a results file path is provided, then start the prediction output file
        if self.results_file_path:
          with open(self.results_file_path, "w+") as fp:
            fp.write("filename,actual,prediction\n")
      else:
        print("Inference with dummy data.")
        input_shape = [self.batch_size, RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE, 3]
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

      # log
      print("   # opname:")
      opname = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node] 
      print(opname)

    # Define input and output Tensors for detection_graph
    input_tensor = infer_graph.get_tensor_by_name('input:0')
    output_tensor = infer_graph.get_tensor_by_name('predict:0')

    # log
    print("   input:", input_tensor)
    print("   output:", output_tensor)

    data_sess = tf.compat.v1.Session(graph=data_graph,  config=data_config)
    infer_sess = tf.compat.v1.Session(graph=infer_graph, config=infer_config)

    num_processed_images = 0
    num_remaining_images = dataset.num_examples_per_epoch(subset=subset) - num_processed_images \
        if self.data_location else (self.batch_size * self.steps)

    # warm_up_iteration = self.warmup_steps
    total_time = 0

    tf_filenames = None
    np_labels = None
    data_load_start = time.time()
    if self.results_file_path:
      image_np, np_labels, tf_filenames = data_sess.run([images, labels, filenames])
    else:
      image_np = data_sess.run(images)

    data_load_time = time.time() - data_load_start

    num_processed_images += self.batch_size
    num_remaining_images -= self.batch_size

    start_time = time.time()
    predictions = infer_sess.run(output_tensor, feed_dict={input_tensor: image_np})
    time_consume = time.time() - start_time

    top_predictions = np.argmax(predictions, 1)
    print("top_predictions = ", top_predictions)
    # Write out the file name, expected label, and top prediction
    self.write_results_output(predictions, tf_filenames, np_labels)

    # only add data loading time for real data, not for dummy data
    if self.data_location:
      time_consume += data_load_time

    total_time = time_consume

  def validate_args(self):
    """validate the arguments"""

    if not self.data_location:
      if self.accuracy_only:
        raise ValueError("You must use real data for accuracy measurement.")

# def inference(req: Request) -> str:
#     if req.method == "GET":
#         inference = image_classifier_optimized_graph(1,MODEL_NAME,"resnet50_fp32_pretrained_model.pb","","./",1,False)
#         prediction, inference_latency = inference.run()
#         return {'prediction': prediction, 'inference_latency': inference_latency}
#     elif req.method == "POST":
#         print("request form: ", req.form)
#         print("request url: ", req.form.get('url'))
#         input_url = req.form.get('url')
#         inference = image_classifier_optimized_graph(1,MODEL_NAME,"resnet50_fp32_pretrained_model.pb",input_url,"./",1,False)
#         prediction, inference_latency = inference.run()
#         return {'prediction': prediction, 'inference_latency': inference_latency}
        
def main(context: Context):
    """ 
    Function template
    The context parameter contains the Flask request object and any
    CloudEvent received with the request.
    """

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
                                                    "resnet50_fp32_pretrained_model.pb",
                                                    "",
                                                    20,
                                                    30,
                                                    1,
                                                    36)
        prediction, inference_latency = inference.run()
        print("prediction: ", prediction)
        print("inference_latency: ", inference_latency)
        return {'prediction': prediction, 'inference_latency': inference_latency}, 200
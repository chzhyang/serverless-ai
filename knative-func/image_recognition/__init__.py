# __init__.py
import os
import subprocess
import tensorflow as tf
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from tensorflow.python.framework import dtypes

RESNET_IMAGE_SIZE = 224
INPUTS = 'input'
OUTPUTS = 'predict'
MODEL_PATH = 'models/resnet50_fp32_pretrained_model.pb'

def optimize_config():
  # Get all physical cores
  num_physical_cores = subprocess.getoutput('lscpu -b -p=Core,Socket | grep -v \'^#\' | sort -u | wc -l')
  os.environ["KMP_BLOCKTIME"] = "1"
  os.environ["KMP_SETTINGS"] = "1"
  os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
  os.environ["OMP_NUM_THREADS"]= num_physical_cores
  ### test
  os.environ["ONEDNN_VERBOSE"]="0"
  tf.config.threading.set_inter_op_parallelism_threads(1)
  tf.config.threading.set_intra_op_parallelism_threads(int(num_physical_cores))

def __load_model__(input_graph):
  # Load model
  infer_graph = tf.Graph()
  with infer_graph.as_default():
    graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v1.gfile.FastGFile(input_graph, 'rb') as input_file:
      input_graph_content = input_file.read()
      graph_def.ParseFromString(input_graph_content)
      output_graph = optimize_for_inference(graph_def, [INPUTS], [OUTPUTS], dtypes.float32.as_datatype_enum, False)
    tf.import_graph_def(output_graph, name='')
  # Use random data to cache model
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
  return infer_graph, infer_sess

optimize_config()
infer_graph, infer_sess = __load_model__(MODEL_PATH)
print("##########   Ready for inference   ##########", flush=True)
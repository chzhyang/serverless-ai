import os
import sys
import tensorflow as tf

def optimize_tf_runtime():
  model=sys.argv[2]
  physical_cores_per_socket=sys.argv[3]
  all_physical_cores=sys.argv[4]
  FLAGS = tf.app.flags.FLAGS
  # FLAGS = tf.flags.FLAGS
  
  print("PHYSICAL_CORES_PER_SOCKET:",physical_cores_per_socket)
  print("ALL_PHYSICAL_CORES: ",all_physical_cores)

  if FLAGS.num_intra_threads > 0:
    os.environ["OMP_NUM_THREADS"]= all_physical_cores
  
  if model == "ResNet101":
    tf.config.threading.set_inter_op_parallelism_threads(2) # for ResNet101
  else: # for ResNet50, InceptionV3
    tf.config.threading.set_inter_op_parallelism_threads(1)

  tf.config.threading.set_intra_op_parallelism_threads(all_physical_cores)

def cleanup():
  # TODO
  return

if __name__ == '__main__':
  if sys.argv[1] == "true":
    optimize_tf_runtime()
  elif sys.argv[1] == "false":
    cleanup()

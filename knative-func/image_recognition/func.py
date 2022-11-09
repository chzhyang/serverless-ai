import time
from parliament import Context
from flask import Request, jsonify

import utils

RESNET_IMAGE_SIZE = 224
TEST_IMAGE = './data/test.JPEG'
MODEL_PATH = 'models/resnet50_fp32_pretrained_model.pb'

utils.optimized_config()
INFER_GRAPH, INFER_SESS = utils.load_model(MODEL_PATH)

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
def request_handler(req: Request, infer_graph, infer_sess) -> str:
  """Handle the request"""
  if req.method == "GET":
    start_time = time.time()
    predictions, data_time, infer_time= utils.run_inference(TEST_IMAGE, infer_graph, infer_sess)
    predictions_lables = utils.get_top_predictions(predictions, False, 5)
    total_time = time.time()-start_time
    result = {
      "top5_predictions": predictions_lables, 
      "data_latency(ms)": data_time,
      "inference_latency(ms)": infer_time,
      "total_time(ms)": total_time * 1000
    }
    print(result, flush=True)
    return jsonify(result)
  elif req.method == "POST":
    start_time = time.time()
    data = req.get_json()
    img_url = data["imgURL"]
    img_filepath = utils.download_image(img_url)
    predictions, data_time, infer_time = utils.run_inference(img_filepath, infer_graph, infer_sess)
    total_time = time.time()-start_time
    predictions_lables = utils.get_top_predictions(predictions, False, 5)
    result = {
      "top5_predictions": predictions_lables, 
      "data_latency(ms)": data_time,
      "inference_latency(ms)": infer_time,
      "total_time(ms)": total_time * 1000
    }
    print(result, flush=True)
    return jsonify(result)

def main(context: Context):
  """
  Image classifier with optimized TensorFlow graph
  """
  if 'request' in context.keys():
    return request_handler(context.request, INFER_GRAPH, INFER_SESS)
  else:
    # performance test
    img_url = "https://raw.githubusercontent.com/chzhyang/faas-workloads/main/tensorflow/image_recognition/tensorflow_image_classification/data/ILSVRC2012_test_00000181.JPEG"
    img_filepath = utils.download_image(img_url)
    predictions, latency = utils.run_inference(img_filepath, INFER_GRAPH, INFER_SESS)
    predictions_lables = utils.get_top_predictions(predictions, False, 5)
    result = {
      "top5_predictions" : predictions_lables, 
      "inference_latency(ms)" : latency
    }
    print(result, flush=True)
    # headers = { "content-type": "application/json" }
    return jsonify(result)
    # test end
    # print("Empty request", flush=True)
    # print("Empty request", flush=True)
    # return "{}", 400
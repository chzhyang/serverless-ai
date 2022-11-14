import os
import time
from urllib import request
from parliament import Context
from flask import Request, json
from pathlib import Path
import image_recognition_service

MODEL_NAME = "resnet50_v1_5_fp32.pb"
DATA_DIR = Path(__file__).resolve().parent / 'data'
# A local image used for handling GET request
TEST_IMAGE = os.path.join(DATA_DIR, 'test.JPEG')
MODEL_PATH = os.path.join(DATA_DIR, MODEL_NAME)
# Labels used for mapping inference results to human-readable predictions
LABELS_PATH = os.path.join(DATA_DIR, 'labellist.json')
# Number of top human-readable predictions
NUM_TOP_PREDICTIONS = 5

# Init the image recognition service class
SERVICE = image_recognition_service.ImageRecognitionService(MODEL_PATH)

def download_image(img_url, img_dir):
  """Download the image to target path if it doesn't exist"""
  img_name = img_url.split('/')[-1]
  img_filepath = os.path.join(img_dir, img_name)
  if not os.path.exists(img_filepath):
    if not os.path.exists(img_dir):
      os.makedirs(img_dir)
    img_data = request.get(img_url)
    with open(img_filepath, 'wb') as f:
      f.write(img_data.content)
    print("Download image to ", img_filepath, flush=True)
  else:
    print("Image exists: ", img_filepath)
  return img_filepath

def request_handler(req: Request, svc) -> str:
  """Handle the request"""
  if req.method == "GET":
    # Inference a local image
    predictions, data_time, infer_time= svc.run_inference(TEST_IMAGE, LABELS_PATH, NUM_TOP_PREDICTIONS)
    result = {
      "top_predictions": predictions
    }
    print(result, flush=True)
    return json.dumps(result), 200
  elif req.method == "POST":
    # Inference from a image url in POST request, download the image firstly, then run inference
    data = req.get_json()
    img_url = data["imgURL"]
    img_filepath = download_image(img_url, DATA_DIR)
    predictions, data_time, infer_time = svc.run_inference(img_filepath, LABELS_PATH, NUM_TOP_PREDICTIONS)
    result = {
      "top_predictions": predictions
    }
    print(result, flush=True)
    return json.dumps(result), 200

def main(context: Context):
  """
  Image recognition inference with optimized TensorFlow
  """
  if 'request' in context.keys():
    return request_handler(context.request, SERVICE)
  else:
    print("Empty request", flush=True)
    return "{}", 200
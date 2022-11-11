import os
import time
from urllib import request
from parliament import Context
from flask import Request, jsonify
from pathlib import Path
import image_recognition_service

MODEL_FP32 = "resnet50_fp32_pretrained_model.pb"
MODEL_BF16 = "resnet50_v1_5_bfloat16.pb"
DATA_DIR = Path(__file__).resolve().parent / 'data'
TEST_IMAGE = os.path.join(DATA_DIR, 'test.JPEG')
MODEL_PATH = os.path.join(DATA_DIR, MODEL_BF16)
LABELS_PATH = os.path.join(DATA_DIR, 'labellist.json')
SERVICE = image_recognition_service.ImageRecognitionService(MODEL_PATH, LABELS_PATH)

def download_image(img_url, img_dir):
  """Download image from URL to default filepath"""
  if not os.path.exists(img_dir):
      os.makedirs(img_dir)
  img_name = img_url.split('/')[-1].split('.')[0]+'.jpg'
  img_filepath = os.path.join(img_dir, img_name)
  if not os.path.exists(img_filepath):
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
    start_time = time.time()
    predictions, data_time, infer_time= svc.run_inference(TEST_IMAGE)
    total_time = time.time()-start_time
    result = {
      "top5_predictions": predictions, 
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
    img_filepath = download_image(img_url, DATA_DIR)
    predictions, data_time, infer_time = svc.run_inference(img_filepath)
    total_time = time.time()-start_time
    result = {
      "top5_predictions": predictions, 
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
    return request_handler(context.request, SERVICE)
  else:
    print("Empty request", flush=True)
    return "{}", 400
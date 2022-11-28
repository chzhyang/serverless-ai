import os
from http.client import BAD_REQUEST
from pathlib import Path
import requests
import image_recognition_service
from flask import json
from werkzeug.exceptions import MethodNotAllowed, BadRequest, InternalServerError, HTTPException

PRETRAINED_MODEL = "resnet50_v1_5_fp32.pb"
DATA_DIR = Path(__file__).resolve().parent / 'data'
MODEL_PATH = os.path.join(DATA_DIR, PRETRAINED_MODEL)
# Labels used for mapping inference results to human-readable predictions
LABELS_PATH = os.path.join(DATA_DIR, 'labellist.json')
# A local image used for handling GET request
TEST_IMAGE = os.path.join(DATA_DIR, 'test.JPEG')

# Number of top human-readable predictions
NUM_TOP_PREDICTIONS = 5

# Init the image recognition service class
SERVICE = image_recognition_service.ImageRecognitionService(MODEL_PATH)


def download_image(img_url, img_dir):
    """Download the image to target path if it doesn't exist"""
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    img_name = img_url.split('/')[-1]
    img_filepath = os.path.join(img_dir, img_name)
    if not os.path.exists(img_filepath):
        img_data = requests.get(img_url)
        with open(img_filepath, 'wb') as f:
            f.write(img_data.content)
        print("Download image to ", img_filepath, flush=True)
    else:
        print("Image exists: ", img_filepath)
    return img_filepath


# @functions_framework.http
def handler(request):
    """
    Handle the request.

    Image recognition inference with optimized TensorFlow.

    """

    if request == None:
        print("Empty request", flush=True)
        return "{}", 200
    if request.method == "GET":
        # Inference a local image
        predictions = SERVICE.run_inference(
            TEST_IMAGE, LABELS_PATH, NUM_TOP_PREDICTIONS)
        result = {
            "top_predictions": predictions
        }
        print(result, flush=True)
        return json.dumps(result), 200
    elif request.method == "POST":
        # Download the image, then run inference
        if not request.is_json:
            raise BadRequest(description="only JSON body allowed")
        try:
            data = request.get_json()
            img_url = data["imgURL"]
            img_filepath = download_image(img_url, DATA_DIR)
            predictions = SERVICE.run_inference(
                img_filepath, LABELS_PATH, NUM_TOP_PREDICTIONS)
            result = {
                "top_predictions": predictions
            }
            print(result, flush=True)
            return json.dumps(result), 200
        except KeyError:
            raise BAD_REQUEST(description='missing imgURL in JSON')
        except Exception as e:
            raise InternalServerError(original_exception=e)
    else:
        raise MethodNotAllowed(valid_methods=['GET, POST'])

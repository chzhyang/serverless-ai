import os
import json
from pathlib import Path
import requests
import image_recognition_service
from parliament import Context
import boto3
import ceph_s3

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

# s3 connection
BUCKET_NAME = "fish"
AWS_REGION = "my-store"
EVENT_NAME = "ObjectCreated:Put"

ACCESS_KEY = "Y51LDIT65ZN41VVLKG0H"
SECRET_KEY = "GOxbWKx5NunhlAt3xTvDUh3uHP04A6Cv3UFEwdGS"
ENDPOINT_URL = "http://rook-ceph-rgw-my-store:80"

S3 = ceph_s3.CephS3(ENDPOINT_URL, ACCESS_KEY, SECRET_KEY)

def main(context: Context):
    """
    Image recognition inference with optimized TensorFlow
    """
    if context.cloud_event.data != None:
        data_json = context.cloud_event.data
        data_dict = json.loads(data_json)
        if data_dict["awsRegion"] == AWS_REGION and data_dict["eventName"] = EVENT_NAME:
            object_key = data_dict["s3"]["object"]["key"]
            # TODO: get object from ceph
            file_bytes = S3.download_s3_file(BUCKET_NAME, object_key)
            print(file_bytes.getvalue())
            # TODO: inference from bytes not file
            predictions = svc.run_inference(
                img_filepath, LABELS_PATH, NUM_TOP_PREDICTIONS)
            result = {
                "top_predictions": predictions
            }
            print(result, flush=True)
            return json.dumps(result), 200
    else:
        print("Empty event", flush=True)
        return "{}", 200

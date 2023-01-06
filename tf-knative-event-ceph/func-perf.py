import datetime
import os
import json
from pathlib import Path
import image_recognition_service
from parliament import Context
import boto3
import logging as log
PRETRAINED_MODEL = "resnet50_v1_5_fp32.pb"
DATA_DIR = Path(__file__).resolve().parent / 'data'
MODEL_DIR = Path(__file__).resolve().parent / 'model'
MODEL_PATH = os.path.join(MODEL_DIR, PRETRAINED_MODEL)
# Labels used for mapping inference results to human-readable predictions
LABELS_PATH = os.path.join(MODEL_DIR, 'labellist.json')
# A local image used for handling GET request
TEST_IMAGE = os.path.join(DATA_DIR, 'test.JPEG')

# Number of top human-readable predictions
NUM_TOP_PREDICTIONS = 1

# Init the image recognition service class
SERVICE = image_recognition_service.ImageRecognitionService(MODEL_PATH)

# s3 connection
BUCKET_NAME = "fish"
AWS_REGION = "my-store"
EVENT_NAME = "ObjectCreated:Put"

ACCESS_KEY = "Y51LDIT65ZN41VVLKG0H"
SECRET_KEY = "GOxbWKx5NunhlAt3xTvDUh3uHP04A6Cv3UFEwdGS"
ENDPOINT_URL = "http://10.110.230.223:80"

S3 = boto3.client(
    's3',
    endpoint_url=ENDPOINT_URL,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY)

COUNT = {
    "event_count": 0,
}


def main(context: Context):
    """
    Image recognition inference with optimized TensorFlow
    """
    if context.cloud_event is not None:
        COUNT["event_count"] += 1
        log.info(f'Event number: {COUNT["event_count"]}')
        data_dict = context.cloud_event.data
        if data_dict["awsRegion"] == AWS_REGION and data_dict["eventName"] == EVENT_NAME:
            object_key = data_dict["s3"]["object"]["key"]
            log.info(f'Object key: {object_key}')
            file_path = os.path.join(DATA_DIR, str(object_key))
            # cover file if existed
            try:
                start = datetime.now()
                S3.download_file(BUCKET_NAME, object_key, file_path)
                end = datetime.now()
                log.info(
                    f'S3 download start: {start}, end: {end}, cost time: {(end-start).microseconds/1000} ms')
            except Exception as e:
                resp = "Failed to download file from s3: " + e
                log.error(resp)
                return json.dumps(resp), 400
            try:
                start = datetime.now()
                predictions = SERVICE.run_inference(
                    file_path, LABELS_PATH, NUM_TOP_PREDICTIONS)
                end = datetime.now()
                log.info(
                    f'Infer start: {start}, end: {end}, cost time: {(end-start).microseconds/1000} ms')
                resp = {
                    "top_prediction": predictions
                }
                log.info(resp)
                return json.dumps(resp), 200
            except Exception as e:
                resp = "Failed to inference: " + e
                log.error(resp)
                return json.dumps(resp), 400
        else:
            resp = "Unexpected event name/aws region"
            log.error(resp)
            return json.dumps(resp), 400
    else:
        log.error("Empty event")
        return "{}", 400

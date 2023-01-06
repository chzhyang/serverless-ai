import datetime
import imghdr
import os
import json
from pathlib import Path
import image_recognition_service
from parliament import Context
import boto3

PRETRAINED_MODEL = "resnet50_v1_5_fp32.pb"
DATA_DIR = Path(__file__).resolve().parent / 'data'
MODEL_PATH = os.path.join(DATA_DIR, PRETRAINED_MODEL)
# Labels used for mapping inference results to human-readable predictions
LABELS_PATH = os.path.join(DATA_DIR, 'labellist.json')
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


def inference(svc, file_path):
    print("Inference ", file_path, flush=True)
    start = datetime.now()
    print("Start infer: ", start, flush=True)
    predictions = svc.run_inference(
        file_path, LABELS_PATH, NUM_TOP_PREDICTIONS)
    end = datetime.now()
    print("End infer: ", end, flush=True)
    print("Infer time(ms): ", (end-start).microseconds/1000, flush=True)
    result = {
        "top_predictions": predictions
    }
    print(result, flush=True)
    return result


def main(context: Context):
    """
    Image recognition inference with optimized TensorFlow
    """
    if context.cloud_event is not None:
        COUNT["event_count"] += 1
        print("Event number: ", COUNT["event_count"], flush=True)
        data_dict = context.cloud_event.data
        if data_dict["awsRegion"] == AWS_REGION and data_dict["eventName"] == EVENT_NAME:
            object_key = data_dict["s3"]["object"]["key"]
            print("Object key: ", object_key)
            file_path = os.path.join(DATA_DIR, str(object_key))
            if not os.path.exists(file_path):
                print("Download file from s3", flush=True)
                try:
                    S3.download_file(BUCKET_NAME, object_key, file_path)
                except Exception as e:
                    resp = "Failed to download file from s3: " + e
                    print(resp, flush=True)
                    # return resp, 400
            resp = inference(SERVICE, file_path)
            # return json.dumps(resp), 200
        else:
            resp = "Unexpected event name/aws region"
            print(resp, flush=True)
            # return resp, 400
    else:
        print("Empty event", flush=True)
        # return "{}", 400

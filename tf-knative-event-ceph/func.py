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

# event count
EVENT_COUNT = 0


def main(context: Context):
    """
    Image recognition inference with optimized TensorFlow
    """
    if context.cloud_event.data != None:
        EVENT_COUNT = EVENT_COUNT + 1
        print("Event number: ", EVENT_COUNT, flush=True)
        data_json = context.cloud_event.data
        # print(data_json, flush=True)
        data_dict = json.loads(data_json)
        if data_dict["awsRegion"] == AWS_REGION and data_dict["eventName"] == EVENT_NAME:
            object_key = data_dict["s3"]["object"]["key"]
            print("Object key: ", object_key)
            file_path = Path(__file__).resolve().parent / 'data' / object_key
            if not os.path.exists(file_path):
                print("Download file from s3", flush=True)
                S3.download_file(BUCKET_NAME, object_key, file_path)
                if not os.path.exists(file_path):
                    print("Failed to download ", file_path)
                else:
                    print("Inference ", file_path, flush=True)
                    predictions = SERVICE.run_inference(
                        file_path, LABELS_PATH, NUM_TOP_PREDICTIONS)
                    result = {
                        "top_predictions": predictions
                    }
                    print(result, flush=True)
                    return json.dumps(result), 200
    else:
        print("Empty event", flush=True)
        return "{}", 400

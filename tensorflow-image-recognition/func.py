from datetime import datetime
import os
import json
from pathlib import Path

import image_recognition_service
from parliament.parliament import Context
import boto3
import logging as log

PRETRAINED_MODEL = "resnet50_v1_5_fp32.pb"
DATA_DIR = Path(__file__).resolve().parent / 'data'
MODEL_DIR = Path(__file__).resolve().parent / 'model'
MODEL_PATH = os.path.join(MODEL_DIR, PRETRAINED_MODEL)
# Labels used for mapping inference results to human-readable predictions
LABELS_PATH = os.path.join(MODEL_DIR, 'labellist.json')
# A local image used for handling GET request
TEST_IMAGE = os.path.join(DATA_DIR, 'test1.JPEG')

# Number of top human-readable predictions
NUM_TOP_PREDICTIONS = 1

# Init the image recognition service class
SERVICE = image_recognition_service.ImageRecognitionService(MODEL_PATH)

# s3 connection
# BUCKET_NAME = "fish"
# AWS_REGION = "my-store"
# EVENT_NAME = "ObjectCreated:Put"

# ACCESS_KEY = "Y51LDIT65ZN41VVLKG0H"
# SECRET_KEY = "GOxbWKx5NunhlAt3xTvDUh3uHP04A6Cv3UFEwdGS"
# ENDPOINT_URL = "http://10.110.230.223:80"


INIT_LIST = {}

COUNT = {
    "cloudevent_count": 0,
    "GET_count": 0
}

log.basicConfig(
    level=log.INFO,
    format='%(asctime)s::%(levelname)s::%(message)s',
)


def init():
    """
    variables with uniquess will be initalized after loading snapshot, before inference service
    """
    if os.environ.get('S3_ENABLED', 'false') == "true":
        log.info(f'Initialize Ceph connection(S3)')
        s3_sess = boto3.client(
            's3',
            endpoint_url=os.environ.get('S3_ENDPOINT_URL'),
            aws_access_key_id=os.environ.get('S3_ACCESS_KEY'),
            aws_secret_access_key=os.environ.get('S3_SECRET_KEY'))
        INIT_LIST["s3"] = s3_sess
        INIT_LIST["aws_region"] = os.environ.get('S3_AWS_REGION')
        INIT_LIST["event_name"] = os.environ.get('S3_EVENT_NAME')
        INIT_LIST["bucket_name"] = os.environ.get('S3_BUCKET_NAME')


def inference(file_path, labels_path, top_prediction):
    try:
        infer_start = datetime.now()
        predictions = SERVICE.run_inference(
            file_path, labels_path, top_prediction)
        infer_end = datetime.now()
        log.info(
            f'Run infer time: {(infer_end-infer_start).microseconds/1000} ms')
        # log.info(
        #     f'Total time(s3+infer): {(infer_end-s3_start).microseconds/1000} ms')
        resp = {
            "top_prediction": predictions
        }
        log.info(resp)
        return json.dumps(resp), 200
    except Exception as e:
        resp = "Failed to inference: " + str(e)
        log.error(resp)
        return json.dumps(resp), 400


def main(context: Context):
    """
    Image recognition inference with optimized TensorFlow
    """
    if context is None:
        log.info("None context")
        return "{None context}", 400
    # if context.request is not None:
    #     log.info(f"context.request: {context.request.data}")
    #     # return "{get context.request}", 200

    if context.cloud_event is not None:
        COUNT["cloudevent_count"] += 1
        log.info(f'CloudEvent number: {COUNT["cloudevent_count"]}')
        if os.environ.get('S3_ENABLED', 'false') != "true":
            resp = "Ceph connection is not enabled, please config Ceph connection in ENV\n"
            log.error(resp)
            return json.dumps(resp), 400
        data_dict = context.cloud_event.data

        if data_dict["awsRegion"] == INIT_LIST["aws_region"] and data_dict["eventName"] == INIT_LIST["event_name"]:
            object_key = data_dict["s3"]["object"]["key"]
            log.info(f'Object key: {object_key}')
            file_path = os.path.join(DATA_DIR, str(object_key))
            # cover file if existed
            try:
                s3_start = datetime.now()
                INIT_LIST["s3"].download_file(
                    INIT_LIST["bucket_name"], object_key, file_path)
                s3_end = datetime.now()
                log.info(
                    f'S3 download time: {(s3_end-s3_start).microseconds/1000} ms')
            except Exception as e:  # todo: set e into resp
                resp = "Failed to download file from s3"
                log.error(resp)
                return json.dumps(resp), 400
            return inference(file_path, LABELS_PATH, NUM_TOP_PREDICTIONS)
        else:
            resp = "Unexpected event name/aws region"
            log.error(resp)
            return json.dumps(resp), 400
    # else:
    #     log.error("Empty event")
    #     return "{Empty event}", 400

    # request is not cloudevent type
    elif context.request is not None:
        # reuqest example: curl http://localhost:8080?imageName=test1
        if context.request.method == "GET":
            COUNT["GET_count"] += 1
            log.info(f'GET request number: {COUNT["GET_count"]}')
            img_name = context.request.args.get("imageName", default="test1")
            img_filepath = os.path.join(DATA_DIR, img_name + ".JPEG")
            if os.path.exists(img_filepath) is False:
                log.info("Image file is not exist, use default image!")
                img_filepath = os.path.join(DATA_DIR, TEST_IMAGE)
                if os.path.exists(img_filepath) is False:
                    resp = img_filepath + " and default image are not exist."
                    log.error(resp)
                    return json.dumps(resp), 400
            return inference(img_filepath, LABELS_PATH, NUM_TOP_PREDICTIONS)
        else:
            resp = "Server just supports GET and CloudEvent(POST) requeset now."
            log.error(resp)
            return json.dumps(resp), 400

    else:
        log.error("Empty request")
        return "{Empty request}", 400

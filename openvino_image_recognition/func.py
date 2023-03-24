#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging as log
import os
from pathlib import Path
import subprocess
import sys
import time
import boto3

import cv2
import numpy as np
from openvino.preprocess import PrePostProcessor, ResizeAlgorithm
from openvino.runtime import Core, Layout, Type
from parliament.parliament import Context

MODEL_NAME = "resnet-50-pytorch"
PRECISION = "FP32"
DATA_DIR = Path(__file__).resolve().parent / 'data'
MODEL_DIR = Path(__file__).resolve().parent / 'model'
MODEL_PATH = os.path.join(MODEL_DIR, PRECISION, "resnet-50-pytorch.xml")
# Labels used for mapping inference results to human-readable predictions
LABELS_PATH = os.path.join(MODEL_DIR, 'imagenet2012.json')
# A local image used for handling GET request
TEST_IMAGE = os.path.join(DATA_DIR, 'test1.jpg')
NUM_TOP_PREDICTIONS = 5


COUNT = {
    "cloudevent_count": 0,
    "GET_count": 0
}
INIT_LIST = {}

log.basicConfig(format='[ %(levelname)s ] %(message)s',
                level=log.INFO, stream=sys.stdout)


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


def get_labels(lables_path):
    """Get the set of possible labels for classification"""
    with open(lables_path, "r") as labels_file:
        labels = json.load(labels_file)

    return labels


def top_predictions(result, n):
    """Get the top n predictions given the array of softmax results"""
    # Only care about the first example
    probabilities = result
    # Get the ids of most probable labels. Reverse order to get greatest first
    ids = np.argsort(probabilities)[::-1]
    return ids[:n]


def get_labels_for_ids(labels, ids, ids_are_one_indexed=False):
    """Get the human-readable labels for given ids"""
    return [labels[str(x + int(ids_are_one_indexed))] for x in ids]


def get_top_predictions(results, lables_path, ids_are_one_indexed=False, preds_to_print=5):
    """Given an array of mode, graph_name, predicted_ID, print labels"""
    labels = get_labels(lables_path)
    pred_labels = []
    pred_labels = get_labels_for_ids(
        labels, results, ids_are_one_indexed)

    return pred_labels[:preds_to_print]


class ImageRecognitionService():
    def __init__(self, model_path, model_download, model_name, model_dir):
        self.model_path = model_path

        # Initialize OpenVINO Runtime Core
        log.info('Creating OpenVINO Runtime Core')
        core = Core()

        if model_download and os.path.exists(model_path) is False:
            model_path = self._download_convert_model(model_name, model_dir)
        # Read a model
        log.info(f'Reading the model: {model_path}')
        # (.xml and .bin files) or (.onnx file)
        model = core.read_model(model_path)

        if len(model.inputs) != 1:
            log.error('Sample supports only single input topologies')
            return -1

        # Apply preprocessing
        ppp = PrePostProcessor(model)

        # 1) Set input tensor information:
        # - input() provides information about a single model input
        # - precision of tensor is supposed to be 'u8'
        # - layout of data is 'NHWC'
        ppp.input().tensor() \
            .set_element_type(Type.u8) \
            .set_layout(Layout('NHWC'))  # noqa: N400

        # 2) Here we suppose model has 'NCHW' layout for input
        ppp.input().model().set_layout(Layout('NCHW'))

        # 3) Set output tensor information:
        # - precision of tensor is supposed to be 'f32'
        ppp.output().tensor().set_element_type(Type.f32)

        # 4) Apply preprocessing modifing the original 'model'
        model = ppp.build()

        # Loading model to the device
        log.info('Loading the model to the plugin')
        self.compiled_model = core.compile_model(model, "CPU")

        # Warm up inference
        log.info('Warm up inference')
        # Read input image
        image = cv2.imread(TEST_IMAGE)
        # resize for resnet50
        w = 224
        h = 224
        resized_image = cv2.resize(image, (w, h))
        # Add N dimension
        test_input_tensor = np.expand_dims(resized_image, 0)
        self.compiled_model.infer_new_request({0: test_input_tensor})

    def _download_convert_model(self, model_name, model_dir):
        # Download model
        log.info('Download model')
        cmd = f"omz_downloader --name {model_name} --output_dir {model_dir}"
        # Execute the command, print the log output by the command, and judge whether the command is successful
        p = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        print('omz_downloader output: ', stdout.decode("utf-8"))
        if p.returncode != 0:
            log.error('omz_downloader error', stderr.decode("utf-8"))
            return

        # Convert model
        log.info(f'Convert model to IR')
        cmd = f"omz_converter --name {MODEL_NAME} --output_dir {MODEL_DIR} --download_dir {model_dir} --precisions {PRECISION}"
        p = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        print('omz_converter output: ', stdout.decode("utf-8"))
        if p.returncode != 0:
            log.error('omz_converter error', stderr.decode("utf-8"))
            return
        model_path = os.path.join(
            MODEL_DIR, "public", MODEL_NAME, PRECISION, MODEL_NAME + ".xml")
        if os.path.exists(model_path) is False:
            log.error('model download and convert error')
            return
        log.info(f'model path: {model_path}')
        return model_path

    def run_inference(self, img_filepath, label_path, num_top_predictions):
        log.info('Starting inference in synchronous mode')
        # Read input image
        image = cv2.imread(img_filepath)
        # resize for resnet50
        w = 224
        h = 224
        resized_image = cv2.resize(image, (w, h))
        # Add N dimension
        input_tensor = np.expand_dims(resized_image, 0)
        results = self.compiled_model.infer_new_request({0: input_tensor})
        # Process output
        predictions = next(iter(results.values()))

        # Change a shape of a numpy.ndarray with results to get another one with one dimension
        probs = predictions.reshape(-1)

        # Get an array of 10 class IDs in descending order of probability
        top_predictions = np.argsort(probs)[-num_top_predictions:][::-1]

        # Get lables of the predictions from class id
        predictions_lable = get_top_predictions(
            top_predictions, label_path, True, 10)

        header = 'class_id probability label'
        log.info(f'Image path: {img_filepath}')
        log.info(f'Top {num_top_predictions} results: ')
        log.info(header)
        log.info('-' * len(header))
        for i in range(len(top_predictions)):
            class_id = top_predictions[i]
            log.info(
                f'{class_id:6}{probs[class_id]:12.7f}    {predictions_lable[i]}')

        result = {"top_prediction": predictions_lable[0]}
        return result


service = ImageRecognitionService(MODEL_PATH, True, MODEL_NAME, MODEL_DIR)


def main(context: Context):
    """
    Image recognition inference with OpenVINO
    """

    if context is None:
        log.info("None context")
        return "{None context}", 400
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
            img_filepath = os.path.join(DATA_DIR, str(object_key))
            # cover file if existed
            try:
                s3_start = time.perf_counter()
                INIT_LIST["s3"].download_file(
                    INIT_LIST["bucket_name"], object_key, img_filepath)
                s3_end = time.perf_counter()
                log.info(
                    f'S3 download time: {s3_end-s3_start} s')
            except Exception as e:
                resp = "Failed to download file from s3"
                log.error(e)
                return json.dumps(resp), 400
            return json.dumps(service.run_inference(img_filepath, LABELS_PATH, NUM_TOP_PREDICTIONS)), 200
        else:
            resp = "Unexpected event name/aws region"
            log.error(resp)
            return json.dumps(resp), 400
    elif context.request is not None:
        # reuqest example: curl http://localhost:8080?imageName=test1
        if context.request.method == "GET":
            img_name = context.request.args.get("imageName", default="test1")
            img_filepath = os.path.join(DATA_DIR, img_name + ".jpg")
            if os.path.exists(img_filepath) is False:
                log.info("Image file is not exist, use default image!")
                img_filepath = TEST_IMAGE
                if os.path.exists(img_filepath) is False:
                    resp = img_filepath + " and default image are not exist."
                    log.error(resp)
                    return json.dumps(resp), 400
            return json.dumps(service.run_inference(img_filepath, LABELS_PATH, NUM_TOP_PREDICTIONS)), 200
        else:
            resp = "Server just supports requeset of HTTP GET or Cloudevent.events now"
            log.error(resp)
            return json.dumps(resp), 400
    else:
        log.error("Empty request")
        return "{Empty request}", 400

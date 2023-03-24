#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging as log
import os
from pathlib import Path
import sys
import time
import boto3

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from parliament.parliament import Context

MODEL_NAME = "resnet-50-pytorch"
PRECISION = "FP32"
DATA_DIR = Path(__file__).resolve().parent / 'data'
MODEL_DIR = Path(__file__).resolve().parent / 'model'
MODEL_PATH = os.path.join(MODEL_DIR, PRECISION, "resnet-50-pytorch.pth")
# Labels used for mapping inference results to human-readable predictions
LABELS_PATH = os.path.join(MODEL_DIR, 'imagenet2012.json')
# A local image used for handling GET request
TEST_IMAGE = os.path.join(DATA_DIR, 'test1.jpg')
NUM_TOP_PREDICTIONS = 1


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
    def __init__(self, model_path, model_download, intel_optimize, jit):
        # Load model
        if model_download:
            log.info("Download and load model...")
            self.model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        elif model_path is not None:
            log.info("Load model from path: {}".format(model_path))
            self.model = torch.load(model_path)
        self.model.eval()

        # Optimize model with intel extension for pytorch
        if intel_optimize:
            import intel_extension_for_pytorch as ipex
            log.info('ipex:', ipex.__version__)
            self.model = ipex.optimize(self.model)

        # Preprocess
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])

        # Warmup
        with torch.no_grad():
            for i in range(10):
                data = torch.rand(1, 3, 224, 224)
                if jit:
                    self.model = torch.jit.script(self.model, data)
                    self.model = torch.jit.freeze(self.model)
                self.model(data)

    def run_inference(self, image_path, label_path, num_top_predictions, jit=False):
        log.info('Starting inference...')
        with torch.no_grad():
            # Load image
            img = Image.open(image_path).convert('RGB')
            img_preprocessed = self.preprocess(img)
            input_tensor = torch.unsqueeze(img_preprocessed, 0)

            if jit:
                model = torch.jit.script(model, input_tensor)
                model = torch.jit.freeze(model)

            # Inference
            result_torch = self.model(torch.as_tensor(input_tensor).float())
            _, preds = torch.max(result_torch, 1)
            predictions_lable = get_top_predictions(
                preds.cpu().numpy(), label_path, True, num_top_predictions)
            result = {"top_prediction": predictions_lable[0]}

            log.info(result)
            return result


service = ImageRecognitionService(MODEL_PATH, True, False, False)


def main(context: Context):
    """
    Image recognition inference with PyTorch
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

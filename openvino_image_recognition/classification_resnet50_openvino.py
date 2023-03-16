#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import logging as log
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
from openvino.preprocess import PrePostProcessor, ResizeAlgorithm
from openvino.runtime import Core, Layout, Type
from openvino.tools.mo import convert_model

test_image_path = "./banana.jpg"
model_name = "resnet-50-pytorch"


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


def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

    # Parsing and validation of input arguments
    if len(sys.argv) != 9:
        log.info(f'Usage: {sys.argv[0]} <path_to_model> <path_to_image> <device_name> <label_path>')
        return 1

    model_path = sys.argv[1]
    image_path = sys.argv[2]
    device_name = sys.argv[3]
    label_path = sys.argv[4]
    iterations = sys.argv[5]
    model_download = sys.argv[6]
    model_name = sys.argv[7]  # "resnet-50-pytorch"
    data_type = sys.argv[8]  # "FP32"
    model_dir = os.path.dirname(os.path.realpath('__file__'))

# --------------------------- Step 1. Initialize OpenVINO Runtime Core ------------------------------------------------
    log.info('Creating OpenVINO Runtime Core')

    t_start = time.perf_counter()
    core = Core()
    t_runtime = time.perf_counter()
    t_download = 0.0
    t_convert = 0.0
    if os.path.exists(model_path) is False or model_download == "true":
        # Download model
        log.info(f'Download model')
        cmd = f"omz_downloader --name {model_name} --output_dir {model_dir}"
        # Execute the command, print the log output by the command, and judge whether the command is successful
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        print('omz_downloader output: ', stdout.decode("utf-8"))
        if p.returncode != 0:
            log.error('omz_downloader error', stderr.decode("utf-8"))
            return
        t_download = time.perf_counter()
        # Convert model
        log.info(f'Convert model to IR')
        cmd = f"omz_converter --name {model_name} --output_dir {model_dir} --precisions {data_type}"
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        print('omz_converter output: ', stdout.decode("utf-8"))
        if p.returncode != 0:
            log.error('omz_converter error', stderr.decode("utf-8"))
            return
        model_path = os.path.join(model_dir, "public", model_name, data_type, model_name + ".xml")
        if os.path.exists(model_path) is False:
            log.error('model download and convert error')
            return
        log.info(f'model path: {model_path}')
        t_convert = time.perf_counter()


# --------------------------- Step 2. Read a model --------------------------------------------------------------------
    log.info(f'Reading the model: {model_path}')
    # (.xml and .bin files) or (.onnx file)
    model = core.read_model(model_path)

    if len(model.inputs) != 1:
        log.error('Sample supports only single input topologies')
        return -1

    if len(model.outputs) != 1:
        log.error('Sample supports only single output topologies')
        return -1
    t_readmodel = time.perf_counter()

# --------------------------- Step 3. Set up input --------------------------------------------------------------------
    # # Read input image
    # image = cv2.imread(image_path)
    # # resize for resnet50
    # w = 224
    # h = 224
    # resized_image = cv2.resize(image, (w, h))
    # # Add N dimension
    # input_tensor = np.expand_dims(resized_image, 0)

    # t_3 = time.perf_counter()

# --------------------------- Step 4. Apply preprocessing ------------------------------------------------------------

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

    t_ppp = time.perf_counter()

# --------------------------- Step 5. Loading model to the device -----------------------------------------------------
    log.info('Loading the model to the plugin')
    compiled_model = core.compile_model(model, device_name)

    t_loadmodel = time.perf_counter()

# --------------------------- Step 6. warm inference -------------------------------------------------------------------
    log.info('Warm up inference')
    # Read input image
    image = cv2.imread(test_image_path)
    # resize for resnet50
    w = 224
    h = 224
    resized_image = cv2.resize(image, (w, h))
    # Add N dimension
    test_input_tensor = np.expand_dims(resized_image, 0)
    results = compiled_model.infer_new_request({0: test_input_tensor})

    t_warmup = time.perf_counter()

# --------------------------- Step 7. Create infer request and do inference synchronously -----------------------------
    log.info('Starting inference in synchronous mode')
    for i in range(int(iterations)):
        t_infer_start = time.perf_counter()
        # Read input image
        image = cv2.imread(image_path)
        # resize for resnet50
        w = 224
        h = 224
        resized_image = cv2.resize(image, (w, h))
        # Add N dimension
        input_tensor = np.expand_dims(resized_image, 0)

        results = compiled_model.infer_new_request({0: input_tensor})

        t_infer_end = time.perf_counter() - t_infer_start
        # log.info(f'iteration {i}, infer time: {t_infer_end}')

    t_infer = time.perf_counter()
# --------------------------- Step 8. Process output ------------------------------------------------------------------
    predictions = next(iter(results.values()))

    # Change a shape of a numpy.ndarray with results to get another one with one dimension
    probs = predictions.reshape(-1)

    # Get an array of 10 class IDs in descending order of probability
    top_10 = np.argsort(probs)[-10:][::-1]

    # Get lables of the predictions from class id
    predictions_lable = get_top_predictions(top_10, label_path, True, 10)

    header = 'class_id probability label'

    log.info(f'Image path: {image_path}')
    log.info('Top 10 results: ')
    log.info(header)
    log.info('-' * len(header))
    for i in range(len(top_10)):
        class_id = top_10[i]
        # probability_indent = ' ' * (len('class_id') - len(str(class_id)) + 1)
        log.info(f'{class_id:6}{probs[class_id]:12.7f}    {predictions_lable[i]}')
    # for class_id in top_10:
    #     probability_indent = ' ' * (len('class_id') - len(str(class_id)) + 1)
    #     log.info(f'{class_id}{probability_indent}{probs[class_id]:.7f}')

    log.info('')

    log.info(f'top1: {predictions_lable[0]}')

    t_result = time.perf_counter()

# ----------------------------------------------------------------------------------------------------------------------
    log.info('')
    log.info("t_total_coldstart        {:.6f} sec".format(t_warmup - t_start))
    log.info("  t_runtime              {:.6f} sec, {:.0%}".format(t_runtime - t_start, (t_runtime - t_start) / (t_warmup - t_start)))

    if t_convert == 0.0 and t_download == 0.0:
        log.info("  t_readmodel            {:.6f} sec, {:.0%}".format(t_readmodel - t_runtime, (t_readmodel - t_runtime) / (t_warmup - t_start)))
    else:
        log.info("  t_download             {:.6f} sec, {:.0%}".format(t_download - t_runtime, (t_download - t_runtime) / (t_warmup - t_start)))
        log.info("  t_convert              {:.6f} sec, {:.0%}".format(t_convert - t_download, (t_convert - t_download) / (t_warmup - t_start)))
        log.info("  t_readmodel            {:.6f} sec, {:.0%}".format(t_readmodel - t_convert, (t_readmodel - t_convert) / (t_warmup - t_start)))

    log.info("  t_ppp                  {:.6f} sec, {:.0%}".format(t_ppp - t_readmodel, (t_ppp - t_readmodel) / (t_warmup - t_start)))
    log.info("  t_loadmodel            {:.6f} sec, {:.0%}".format(t_loadmodel - t_ppp, (t_loadmodel - t_ppp) / (t_warmup - t_start)))
    log.info("  t_warmup               {:.6f} sec, {:.0%}".format(t_warmup - t_loadmodel, (t_warmup - t_loadmodel) / (t_warmup - t_start)))
    log.info("t_total_infer            {:.6f} sec".format(t_infer - t_warmup))
    log.info("  t_avg_infer            {:.6f} sec".format((t_infer - t_warmup) / float(iterations)))
    log.info("t_result                 {:.6f} sec".format(t_result - t_infer))

    return 0


if __name__ == '__main__':
    sys.exit(main())

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
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
test_image_path = "./banana.jpg"
model_name = "resnet-50-pytorch"

# def convert_model(origin_model_path, output_model_dir):
#     model_name = "saved_model"
#     cmd = "mo --framework tf --saved_model_dir " + origin_model_path + " --output_dir " + output_model_dir + " --model_name " + model_name
#     return output_model_dir + "/" + model_name
# omz_converter - -name resnet-50-pytorch - v2 - pytorch - -precisions FP16 - -download_dir model - -output_dir model
# if a file exist


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
    log.basicConfig(format='[ %(levelname)s ] %(message)s',
                    level=log.INFO, stream=sys.stdout)
    log.info('torch:', torch.__version__)
#     # Parsing and validation of input arguments
#     if len(sys.argv) != 9:
#         log.info(f'Usage: {sys.argv[0]} <path_to_model> <path_to_image> <device_name> <label_path>')
#         return 1
    model_download = sys.argv[1]
    image_path = sys.argv[2]
    iterations = sys.argv[3]
    optimize = sys.argv[4]  # "False"
    jit = sys.argv[5]  # "False"
    benchmark = sys.argv[6]  # "True"
    model_path = sys.argv[7]  # local model path
    precision = sys.argv[8]  # "FP32"
    label_path = sys.argv[9]
    device_name = sys.argv[10]

    times = []
    sum = 0.0
    # Load model
    t_start = time.perf_counter()
    # use online model
    if model_download == "True":
        model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    elif model_path is not None:
        model = torch.load(model_path)
    model.eval()
    t_model = time.perf_counter()

    # Optimize model with intel extension for pytorch
    if optimize == True:
        import intel_extension_for_pytorch as ipex
        log.info('ipex:', ipex.__version__)
        model = ipex.optimize(model)
        t_optimize = time.perf_counter()
    # preprocess
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])
    # benchmark
    if benchmark == "True":
        with torch.no_grad():
            for i in range(iterations):
                t_1 = time.perf_counter()
                img = Image.open("./test.jpg").convert('RGB')
                img_preprocessed = preprocess(img)
                input_tensor = torch.unsqueeze(img_preprocessed, 0)

                # open JIT
                if jit == "True":
                    model = torch.jit.script(model, input_tensor)
                    model = torch.jit.freeze(model)
                    t_jit = time.perf_counter()

                result_torch = model(torch.as_tensor(input_tensor).float())
                t_2 = time.perf_counter()
                print(f'iteration {i}, infer time: {t_2-t_1} sec')
                times.append(t_2 - t_1)
                if i > 0:
                    sum = sum + t_2 - t_1
        log.info('')
        log.info("t_model        {:.6f} sec".format(t_model - t_start))
        if optimize == "True":
            log.info("t_optimize     {:.6f} sec".format(
                t_optimize - t_model))
        log.info("t_first_infer                 {:.6f} sec".format(times[0]))
        log.info("t_avg_infer                 {:.6f} sec".format(
            sum / float(iterations - 1)))
    # inference
    else:
        with torch.no_grad():
            # warmup for 10 times
            for i in range(10):
                data = torch.rand(1, 3, 224, 224)
                # model = torch.jit.trace(model, d)
                # model = torch.jit.freeze(model)
                model(data)
            # inference
            img = Image.open(image_path).convert('RGB')
            img_preprocessed = preprocess(img)
            input_tensor = torch.unsqueeze(img_preprocessed, 0)

            # open JIT
            if jit == True:
                model = torch.jit.script(model, input_tensor)
                model = torch.jit.freeze(model)

            result_torch = model(torch.as_tensor(input_tensor).float())
            result_mask_torch = torch.argmax(
                result_torch['out'], dim=1).squeeze(0).numpy().astype(np.uint8)

            # Change a shape of a numpy.ndarray with results to get another one with one dimension
            print("result_mask_torch: ", result_mask_torch)
            # log.info(f'Image path: {image_path}')
            # log.info('Top 10 results: ')
            # header = 'class_id probability label'
            # log.info(header)
            # log.info('-' * len(header))
            # for i in range(len(top_10)):
            #     class_id = top_10[i]
            #     # probability_indent = ' ' * (len('class_id') - len(str(class_id)) + 1)
            #     log.info(
            #         f'{class_id:6}{probs[class_id]:12.7f}    {predictions_lable[i]}')
            # for class_id in top_10:
            #     probability_indent = ' ' * (len('class_id') - len(str(class_id)) + 1)
            #     log.info(f'{class_id}{probability_indent}{probs[class_id]:.7f}')

            # log.info(f'top1: {predictions_lable[0]}')

    return 0


if __name__ == '__main__':
    sys.exit(main())

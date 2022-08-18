#!/usr/bin/env bash
#
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Benchmark Configuration
FRAMEWORK="tensorflow"
MODEL="resnet50"
PRECISION="fp32"
MODE="inference"
BATCH_SIZE_LATENCY=1
SOCKET_ID=0

INITIAL_DIR=`pwd`
MODELS_DIR="${INITIAL_DIR}/models"
BENCHMARK_PATH="${MODELS_DIR}/benchmarks/launch_benchmark.py"
SCRIPT_DIR="${INITIAL_DIR}/script"
RUNTIME_OPT_PATH="${SCRIPT_DIR}/tf_runtime_optimize.py"

echo "INITIAL_DIR: ${INITIAL_DIR}"
echo "BENCHMARK_DIR: ${BENCHMARK_DIR}"
echo "SCRIPT_DIR: ${SCRIPT_DIR}"
echo "RUNTIME_OPT_PATH: ${RUNTIME_OPT_PATH}"
echo "MODELS_DIR: ${MODELS_DIR}"

echo "Download ${MODEL}_${PRECISION}_pretrained_model..."
MODEL_BUCKET="https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8"
MODEL_URL="${MODEL_BUCKET}/${MODEL}_${PRECISION}_pretrained_model.pb"
wget -P ${INITIAL_DIR}/pretrained_models ${MODEL_URL}
MODEL_PATH="${INITIAL_DIR}/pretrained_models/${MODEL}_${PRECISION}_pretrained_model.pb"
echo "Download success, MODEL_PATH: ${MODEL_PATH}"

# TODO
# FRAMEWORK_VERSION=`pip list | grep tensorflow`
# echo "FRAMEWORK_VERSION: ${FRAMEWORK_VERSION}"

# Run latency benchmarking with batch size 1
echo "Run inference #latency# benchmark on ${MODEL}_${PRECISION}_pretrained_model with ORIGINAL and OPTIMIZED runtime setting"

echo "1. Benchmark with ORIGINAL runtime setting"
# unset OMP_NUM_THREADS
# unset KMP_AFFINITY
# unset KMP_BLOCKTIME
# unset KMP_SETTINGS
echo "1.1 Run benchmark"
# Synthetic data
cd ${MODELS_DIR}
python ${BENCHMARK_PATH} \
    --in-graph ${MODEL_PATH} \
    --model-name ${MODEL} \
    --framework ${FRAMEWORK} \
    --precision ${PRECISION} \
    --mode ${MODE} \
    --batch-size=${BATCH_SIZE_LATENCY} \
    --socket-id ${SOCKET_ID}
    --docker-image intel/intel-optimized-tensorflow:latest \

cd ${INITIAL_DIR}

echo "2. Benchmark with OPTIMIZED runtime setting"
echo "2.1 Optimize runtime setting"
PHYSICAL_CORES_PER_SOCKET=`lscpu | grep "Core(s) per socket" | cut -d':' -f2 | xargs`
ALL_PHYSICAL_CORES=`lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`
export OMP_NUM_THREADS=$(ALL_PHYSICAL_CORES)
export KMP_AFFINITY="granularity=fine,verbose,compact,1,0"
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
python ${RUNTIME_OPT_PATH} true ${MODEL} ${PHYSICAL_CORES_PER_SOCKET} ${ALL_PHYSICAL_CORES}

echo "2.2 Run benchmark"
cd ${MODELS_DIR}
python ${BENCHMARK_PATH} \
    --in-graph ${MODEL_PATH} \
    --model-name ${MODEL} \
    --framework ${FRAMEWORK} \
    --precision ${PRECISION} \
    --mode ${MODE} \
    --batch-size=${BATCH_SIZE_LATENCY} \
    --socket-id ${SOCKET_ID}
    --docker-image intel/intel-optimized-tensorflow:latest \

cd $(INITIAL_DIR)

# TODO
# echo "3. Clean up"
# unset OMP_NUM_THREADS
# unset KMP_AFFINITY
# unset KMP_BLOCKTIME
# unset KMP_SETTINGS
# python ${RUNTIME_OPT_PATH} false

echo "All inference benchmark completed successfully!"


## fp32 torch

## fp32 IR

ice lake
torch-pipeline
memory: 1.8GB
latency: 9.45 ms
Setting OpenVINO CACHE_DIR

## int8 dynamic

### quantize

### inference

ice lake
torch_inference_int8_dynamic
memory: 1GB
latency: 9.5 ms
pipeline_int8_dynamic
memory: 1GB
latency: 13.4 ms

## int8 static quantize

### quantize

use model on hub, not support local model

### inference

ice lake
torch-pipeline
memory: 1.2GB
latency: 6.07 ms
Setting OpenVINO CACHE_DIR to /home/sdp/models/distilbert-base-uncased-finetuned-sst-2-english-int8-static/model_cache
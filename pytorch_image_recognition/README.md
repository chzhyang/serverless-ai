# run in container

```
docker pull chzhyang/classification_resnet50_torch:v1
docker run --rm chzhyang/classification_resnet50_torch:v1
```
# run in host


## benchmark

benchmark with 100 iterations
```
python3 classification_resnet50_openvino.py True banana.jpg 100 False True
```

## just inference

inference using a local model

```
python3 classification_resnet50_openvino.py False banana.jpg 100 False True public/resnet-50-pytorch/FP32/resnet-50-pytorch.xml
```

# run in knative - todo
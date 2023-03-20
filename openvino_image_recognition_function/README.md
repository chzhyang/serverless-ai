# run in container

```
docker pull chzhyang/classification_resnet50_openvino:v1
docker run --rm chzhyang/classification_resnet50_openvino:v1
```

# run in host

```
# download and convert model
python3 classification_resnet50_openvino.py public/resnet-50-pytorch/FP32/resnet-50-pytorch.xml banana.jpg CPU imagenet2012.json 100 true resnet-50-pytorch

# use local IR model
python3 classification_resnet50_openvino.py public/resnet-50-pytorch/FP32/resnet-50-pytorch.xml banana.jpg CPU imagenet2012.json 100 false resnet-50-pytorch
```

# run in knative - todo
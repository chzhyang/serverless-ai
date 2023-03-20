
# knative function
## build image
```
docker build -t chzhyang/classification-openvino-func:v1 .
docker push chzhyang/classification-openvino-func:v1
```
## deploy locally
```bash
$ docker pull chzhyang/classification-openvino-func:v1
$ docker run --rm -p 9000:8080 chzhyang/classification-openvino-func:v1
```

```bash
$ curl localhost:9000?imageName=test1
{"top_prediction": "king penguin, Aptenodytes patagonica"}
```

## deploy in knative
```
kubectl apply -f knative-func.yaml
```

> detail deployment and test refer to `tensorflow-image-recognition/README.md`

# benchmark 
## run in container

```
docker pull chzhyang/classification_resnet50_openvino:v1
docker run --rm chzhyang/classification_resnet50_openvino:v1
```

## run in host

```
# download and convert model
python3 classification_resnet50_openvino.py model/FP32/resnet-50-pytorch.xml banana.jpg CPU model/imagenet2012.json 100 true resnet-50-pytorch

# use local IR model
python3 classification_resnet50_openvino.py model/FP32/resnet-50-pytorch.xml banana.jpg CPU model/imagenet2012.json 100 false resnet-50-pytorch
```

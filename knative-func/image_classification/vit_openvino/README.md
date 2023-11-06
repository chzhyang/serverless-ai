
# knative function
[ViT](https://huggingface.co/docs/transformers/model_doc/vit)
https://huggingface.co/google/vit-base-patch16-224
model size: 331MB
running with openvino need memory size: 2GB+
## build image
```bash
docker build -t chzhyang/classification-vit-ov:v1 .
docker push chzhyang/classification-vit-ov:v1
```
## deploy locally
1. run container for local image inference
    ```bash
    $ docker pull chzhyang/classification-vit-ov:v1
    $ docker run --rm -p 9000:8080 chzhyang/classification-vit-ov:v1
    ```

    ```bash
    $ curl localhost:9000?imageName=test1
    {"top_prediction": "king penguin, Aptenodytes patagonica"}
    ```
2. run container with Ceph connection configuration, required connection to Ceph object storage

    ```bash
    docker pull chzhyang/classification-vit-ov:v1
    docker run -p 9000:8080 \
    -e S3_ENABLED='true' \
    -e S3_ACCESS_KEY='Y51LDIT65ZN41VVLKG0H' \
    -e S3_SECRET_KEY='GOxbWKx5NunhlAt3xTvDUh3uHP04A6Cv3UFEwdGS' \
    -e S3_ENDPOINT_URL='http://10.110.230.223:80' \
    -e S3_BUCKET_NAME='fish' \
    -e S3_AWS_REGION='my-store' \
    -e S3_EVENT_NAME='ObjectCreated:Put' \
    chzhyang/classification-openvino-ceph:v1
    ```

    curl container with cloudevent

    ```bash
    $ curl -H "Content-Type: application/cloudevents+json" \
    -X POST \
    -d "@{absolutely parent dir}/cloud.faas.faststartplus/benchmark/workload/openvino-image-recognition/test/event.json" \
    http://127.0.0.1:9000

    {"top_prediction": [["king penguin, Aptenodytes patagonica"]]}
    ```
## deploy in knative
1. prepare image

    ```bash
    # build image
    docker build -t chzhyang/classification-vit-ov:v1 .

    # push image to registory
    docker push chzhyang/classification-vit-ov:v1
    ```
2. config `knative-func.yaml`
    > Ceph connection ENV should be configed in `knative-func.yaml` if users need online image inference with cloudevent

3. deploy service to cluster

    `kubectl apply -f knative-func.yaml`

4. send image to Ceph for online image inference(optional)
    use awscli or other tools to send an image to target bucket of Ceph object sotrage

5. curl service for online image inference with cloudevent(optional)

    send function a fake event to simulate notification event from Ceph

    ```bash
    $ curl -H "Content-Type: application/application/cloudevents+json" \
    -H "Host:classification-vit-ov.default.example.com" \
    -X POST \
    -d "@{absolutely parent dir}/cloud.faas.faststartplus/benchmark/workload/openvino-image-recognition/test/event.json" \
    http://{node IP}:{node port of gateway}

    {"top_prediction": [["king penguin, Aptenodytes patagonica"]]}
    ```

    > [Configure DNS](https://knative.dev/docs/install/yaml-install/serving/install-serving-with-yaml/#configure-dns) if you cannot reach the service

6. curl service for local image inference with HTTP GET(optional)

    ```bash
    $ curl -H "Host:classification-openvino-ceph.default.example.com" \
    http://{node IP}:{node port of gateway}?imageName=test1

    {"top_prediction": [["king penguin, Aptenodytes patagonica"]]}
    ```

# benchmark 
## run in container

```
docker pull chzhyang/classification-vit-ov:v1
docker run --rm chzhyang/classification-vit-ov:v1
```

## run in host

download and convert model
```bash
python3 classification-vit-ov.py . data/test1.jpg CPU model/imagenet2012.json 100 true resnet-50-pytorch FP32
```

use local IR model
```bash
python3 classification-vit-ov.py model/public/resnet-50-pytorch/FP32/resnet-50-pytorch.xml data/test1.jpg CPU model/imagenet2012.json 100 false resnet-50-pytorch FP32
```

# image recognition function with tensorflow
This sample project shows an function for real-time image recognition inference with a pretrained model(ResNet50). The function supports image recognition inference of local image(HTTP GET request) and inference of online image stored in Ceph object storage(cloudevent reuqest).

The inference(func.py) is based on class ImageRecognitionService(image_recognition_service.py) which defines serveral functions for data preprocessing, running inference and inference result parsing using intel optimized TensorFlow.

Frameworks used in this project including intel-tensorflow, Knative, Ceph, parliament, Flask, CloudEvent, etc.

## function initilization

- ImageRecognitionService will be initilized before creating snapshot, including tensorflow runtion optimization, loading model, caching model
- For online image(Ceph) inference, `ENV("S3_ENABLED")` should be configured to "true", **S3 connection(Ceph object storage) will be intilized after restoring snapshot and beofre handling the first request**. Users can config S3 connection information to ENV, including:
    - S3_ENABLED
    - S3_ACCESS_KEY
    - S3_SECRET_KEY
    - S3_ENDPOINT_URL
    - S3_BUCKET_NAME
    - S3_AWS_REGION
    - S3_EVENT_NAME

## function use case

### inference of online image stored in Ceph object storage(cloudevent reuqest)
If the service receives a cloudevent request, it will get object key of the image stored in Ceph oject storage from the clouevent, then download the image from Ceph, and run inference finally.

**requirements**

a K8s cluster installed Knative serving, Knative eventing, Knative/eventing-ceph and Ceph

**pipeline**

1. Client sends an image to Ceph
2. Ceph sends notification event to Knative
3. Knative eventing-ceph fowards the event to the function(Knative services)
4. After received an event, the function will download image from Ceph, and run inference

### inference of local image(HTTP GET request)
If the service receives an HTTP GET request, it will get `imageName`(such as test1) from reuqest, and run inference if the image exists in `./data`, otherwise, run inference of default local image(`test1.JPEG`).

## function deployment and test

### deploy function on Knative cluster and Ceph

1. prepare image

```shell
# build image
docker build -t chzhyang/classification-tensorflow-ceph:v3 .

# push image to registory
docker push chzhyang/classification-tensorflow-ceph:v3
```

2. config `knative-func.yaml`
    > Ceph connection ENV should be configed in `knative-func.yaml` if users need online image inference with cloudevent

3. deploy service to cluster

    `kubectl apply -f knative-func.yaml`

4. send image to Ceph for online image inference(optional)
    use awscli or other tools to send an image to target bucket of Ceph object sotrage

5. curl service for online image inference with cloudevent(optional)

    send function a fake event to simulate notification event from Ceph

    ```shell
    $ curl -H "Content-Type: application/application/cloudevents+json" \
    -H "Host:classification-tensorflow-ceph.default.example.com" \
    -X POST \
    -d "@{absolutely parent dir}/cloud.faas.faststartplus/benchmark/workload/tensorflow-image-recognition/test/event.json" \
    http://{node IP}:{node port of gateway}

    {"top_prediction": [["king penguin, Aptenodytes patagonica"]]}
    ```

    > [Configure DNS](https://knative.dev/docs/install/yaml-install/serving/install-serving-with-yaml/#configure-dns) if you cannot reach the service

6. curl service for local image inference with HTTP GET(optional)

    ```shell
    $ curl -H "Host:classification-tensorflow-ceph.default.example.com" \
    http://{node IP}:{node port of gateway}?imageName=test1

    {"top_prediction": [["king penguin, Aptenodytes patagonica"]]}
    ```

### deploy function with docker locally

1. container for inference online image of Ceph object storage

    run container with Ceph connection configuration, required connection to Ceph object storage

    ```shell
    docker pull chzhyang/classification-tensorflow-ceph:v3
    docker run -p 9000:8080 \
    -e S3_ENABLED='true' \
    -e S3_ACCESS_KEY='Y51LDIT65ZN41VVLKG0H' \
    -e S3_SECRET_KEY='GOxbWKx5NunhlAt3xTvDUh3uHP04A6Cv3UFEwdGS' \
    -e S3_ENDPOINT_URL='http://10.110.230.223:80' \
    -e S3_BUCKET_NAME='fish' \
    -e S3_AWS_REGION='my-store' \
    -e S3_EVENT_NAME='ObjectCreated:Put' \
    chzhyang/classification-tensorflow-ceph:v3
    ```

    curl container with cloudevent

    ```shell
    $ curl -H "Content-Type: application/application/cloudevents+json" \
    -X POST \
    -d "@{absolutely parent dir}/cloud.faas.faststartplus/benchmark/workload/tensorflow-image-recognition/test/event.json" \
    http://127.0.0.1:9000

    {"top_prediction": [["king penguin, Aptenodytes patagonica"]]}
    ```

2. container for local image inference

    run container
    ```shell
    docker pull chzhyang/classification-tensorflow-ceph:v3
    docker run -p 9000:8080 chzhyang/classification-tensorflow-ceph:v3
    ```

    curl container with HTTP GET
    ```shell
    $ curl http://localhost:9000?imageName=test1

    {"top_prediction": [["king penguin, Aptenodytes patagonica"]]}
    ```

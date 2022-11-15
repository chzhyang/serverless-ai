# HTTP Function for Image Recognition Inference with TensorFlow

This sample project shows a HTTP function(`func.py`) for real-time image recognition inference 
with TensorFlow on a pretrained model(ResNet50). The function will respond to incoming HTTP GET and POST requests.

The function is based on two modules:
- `image_recognition_service.py` contains a class(`ImageRecognitionService`) for image recognition inference with optimized TensorFlow
- `data_preprocess.py` contains serveral functions for data preprocess

## Endpoints

Running this function will expose three endpoints.

  - `/` The endpoint for inference.
    - A GET request will receive a response with the inference result of a test image(data/test.JPEG)
    - A POST request should contain a image URL in JSON format(`{'imgURL':'your custom URL'}`), the client will receive a response with the inference result of the image
  - `/health/readiness` The endpoint for a readiness health check
  - `/health/liveness` The endpoint for a liveness health check

## Testing

This function project includes three test functions in [unit test](./test_func.py):
- test GET request
- test POST request
- test empty request

```bash
$ python test_func.py
```
## Build and run locally

Prerequirements:
- [func](https://github.com/knative/func) is installed

```bash
$ cd .../func-tastic/tensorflow/image_recognition
$ func build
$ func run
  🙌 Function image built: docker.io/myrepo/tensorflow-image_recognition:latest
Detected function was already built.  Use --build to override this behavior.
Function started on port 8080
Load model...
...
Cache model...
...
Ready for inference... 
```

## Function invocation

Invoke using `func invoke`

```bash
$ func invoke --data '{"imgURL": "https://raw.githubusercontent.com/chzhyang/faas-workloads/main/tensorflow/image_recognition/tensorflow_image_classification/data/ILSVRC2012_test_00000181.JPEG"}'
{"top_predictions": [["king penguin, Aptenodytes patagonica", "drake", "albatross, mollymawk", "toucan", "guenon, guenon monkey"]]}
```

Invoke using `curl`

```bash
curl http://127.0.0.1:8080
{"top_predictions": [["king penguin, Aptenodytes patagonica", "drake", "albatross, mollymawk", "toucan", "guenon, guenon monkey"]]}

$ curl -X POST http://127.0.0.1:8080 \
-H "Content-Type: application/json"  \
-d '{"imgURL": "https://raw.githubusercontent.com/chzhyang/faas-workloads/main/tensorflow/image_recognition/tensorflow_image_classification/data/ILSVRC2012_test_00000181.JPEG"}'
{"top_predictions": [["king penguin, Aptenodytes patagonica", "drake", "albatross, mollymawk", "toucan", "guenon, guenon monkey"]]}
```


## Deploy into cluster

This command will build and deploy the function into cluster.

## Cleanup

To remove the deployed function from your cluster, run:

```shell
func delete
```

FROM tensorflow/tensorflow:2.9.1

#ENV http_proxy=http://proxy-fm.intel.com:911
#ENV https_proxy=http://proxy-fm.intel.com:912
#ENV no_proxy="10.105.159.2,10.105.159.2,10.105.159.0/24,10.250.0.0/16,10.254.0.0/15,10.253.0.0/16,127.0.0.1,localhost,.svc,.pod,.cluster.local,.svc.cluster.local,.intel.com"

RUN apt-get update
RUN apt-get install python3-venv -y
# libs for cv2
RUN apt-get install ffmpeg libsm6 libxext6 -y

WORKDIR /home
RUN python3 -m venv openvino_env
RUN source openvino_env/bin/activate
RUN python3 -m pip install --upgrade pip
RUN pip install openvino-dev[tensorflow2,pytorch,onnx]
# fix /usr/lib/python3.8/site-packages/openvino/model_zoo/internal_scripts/pytorch_to_onnx.py line 49
RUN wget https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/master/tools/model_tools/src/openvino/model_zoo/internal_scripts/pytorch_to_onnx.py
RUN cp pytorch_to_onnx.py /usr/lib/python3.8/site-packages/openvino/model_zoo/internal_scripts/pytorch_to_onnx.py

RUN git clone --branch dev https://github.com/chzhyang/openvino.git
WORKDIR /home/openvino/samples/python/hello_classification

CMD exec python3 classification_resnet50_openvino.py public/resnet-50-pytorch/FP32/resnet-50-pytorch.xml banana.jpg CPU imagenet2012.json 100 true resnet-50-pytorch 
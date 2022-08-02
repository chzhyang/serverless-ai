FROM python:3.7

#ENV http_proxy=http://proxy-fm.intel.com:911
#ENV https_proxy=http://proxy-fm.intel.com:912
#ENV no_proxy="10.105.159.2,10.105.159.2,10.105.159.0/24,10.250.0.0/16,10.254.0.0/15,10.253.0.0/16,127.0.0.1,localhost,.svc,.pod,.cluster.local,.svc.cluster.local,.intel.com"

# COPY requirements.txt /social-network-microservices/requirements.txt
COPY faster-rcnn /faster-rcnn

# RUN pip install --no-cache-dir -r /faster-rcnn/requirements.txt
RUN pip install --no-cache-dir numpy==1.19.2 pillow==8.2.0 tensorflow-cpu==2.5.0 tensorflow-hub=0.12.0 matplotlib=3.5.2
# Warm up: download and cache models
# RUN python /faster-rcnn/TextFilterService_cache_models.py

WORKDIR /faster-rcnn
# ENTRYPOINT ["python", "handler.py"]

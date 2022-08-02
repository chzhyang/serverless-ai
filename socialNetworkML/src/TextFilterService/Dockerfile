FROM python:3.7

#ENV http_proxy=http://proxy-fm.intel.com:911
#ENV https_proxy=http://proxy-fm.intel.com:912
#ENV no_proxy="10.105.159.2,10.105.159.2,10.105.159.0/24,10.250.0.0/16,10.254.0.0/15,10.253.0.0/16,127.0.0.1,localhost,.svc,.pod,.cluster.local,.svc.cluster.local,.intel.com"

COPY requirements.txt /social-network-microservices/requirements.txt
COPY src/TextFilterService /social-network-microservices/src/TextFilterService
COPY third_party /social-network-microservices/third_party
COPY gen-cpp /social-network-microservices/gen-cpp
COPY gen-lua /social-network-microservices/gen-lua
COPY gen-py /social-network-microservices/gen-py
COPY config /social-network-microservices/config
COPY keys /keys

RUN pip install --no-cache-dir -r /social-network-microservices/requirements.txt
RUN pip install --no-cache-dir -r \
    /social-network-microservices/src/TextFilterService/requirements.txt

# Warm up: download and cache models
RUN python -m textblob.download_corpora
RUN python /social-network-microservices/src/TextFilterService/TextFilterService_cache_models.py

#  Run Transformers in a firewalled or a no-network environment.
ENV HF_DATASETS_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

WORKDIR /social-network-microservices/src/TextFilterService
ENTRYPOINT ["python", "TextFilterService.py"]

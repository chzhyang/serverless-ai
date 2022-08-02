FROM locustio/locust

ENV http_proxy=http://proxy-dmz.intel.com:912
ENV https_proxy=http://proxy-dmz.intel.com:912
#ENV no_proxy="10.105.159.2,10.105.159.2,10.105.159.0/24,10.250.0.0/16,10.254.0.0/15,10.253.0.0/16,127.0.0.1,localhost,.svc,.pod,.cluster.local,.svc.cluster.local,.intel.com"

COPY docker-entrypoint.sh /docker-entrypoint.sh
COPY src /mnt/locust/src
COPY lib /mnt/locust/lib
COPY data /mnt/locust/data
COPY configs /mnt/locust/configs

USER root
RUN chown -R locust:locust /mnt/locust

USER locust
RUN mkdir -p /mnt/locust/logs
RUN pip install numpy

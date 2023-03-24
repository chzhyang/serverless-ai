FROM intel/intel-optimized-tensorflow:2.9.1

#ENV http_proxy=http://proxy-fm.intel.com:911
#ENV https_proxy=http://proxy-fm.intel.com:912
#ENV no_proxy="10.105.159.2,10.105.159.2,10.105.159.0/24,10.250.0.0/16,10.254.0.0/15,10.253.0.0/16,127.0.0.1,localhost,.svc,.pod,.cluster.local,.svc.cluster.local,.intel.com"

COPY func.py /home/func.py
# COPY test_func.py /home/test_func.py
COPY image_recognition_service.py /home/image_recognition_service.py
# COPY parliament home/parliament
COPY data /home/data
COPY model /home/model
COPY requirements.txt /home/requirements.txt
RUN pip install -r /home/requirements.txt

# RUN pip install parliament-functions==0.1.0
# RUN pip install intel-tensorflow==2.9.1

WORKDIR /home
RUN apt-get update
RUn apt-get install git -y
RUN git clone -b coldstartup https://github.com/chzhyang/parliament.git
# Run the web service on container startup.
CMD exec python3 -m parliament.parliament .

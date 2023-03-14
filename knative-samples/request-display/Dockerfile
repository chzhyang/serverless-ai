FROM python:slim

#ENV http_proxy=
#ENV https_proxy=
#ENV no_proxy="127.0.0.1,localhost,.svc,.pod,.cluster.local,.svc.cluster.local"
WORKDIR /home
COPY app.py /home/func.py
COPY requirements.txt /home/requirements.txt
RUN pip install -r requirements.txt

# Run the web service on container startup.
CMD exec python3 -m parliament .
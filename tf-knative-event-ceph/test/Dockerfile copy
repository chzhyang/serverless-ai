# Use the official Python image.
# https://hub.docker.com/_/python
FROM python:3.7-slim

# Copy local code to the container image.
ENV APP_HOME /app
ENV PYTHONUNBUFFERED TRUE

WORKDIR $APP_HOME
COPY test_s3_boto3.py $APP_HOME/test.py
COPY requirements.txt $APP_HOME/requirements.txt
COPY test4.txt $APP_HOME/test4.txt
# Install production dependencies.
RUN pip install -r requirements.txt

# Run the web service on container startup.
CMD exec python3 test.py

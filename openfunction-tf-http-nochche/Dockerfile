# Use the official Python image.
# https://hub.docker.com/_/python
FROM python:3.7-slim

# Copy local code to the container image.
ENV APP_HOME /app
ENV PYTHONUNBUFFERED TRUE

WORKDIR $APP_HOME
COPY data $APP_HOME/data
COPY main.py $APP_HOME/main.py
COPY image_recognition_service.py $APP_HOME/image_recognition_service.py

COPY requirements.txt $APP_HOME/requirements.txt

# Install production dependencies.
RUN pip install gunicorn functions-framework
RUN pip install -r requirements.txt

# Run the web service on container startup.
CMD exec functions-framework --target=handler --signature-type=http

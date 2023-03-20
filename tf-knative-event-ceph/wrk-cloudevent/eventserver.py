import datetime
import os
import json
from pathlib import Path
from parliament import Context
import logging as log
from flask import request
BUCKET_NAME = "fish"
AWS_REGION = "my-store"
EVENT_NAME = "ObjectCreated:Put"


def main(context: Context):
    """
    Image recognition inference with optimized TensorFlow
    """
    if context.cloud_event is not None:
        log.info(f'event data: {context.cloud_event}')
        data_dict = context.cloud_event.data
        log.info(f'event data: {data_dict}')
        return json.dumps(context.cloud_event), 200
    else:
        log.error("Empty event")
        return "{}", 400

import os
from http.client import BAD_REQUEST
from pathlib import Path
import requests
from flask import json
from werkzeug.exceptions import MethodNotAllowed, BadRequest, InternalServerError, HTTPException


def handler(request):
    """
    Handle the request.

    Image recognition inference with optimized TensorFlow.

    """

    if request == None:
        print("Empty request", flush=True)
        return "{}", 200
    if request.method == "GET":
        print("#-------------GET", flush=True)
        return "This is GET test", 200
    # elif request.method == "POST":
    #     # Download the image, then run inference
    #     if not request.is_json:
    #         raise BadRequest(description="only JSON body allowed")
    #     try:
    #         data = request.get_json()
    #         img_url = data["imgURL"]
    #         img_filepath = download_image(img_url, DATA_DIR)
    #         predictions = SERVICE.run_inference(
    #             img_filepath, LABELS_PATH, NUM_TOP_PREDICTIONS)
    #         result = {
    #             "top_predictions": predictions
    #         }
    #         print(result, flush=True)
    #         return json.dumps(result), 200
    #     except KeyError:
    #         raise BAD_REQUEST(description='missing imgURL in JSON')
    #     except Exception as e:
    #         raise InternalServerError(original_exception=e)
    else:
        raise MethodNotAllowed(valid_methods=['GET, POST'])

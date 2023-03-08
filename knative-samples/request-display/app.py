import json

from parliament import Context
import logging as log

log.basicConfig(
    level=log.INFO,
    format='%(asctime)s::%(levelname)s::%(message)s',
)


def main(context: Context):
    if context is None:
        log.info("None context")
        return "{None context}", 400
    # display cloudevent
    if context.cloud_event is not None:
        dict_data = context.cloud_event.data
        log.info(
            f'Reveived cloudevents.Event {dict_data["id"]} from {dict_data["source"]}')
        return json.dumps(resp), 200

    # display http request
    elif context.request is not None:
        if context.request.method == "GET":
            log.info(f'Received a GET request')
            return json.dumps("hello world")
        elif context.request.method == "POST":
            log.info(f'Received a POST request: {context.request.get_data()}')
            return json.dumps("hello world")
        else:
            resp = "Server just supports GET, POST and CloudEvent(POST) requeset now"
            log.error(resp)
            return json.dumps(resp), 200
    else:
        log.error("Empty request")
        return "{Empty request}", 400

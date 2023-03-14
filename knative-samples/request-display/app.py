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
        resp = "Reveived a cloudevents.Event"
        log.info(
            f'Reveived cloudevents.Event from {dict_data["eventSource"]}, event id is {dict_data["eventId"]}')
        return json.dumps(resp), 200

    # display http request
    elif context.request is not None:
        if context.request.method == "GET":
            resp = f'Received a GET request'
            log.info(resp)
            return json.dumps(resp)
        elif context.request.method == "POST":
            resp = f'Received a POST request'
            log.info(
                f'Received a POST request: {context.request.get_data(as_text=True)}')
            return json.dumps(resp), 200
        else:
            resp = "Server just supports GET, POST and CloudEvent(POST) requeset now"
            log.error(resp)
            return json.dumps(resp), 400
    else:
        log.error("Empty request")
        return "{Empty request}", 400

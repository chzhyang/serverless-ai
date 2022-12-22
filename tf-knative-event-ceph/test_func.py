import json
import unittest
from werkzeug.test import EnvironBuilder
from parliament import Context
from cloudevents.http import CloudEvent, to_binary
import requests

func = __import__("func")


class TestFunc(unittest.TestCase):

    def test_func_empty_request(self):
        resp, code = func.main({})
        self.assertEqual(resp, "{}")
        self.assertEqual(code, 200)

    def test_func_event(self):
        attributes = {
            "type": "com.amazonaws.ObjectCreated:Put",
            "source": "ceph:s3.my-store.fish",
            "id": "d5a87e11-31a2-4e74-9f6d-ebc48a187271.4464.22267114304028917371170-my-store-my-store",
            "datacontenttype": "application/json",
        }
        data = {
            "eventVersion": "2.2",
            "eventSource": "ceph:s3",
            "awsRegion": "my-store",
            "eventTime": "2022-12-21T08:07:01.958659Z",
            "eventName": "ObjectCreated:Put",
            "userIdentity": {
                "principalId": "rgw-admin-ops-user"
            },
            "requestParameters": {
                "sourceIPAddress": ""
            },
            "responseElements": {
                "x-amz-request-id": "d5a87e11-31a2-4e74-9f6d-ebc48a187271.4464.5813176538436958134",
                "x-amz-id-2": "1170-my-store-my-store"
            },
            "s3": {
                "s3SchemaVersion": "1.0",
                "configurationId": "notif1",
                "bucket": {
                    "name": "fish",
                    "ownerIdentity": {
                        "principalId": "rgw-admin-ops-user"
                    },
                    "arn": "arn:aws:s3:::fish",
                    "id": "d5a87e11-31a2-4e74-9f6d-ebc48a187271.4466.3"
                },
                "object": {
                    "key": "test_img.JPEG",
                    "size": 156810,
                    "eTag": "a0c1a4b3e7b1914513a4a3f906d94ca0",
                    "versionId": "",
                    "sequencer": "A5BEA263AA549339",
                    "metadata": [
                        {
                            "key": "x-amz-content-sha256",
                            "value": ""
                        },
                        {
                            "key": "x-amz-date",
                            "value": ""
                        }
                    ]
                }
            },
            "eventId": "1671610021.965956.a0c1a4b3e7b1914513a4a3f906d94ca0"
        }
        event = CloudEvent(attributes, data)
        builder = EnvironBuilder(method="POST")
        req = builder.get_request()
        ctx = Context(req)
        ctx.cloud_event = event
        print("test get event:")
        print(ctx.cloud_event)
        resp, code = func.main(ctx)
        print("resp:")
        print(resp)
        self.assertIn("king penguin", resp)
        self.assertEqual(code, 200)


if __name__ == "__main__":
    unittest.main()

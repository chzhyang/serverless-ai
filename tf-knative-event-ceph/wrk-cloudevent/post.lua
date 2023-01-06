json = require "json"
wrk.method = "POST"
wrk.body = json.encode(dict)
body = '{
    "data": {
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
        "eventId": "client.test"
    }
}'
-- wrk.headers["Content-Type"] = "application/json"
wrk.headers["Content-Type"] = "application/cloudevents+json"
print(wrk.body)

wrk.method = "POST"
-- wrk.host = "http://tf-function-event-perf-warm.default.svc.cluster.local"
wrk.body = [[{
    "specversion": "1.0",
    "time": "2022-12-19T06:22:27.878508Z",
    "type": "com.amazonaws.ObjectCreated:Put",
    "source": "ceph:s3.my-store.fish",
    "id": "d5a87e11-31a2-4e74-9f6d-ebc48a187271.4464.22267114304028917371170-my-store-my-store",
    "datacontenttype": "application/json",
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
                "key": "test1.JPEG",
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
        "eventId": "curl.event.1"
    }
}]]
wrk.headers = {}
wrk.headers["Content-Type"] = "application/cloudevents+json"


import os
import json
import boto
from io import BytesIO
from pathlib import Path

PRETRAINED_MODEL = "resnet50_v1_5_fp32.pb"
DATA_DIR = Path(__file__).resolve().parent / 'data'
MODEL_PATH = os.path.join(DATA_DIR, PRETRAINED_MODEL)
# Labels used for mapping inference results to human-readable predictions
LABELS_PATH = os.path.join(DATA_DIR, 'labellist.json')
# A local image used for handling GET request
TEST_IMAGE = os.path.join(DATA_DIR, 'test.JPEG')

# Number of top human-readable predictions
NUM_TOP_PREDICTIONS = 5

# Init the image recognition service class

# s3 connection
BUCKET_NAME = "fish"
AWS_REGION = "my-store"
EVENT_NAME = "ObjectCreated:Put"

ACCESS_KEY = "Y51LDIT65ZN41VVLKG0H"
SECRET_KEY = "GOxbWKx5NunhlAt3xTvDUh3uHP04A6Cv3UFEwdGS"
# ENDPOINT_URL = "http://rook-ceph-rgw-my-store:80"
ENDPOINT_URL = "http://10.110.230.223"

class CephS3():
    def __init__(self, endpoint_url, access_key, secret_key):
        self.s3_conn = bboto.connect_s3(aws_access_key_id = access_key,
                                        aws_secret_access_key = secret_key,
                                        host = endpoint_url,
                                        is_secure=False,               # uncomment if you are not using ssl
                                        calling_format = boto.s3.connection.OrdinaryCallingFormat(),
                                        )

    def get_bucket(self, bucket_name):
        return self.s3_conn.Bucket(bucket_name)
    
    def download_s3_file(self, bucket, object_key):
        data = BytesIO()
        bucket.download_fileobj(Fileobj=data, Key=object_key)
        return data

def main():
    S3 = CephS3(ENDPOINT_URL, ACCESS_KEY, SECRET_KEY)
    bucket = S3.get_bucket(BUCKET_NAME)

    print("test download file to bytes")
    object_key = 'test3.txt'
    file_bytes = S3.download_s3_file(bucket, object_key)
    print(file_bytes.getvalue())

    print("test upload file to bucket")
    up_file_path = Path(__file__).resolve().parent / 'requirements.txt'
    bucket.upload_file(Filename=up_file_path,
                       Key='requirements.txt')

    print("test list object")
    response = S3.list_objects_v2(Bucket=BUCKET_NAME)
    for item in response['Contents']:
        print(item['Key'])

if __name__ == '__main__':
    main()

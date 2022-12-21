import os
import json
import boto3
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
ENDPOINT_URL = "http://10.110.230.223:80"


class CephS3():
    def __init__(self, endpoint_url, access_key, secret_key):
        # self.s3_conn = boto3.resource(
        #     's3',
        #     endpoint_url=endpoint_url,
        #     aws_access_key_id=access_key,
        #     aws_secret_access_key=secret_key)
        self.client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key)

    # def get_bucket(self, bucket_name):
    #     return self.s3_conn.Bucket(bucket_name)

    # def download_s3_file_bytes(self, bucket, object_key):
    #     data = BytesIO()
    #     bucket.download_fileobj(Fileobj=data, Key=object_key)
    #     return data

    # def download_s3_file(self, bucket_name, object_key, file_name):
    #     self.client.download_file(bucket_name, object_key, file_name)


def main():
    S3 = CephS3(ENDPOINT_URL, ACCESS_KEY, SECRET_KEY)
    # bucket = S3.get_bucket(BUCKET_NAME)

    # print("test download file to bytes")
    print("download object")
    object_key = 'test3.txt'
    file_name = Path(__file__).resolve().parent / 'test3.txt'
    # file_bytes =
    # print(file_bytes.getvalue())
    S3.client.download_file(BUCKET_NAME, object_key, file_name)
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            for line in f:
                print(line)
    else:
        print(file_name, " not exist.")

    print("upload file")
    up_file_path = Path(__file__).resolve().parent / 'test4.txt'
    # bucket.upload_file(Filename=up_file_path,
    #                    Key='test4.txt')
    response = S3.client.upload_file(up_file_path, BUCKET_NAME, 'test4.txt')
    print(response)

    print("list object")
    # response = S3.list_objects_v2(Bucket=BUCKET_NAME)
    # for item in response['Contents']:
    #     print(item['Key'])
    response = S3.client.list_objects(Bucket=BUCKET_NAME)
    print(response)


if __name__ == '__main__':
    main()

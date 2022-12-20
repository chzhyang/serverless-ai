import os
import json
import boto3
from io import BytesIO

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
ENDPOINT_URL = "http://rook-ceph-rgw-my-store:80"

class CephS3():
    def __init__(self, endpoint_url, access_key, secret_key):
        self.s3_conn = boto3.resource('s3',
                                    endpoint_url=endpoint_url,
                                    aws_access_key_id=access_key,
                                    aws_secret_access_key=secret_key)

    def get_bucket(self, bucket_name):
        return s3_conn.Bucket(bucket_name)
    
    def download_s3_file(self, bucket_name, object_key):
        data = BytesIO()
        bucket = self.get_bucket(bucket_name)
        bucket.download_fileobj(Fileobj=data, Key=object_key)
        return data

def main():
    S3 = ceph_s3.CephS3(ENDPOINT_URL, ACCESS_KEY, SECRET_KEY)
    object_key = test3.txt
    file_bytes = S3.download_s3_file(BUCKET_NAME, object_key)
    print(file_bytes.getvalue())

if __name__ == '__main__':
    main()

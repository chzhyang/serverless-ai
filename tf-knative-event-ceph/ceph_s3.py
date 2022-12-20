import boto3
from io import BytesIO

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
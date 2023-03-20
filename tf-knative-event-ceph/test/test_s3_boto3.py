import os
import json
import boto3
from io import BytesIO
from pathlib import Path

# s3 connection
BUCKET_NAME = "fish"
AWS_REGION = "my-store"
EVENT_NAME = "ObjectCreated:Put"

ACCESS_KEY = "Y51LDIT65ZN41VVLKG0H"
SECRET_KEY = "GOxbWKx5NunhlAt3xTvDUh3uHP04A6Cv3UFEwdGS"
ENDPOINT_URL = "http://10.110.230.223:80"


def main():
    s3 = boto3.client(
        's3',
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY)

    print("download object")
    object_key = 'test3.txt'
    file_name = Path(__file__).resolve().parent / 'test3.txt'
    s3.download_file(BUCKET_NAME, object_key, file_name)
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            for line in f:
                print(line)
    else:
        print(file_name, " not exist.")

    print("upload file")
    up_file_path = Path(__file__).resolve().parent / 'test4.txt'
    response = s3.upload_file(up_file_path, BUCKET_NAME, 'test4.txt')
    print(response)

    print("list object")
    response = s3.list_objects(Bucket=BUCKET_NAME)
    print(response)


if __name__ == '__main__':
    main()

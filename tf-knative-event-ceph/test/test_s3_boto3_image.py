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

    print("upload image")
    up_file_path = Path(__file__).resolve().parent / 'test_img.JPEG'
    response = s3.upload_file(up_file_path, BUCKET_NAME, 'test_img.JPEG')
    print(response)

    print("download image")
    object_key = 'test_img.JPEG'
    file_name = Path(__file__).resolve().parent / 'download_img.JPEG'
    s3.download_file(BUCKET_NAME, object_key, file_name)
    if not os.path.exists(file_name):
        print(file_name, " not exist.")

    print("list object")
    response = s3.list_objects(Bucket=BUCKET_NAME)
    print(response)


if __name__ == '__main__':
    main()

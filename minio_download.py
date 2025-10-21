from minio import Minio
import os
from dotenv import load_dotenv

load_dotenv()

MINIO_URL = os.getenv('MINIO_URL')
MINIO_USER = os.getenv('MINIO_USER')
MINIO_PASSWORD = os.getenv('MINIO_PASSWORD')
MINIO_BUCKET_NAME = os.getenv('MINIO_BUCKET_NAME')

client = Minio(
    MINIO_URL,
    access_key=MINIO_USER,
    secret_key=MINIO_PASSWORD,
    secure=False
)

for bucket in client.list_buckets():
    print(bucket.name)
    if bucket.name == MINIO_BUCKET_NAME:
        for item in client.list_objects(bucket.name, recursive=True):
            client.fget_object(bucket.name, item.object_name, item.object_name)

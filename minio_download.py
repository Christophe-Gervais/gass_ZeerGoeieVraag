from minio import Minio
from minio.error import S3Error
import tempfile
import cv2

MINIO_URL = "193.191.177.33:22555"
MINIO_USER = "ai-app-students"
MINIO_PASSWORD = "ai-app-students-welcome"
MINIO_BUCKET_NAME = "eyes4rescue"

client = Minio(
    MINIO_URL,
    access_key=MINIO_USER,
    secret_key=MINIO_PASSWORD,
    secure=False
)

for bucket in client.list_buckets():
    if bucket.name == MINIO_BUCKET_NAME:
        for item in client.list_objects(bucket.name, recursive=True):
            client.fget_object(bucket.name, item.object_name, item.object_name)

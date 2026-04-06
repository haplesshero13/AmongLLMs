import os

import boto3


def get_s3_client():
    endpoint_url = os.environ.get("S3_ENDPOINT_URL")
    access_key = os.environ.get("S3_ACCESS_KEY")
    secret_key = os.environ.get("S3_SECRET_KEY")
    region = os.environ.get("S3_REGION", "auto")
    if not all([endpoint_url, access_key, secret_key]):
        raise ValueError(
            "S3_ENDPOINT_URL, S3_ACCESS_KEY, and S3_SECRET_KEY env vars must be set"
        )

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )


def upload_logs_to_r2(logs_path: str, bucket_name: str = None):
    bucket_name = bucket_name or os.environ.get("S3_BUCKET_NAME")
    if not bucket_name:
        raise ValueError("S3_BUCKET_NAME env var not set")

    if not os.path.exists(logs_path):
        print(f"[R2] Logs path does not exist: {logs_path}")
        return

    client = get_s3_client()

    folder_name = os.path.basename(os.path.normpath(logs_path))

    for dirpath, dirnames, filenames in os.walk(logs_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(filepath, logs_path)
            key = f"{folder_name}/{rel_path}"
            client.upload_file(filepath, bucket_name, key)
            print(f"[R2] Uploaded {key}")

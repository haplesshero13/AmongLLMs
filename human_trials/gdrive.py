import os
import json
from google.cloud import storage
from google.oauth2 import service_account


def get_storage_client():
    service_account_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not service_account_json:
        raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON env var not set")

    info = json.loads(service_account_json)
    creds = service_account.Credentials.from_service_account_info(info)
    return storage.Client(credentials=creds, project=info["project_id"])


def upload_logs_to_drive(logs_path: str, bucket_name: str = None):
    bucket_name = bucket_name or os.environ.get("GCS_BUCKET_NAME")
    if not bucket_name:
        raise ValueError("GCS_BUCKET_NAME env var not set")

    if not os.path.exists(logs_path):
        print(f"[GCS] Logs path does not exist: {logs_path}")
        return

    client = get_storage_client()
    bucket = client.bucket(bucket_name)

    # Walk subdirectories recursively, mirroring folder structure in GCS
    for dirpath, dirnames, filenames in os.walk(logs_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            # Use relative path as the GCS object name to preserve folder structure
            rel_path = os.path.relpath(filepath, logs_path)
            blob = bucket.blob(rel_path)
            blob.upload_from_filename(filepath)
            print(f"[GCS] Uploaded {rel_path}")

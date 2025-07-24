import json
import math
import matplotlib.pyplot as plt
from google.cloud import storage

BUCKET_NAME = "ks_datasets"
PREFIX      = "preprocessed_data/"

def list_preprocessed_files(bucket, prefix):
    return [blob.name for blob in bucket.list_blobs(prefix=prefix) if blob.name.endswith(".json")]

def load_records(bucket, blob_name):
    raw = bucket.blob(blob_name).download_as_text()
    return json.loads(raw)

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)
files  = list_preprocessed_files(bucket, PREFIX)
n_files = len(files)

n_cols = 4
n_rows = math.ceil(n_files / n_cols)

plt.figure(figsize=(4 * n_cols, 3 * n_rows))

for idx, fname in enumerate(files):
    records = load_records(bucket, fname)
    lengths = [len(r["chunk"]) for r in records]

    row = idx // n_cols
    col = idx % n_cols
    ax = plt.subplot(n_rows, n_cols, idx + 1)
    ax.hist(lengths, bins=30)
    ax.set_title(fname.split("/")[-1], fontsize=8)
    ax.set_xlabel("chunk len", fontsize=7)
    ax.set_ylabel("count", fontsize=7)
    ax.tick_params(labelsize=6)

plt.tight_layout()
plt.savefig("chunk_lengths.png", dpi=300, bbox_inches='tight') 
plt.show()
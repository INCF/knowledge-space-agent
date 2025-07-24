
import os
import json
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

INPUT_JSONL_PATH = "/kaggle/input/ks-datasets/all_chunks.jsonl"
OUTPUT_FILE = "embeddings.jsonl"
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

ENCODE_BATCH_SIZE = 32

def generate_embeddings_multi_gpu():
    """
    Generates embeddings from a JSONL file, automatically using all available GPUs
    for parallel processing.
    """
    if not torch.cuda.is_available():
        print("RROR: No CUDA-enabled GPU found. Cannot perform multi-GPU encoding.")
        return

    gpu_count = torch.cuda.device_count()
    print(f"Found {gpu_count} CUDA-enabled GPU(s).")

    print(f"Loading SentenceTransformer model: '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

    print(f"Reading records from source file: {INPUT_JSONL_PATH}")
    with open(INPUT_JSONL_PATH, "r", encoding="utf-8") as f:
        texts_to_embed = [json.loads(line)["chunk"] for line in f if line.strip()]
        all_ids = [json.loads(line)["datapoint_id"] for line in open(INPUT_JSONL_PATH, "r", encoding="utf-8") if line.strip()]

    print(f"Successfully parsed {len(texts_to_embed):,} text chunks to embed.")

   
    print("Starting multi-GPU processing pool...")
    pool = model.start_multi_process_pool()

   
    print(f"Encoding embeddings across {gpu_count} GPUs...")
    embeddings = model.encode_multi_process(
        texts_to_embed,
        pool=pool,
        batch_size=ENCODE_BATCH_SIZE,
    )
    print("mbedding generation complete.")

    model.stop_multi_process_pool(pool)

    print(f"Writing {len(embeddings):,} embeddings to '{OUTPUT_FILE}'...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for i, embedding_vector in enumerate(embeddings):
            json_record = {
                "id": all_ids[i],
                "embedding": embedding_vector.tolist(),
            }
            f_out.write(json.dumps(json_record) + "\n")

    print(f"All done! Embeddings saved to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    generate_embeddings_multi_gpu()
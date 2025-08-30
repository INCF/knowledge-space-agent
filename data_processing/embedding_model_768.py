
import os
import json
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

INPUT_JSONL_PATH = "all_chunks.jsonl"
OUTPUT_FILE = "embeddings1.jsonl"
MODEL_NAME = os.getenv('EMBED_MODEL_NAME', "nomic-ai/nomic-embed-text-v1.5")

ENCODE_BATCH_SIZE = 32

def generate_embeddings():
    """
    Generates embeddings from a JSONL file, automatically detecting and using
    available hardware (GPUs if available, otherwise CPU).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        gpu_count = torch.cuda.device_count()
        print(f"Found {gpu_count} CUDA-enabled GPU(s). Using GPU acceleration.")
    else:
        print("No CUDA-enabled GPU found. Using CPU for embedding generation.")

    print(f"Loading SentenceTransformer model: '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=device)

    print(f"Reading records from source file: {INPUT_JSONL_PATH}")
    with open(INPUT_JSONL_PATH, "r", encoding="utf-8") as f:
        texts_to_embed = [json.loads(line)["chunk"] for line in f if line.strip()]
        all_ids = [json.loads(line)["datapoint_id"] for line in open(INPUT_JSONL_PATH, "r", encoding="utf-8") if line.strip()]

    print(f"Successfully parsed {len(texts_to_embed):,} text chunks to embed.")

    if device == "cuda" and gpu_count > 1:
        print("Starting multi-GPU processing pool...")
        pool = model.start_multi_process_pool()
        
        print(f"Encoding embeddings across {gpu_count} GPUs...")
        embeddings = model.encode_multi_process(
            texts_to_embed,
            pool=pool,
            batch_size=ENCODE_BATCH_SIZE,
        )
        
        model.stop_multi_process_pool(pool)
    else:
        print(f"Encoding embeddings on {device.upper()}...")
        embeddings = model.encode(
            texts_to_embed,
            batch_size=ENCODE_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_tensor=False
        )
    
    print("Embedding generation complete.")

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
    generate_embeddings()
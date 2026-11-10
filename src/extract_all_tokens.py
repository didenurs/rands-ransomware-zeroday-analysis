import time
import zipfile
import pandas as pd
import numpy as np
import sys
import os
from concurrent.futures import ProcessPoolExecutor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from config import ZIP_PATH, ZIP_PASSWORD, DATA_DIR, SPLIT_SCENARIO1_CSV
from tokenizer import extract_tokens_from_json

OUTPUT_CACHE = os.path.join(DATA_DIR, "tokens_cache.parquet")

def process_chunk(chunk_tuples):
    """
    Each tuple is (sha256, filepath)
    """
    results = []
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        for sha256, fp in chunk_tuples:
            try:
                content = z.read(fp, pwd=ZIP_PASSWORD)
                tokens = extract_tokens_from_json(content)
                results.append((sha256, tokens))
            except Exception as e:
                results.append((sha256, ""))
    return results

def main():
    if os.path.exists(OUTPUT_CACHE):
        print(f"Token cache already exists at {OUTPUT_CACHE}!")
        return
        
    print(f"Reading sample list from {SPLIT_SCENARIO1_CSV}...")
    df = pd.read_csv(SPLIT_SCENARIO1_CSV)
    print(f"Total samples to extract: {len(df)}")
    
    tuples = list(zip(df['SHA256'], df['Filepath']))
    
    # Split into chunks of 1000 for balanced parallel execution
    chunk_size = 1000
    chunks = [tuples[i:i + chunk_size] for i in range(0, len(tuples), chunk_size)]
    print(f"Divided into {len(chunks)} chunks of ~{chunk_size} samples.")
    
    num_workers = min(10, os.cpu_count() or 4)
    print(f"Starting ProcessPoolExecutor with {num_workers} workers...")
    
    t0 = time.time()
    all_results = []
    processed_count = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for chunk_res in executor.map(process_chunk, chunks):
            all_results.extend(chunk_res)
            processed_count += len(chunk_res)
            if processed_count % 10000 == 0 or processed_count == len(df):
                elapsed = time.time() - t0
                speed = processed_count / elapsed
                remaining = (len(df) - processed_count) / speed if speed > 0 else 0
                print(f"Progress: {processed_count}/{len(df)} samples ({processed_count/len(df)*100:.1f}%) "
                      f"- Speed: {speed:.1f} samples/sec - ETA: {remaining/60:.1f} mins")
                
    t1 = time.time()
    print(f"\nExtraction completed in {(t1 - t0)/60:.2f} minutes!")
    
    print("Saving token cache to Parquet...")
    df_tokens = pd.DataFrame(all_results, columns=['SHA256', 'Tokens'])
    df_tokens.to_parquet(OUTPUT_CACHE, index=False, compression='zstd')
    print(f"Saved {len(df_tokens)} tokenized samples to {OUTPUT_CACHE} (Size: {os.path.getsize(OUTPUT_CACHE)/(1024*1024):.2f} MB)")

if __name__ == '__main__':
    main()

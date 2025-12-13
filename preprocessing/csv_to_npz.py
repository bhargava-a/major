import pandas as pd
import numpy as np
import os
import time

# ==== CONFIGURE PATHS HERE ====
input_csv = r"C:\Users\bharg\Downloads\CESNET\institution_subnets\agg_1_hour\institutions_subnets_1hour_merged.csv"
output_npz = r"C:\Users\bharg\Downloads\CESNET\institution_subnets\agg_1_hour\npz\institutions_subnets_1hour_merged.npz"
# ==============================

# Ensure output folder exists
os.makedirs(os.path.dirname(output_npz), exist_ok=True)

arrays_list = []
chunk_size = 100_000  # adjust based on your RAM
chunk_id = 0

start = time.perf_counter()

for chunk in pd.read_csv(input_csv, chunksize=chunk_size):
    arr = chunk.to_numpy(dtype=np.float32)  # float32 to save space
    arrays_list.append(arr)
    chunk_id += 1
    print(f"Processed chunk {chunk_id}, shape: {arr.shape}")

# Save everything into one compressed NPZ
np.savez_compressed(output_npz, *arrays_list)

elapsed = time.perf_counter() - start
print(f"All done! NPZ saved at: {output_npz}")
print(f"Elapsed time: {elapsed/60:.2f} minutes")

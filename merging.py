import os
import pandas as pd
import time

folder = r"C:\Users\bharg\Downloads\CESNET\institution_subnets\agg_1_day"
output = "institutions_subnets_1day_merged.csv"

first = True

# start a high-resolution timer
start = time.perf_counter()

for file in os.listdir(folder):
    if file.endswith(".csv"):
        filepath = os.path.join(folder, file)
        for chunk in pd.read_csv(filepath, chunksize=100000):
            chunk["ip_id"] = file.replace(".csv", "")
            chunk.to_csv(output, mode="a", header=first, index=False)
            first = False

elapsed = time.perf_counter() - start
# format minutes correctly: divide first, then apply format specifier
print(f"Merging 1day completed in {elapsed/60:.2f} min.")

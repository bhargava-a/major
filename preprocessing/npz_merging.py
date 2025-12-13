import numpy as np
import os

input_npz = r"C:\Users\bharg\Downloads\CESNET\ip_addresses_sample\agg_1_day\npz\ip_1Day_merged.npz"
output_npz = r"C:\Users\bharg\Downloads\CESNET\ip_addresses_sample\agg_1_day\npz\ip_1Day_merged_single.npz"

os.makedirs(os.path.dirname(output_npz), exist_ok=True)

# Load NPZ properly
data = np.load(input_npz)

# Stack all arrays (arr_0, arr_1, ...)
full_matrix = np.vstack([data[k] for k in data.files])

# Save as single matrix
np.savez_compressed(output_npz, full_matrix=full_matrix)

print("Done bro.")
print("Final shape:", full_matrix.shape)

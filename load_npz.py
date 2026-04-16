import numpy as np

# ==== CONFIGURE PATH HERE ====
npz_file_path = r"C:\Users\bharg\Downloads\CESNET\institution_subnets\agg_1_day\npz\institutions_subnets_1day_merged_single.npz"
# ==============================

# Load the NPZ file
data = np.load(npz_file_path)

# Get the first array (or 'full_matrix' if it exists)
if 'full_matrix' in data.files:
    array = data['full_matrix']
else:
    array = data[data.files[0]]

# Display first 5 rows
print("First 5 rows:")
print(array[:5])

# Close the file
data.close()
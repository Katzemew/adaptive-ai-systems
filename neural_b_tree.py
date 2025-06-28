import numpy as np
import time
import sys
import matplotlib.pyplot as plt

# ----- Data Preparation -----
NUM_KEYS = 1_000_000  # 1 million keys for scale
NUM_QUERIES = 50_000
np.random.seed(42)

# Generate sorted unique keys (simulate product IDs or similar)
keys = np.arange(10_000_000, 10_000_000 + NUM_KEYS)
# Add noise to simulate the real-world scenario
# keys = keys + np.random.randint(-10, 10, NUM_KEYS)
queries = np.random.choice(keys, NUM_QUERIES, replace=True)

# ----- Classic "B-Tree" Search (numpy.searchsorted) -----
# Build
t0 = time.perf_counter()
# Sorted array ready (no real "build" needed for numpy)
sorted_arr = keys
classic_build_time = time.perf_counter() - t0
classic_mem = sorted_arr.nbytes / (1024 * 1024)  # MB

# Query
t0 = time.perf_counter()
found_classic = 0
for q in queries:
    idx = np.searchsorted(sorted_arr, q)
    if idx < len(sorted_arr) and sorted_arr[idx] == q:
        found_classic += 1
classic_query_time = time.perf_counter() - t0

# ----- Neural Index (Linear Regression) -----
# Build (train simple linear model)
t0 = time.perf_counter()
X = keys.astype(float)
y = np.arange(len(keys), dtype=float)
a, b = np.polyfit(X, y, 1)
neural_params = (a, b)
neural_build_time = time.perf_counter() - t0
neural_mem = keys.nbytes / (1024 * 1024) + sys.getsizeof(a) + sys.getsizeof(b)  # MB

# Query
t0 = time.perf_counter()
found_neural = 0
for q in queries:
    pred = int(np.clip(round(a * q + b), 0, len(keys)-1))
    # Local scan (cover model error)
    for delta in range(-10, 11):
        idx = pred + delta
        if 0 <= idx < len(keys) and keys[idx] == q:
            found_neural += 1
            break
neural_query_time = time.perf_counter() - t0

# ----- Results -----
print("\nClassic B-Tree (Sorted Array) vs Neural Index (Regression) Demo")
print("-" * 60)
print(f"{'Method':<18} {'Build Time (s)':<16} {'Query Time (s)':<16} {'Mem (MB)':<12} {'Accuracy (%)'}")
print("-" * 60)
print(f"{'Classic B-Tree':<18} {classic_build_time:<16.5f} {classic_query_time:<16.5f} {classic_mem:<12.2f} {found_classic/NUM_QUERIES*100:>10.2f}")
print(f"{'Neural Index':<18} {neural_build_time:<16.5f} {neural_query_time:<16.5f} {neural_mem:<12.2f} {found_neural/NUM_QUERIES*100:>10.2f}")
print("-" * 60)

# ----- Optional: Visual Comparison -----
plt.bar(['Classic B-Tree', 'Neural Index'], [classic_query_time, neural_query_time], color=['#6366f1', '#10b981'])
plt.ylabel("Total Query Time (seconds)")
plt.title("Classic vs Neural Index: Query Speed")
plt.show()
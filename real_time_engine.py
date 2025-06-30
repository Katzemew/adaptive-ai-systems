import numpy as np
import time
from collections import defaultdict
import pickle
import os
import matplotlib.pyplot as plt

# Parameters
NUM_PRODUCTS = 1_000_000
NUM_CATEGORIES = 500
NUM_QUERIES = 100_000
LEARNED_MIN_ITEMS = 1000

np.random.seed(42)

# 1. Simulate Data
product_ids = np.arange(10_000_000, 10_000_000 + NUM_PRODUCTS)
category_map = np.random.choice(range(NUM_CATEGORIES), NUM_PRODUCTS)
category_products = defaultdict(list)
for pid, cat in zip(product_ids, category_map):
    category_products[cat].append(pid)
for cat in category_products:
    category_products[cat] = np.sort(np.array(category_products[cat]))

# 2. Simulate Queries: 80/20 rule
HOT_FRACTION = 0.10
HOT_TRAFFIC = 0.80
cat_counts = np.bincount(category_map)
cat_sorted = np.argsort(-cat_counts)
num_hot = int(NUM_CATEGORIES * HOT_FRACTION)
hot_cats = set(cat_sorted[:num_hot])
cold_cats = set(cat_sorted[num_hot:])

hot_queries = int(NUM_QUERIES * HOT_TRAFFIC)
cold_queries = NUM_QUERIES - hot_queries
request_categories = np.concatenate([
    np.random.choice(list(hot_cats), hot_queries, replace=True),
    np.random.choice(list(cold_cats), cold_queries, replace=True)
])
request_product_ids = np.array([
    np.random.choice(category_products[cat]) for cat in request_categories
])
shuf_idx = np.random.permutation(NUM_QUERIES)
request_categories = request_categories[shuf_idx]
request_product_ids = request_product_ids[shuf_idx]

# 3. Query Types: Point lookup and range search (randomly mix)
query_types = np.random.choice(['point', 'range'], NUM_QUERIES, p=[0.85, 0.15])

# --- Data Structure Candidates ---
def build_hash(arr): return set(arr)
def build_sorted(arr): return np.array(arr)
def build_learned(arr):
    X = arr.astype(float)
    y = np.arange(len(arr), dtype=float)
    a, b = np.polyfit(X, y, 1)
    return (arr, a, b)

def point_query_hash(ds, key):
    return key in ds
def point_query_sorted(ds, key):
    idx = np.searchsorted(ds, key)
    return idx < len(ds) and ds[idx] == key
def point_query_learned(ds, key):
    arr, a, b = ds
    pred = int(np.clip(round(a*key + b), 0, len(arr)-1))
    for delta in range(-20, 21):
        idx = pred + delta
        if 0 <= idx < len(arr) and arr[idx] == key:
            return True
    idx = np.searchsorted(arr, key)
    return idx < len(arr) and arr[idx] == key

def range_query_sorted(ds, lo, hi):
    left = np.searchsorted(ds, lo, side='left')
    right = np.searchsorted(ds, hi, side='right')
    return right - left

# --- 4. Auto-Benchmarker ---
def benchmark_point(arr, keys):
    candidates = []
    if len(arr) > 0:
        hash_ds = build_hash(arr)
        t0 = time.perf_counter()
        found = sum(point_query_hash(hash_ds, k) for k in keys)
        t1 = time.perf_counter()
        candidates.append(('hash', t1-t0, found))
    sorted_ds = build_sorted(arr)
    t0 = time.perf_counter()
    found = sum(point_query_sorted(sorted_ds, k) for k in keys)
    t1 = time.perf_counter()
    candidates.append(('sorted', t1-t0, found))
    if len(arr) >= LEARNED_MIN_ITEMS:
        learned_ds = build_learned(arr)
        t0 = time.perf_counter()
        found = sum(point_query_learned(learned_ds, k) for k in keys)
        t1 = time.perf_counter()
        candidates.append(('learned', t1-t0, found))
    valid = [c for c in candidates if c[2] == len(keys)]
    if not valid:
        valid = candidates
    winner = min(valid, key=lambda x: x[1])
    return winner[0], {c[0]: c[1] for c in candidates}

def benchmark_range(arr, los, his):
    if len(arr) == 0:
        return 'none', {}
    sorted_ds = build_sorted(arr)
    t0 = time.perf_counter()
    found = sum(range_query_sorted(sorted_ds, lo, hi) for lo, hi in zip(los, his))
    t1 = time.perf_counter()
    return 'sorted', {'sorted': t1-t0}

# --- 5. Path Assignment ---
print("\n--- Auto-Benchmarking & Path Assignment ---")
assignment = {}
timings = {}
for cat in range(NUM_CATEGORIES):
    arr = category_products[cat]
    if len(arr) == 0:
        assignment[(cat, 'point')] = ('none', {})
        assignment[(cat, 'range')] = ('none', {})
        continue
    keys = np.random.choice(arr, min(len(arr), 100), replace=False)
    winner, times = benchmark_point(arr, keys)
    assignment[(cat, 'point')] = (winner, times)
    timings[(cat, 'point')] = times
    if len(arr) >= 2:
        los = np.random.choice(arr, 20, replace=True)
        his = [lo + np.random.randint(1, 1000) for lo in los]
        winner, times = benchmark_range(arr, los, his)
        assignment[(cat, 'range')] = (winner, times)
        timings[(cat, 'range')] = times

# --- 6. Save Paths ---
with open('auto_ds_assignments.pkl', 'wb') as f:
    pickle.dump(assignment, f)

print("Assignments saved: auto_ds_assignments.pkl\n")
print(f"Sample assignments (category 0): {assignment[(0, 'point')]}, {assignment[(0, 'range')]}")

# --- 7. Structure Caches (One per Category) ---
hash_cache = {}
sorted_cache = {}
learned_cache = {}

# --- 8. Serve Queries via Fastest Path, with caching per category ---
def serve_query(cat, qtype, *args):
    winner, _ = assignment.get((cat, qtype), ('none', {}))
    arr = category_products[cat]
    if winner == 'hash':
        if cat not in hash_cache:
            hash_cache[cat] = build_hash(arr)
        return point_query_hash(hash_cache[cat], *args)
    elif winner == 'sorted':
        if cat not in sorted_cache:
            sorted_cache[cat] = build_sorted(arr)
        if qtype == 'point':
            return point_query_sorted(sorted_cache[cat], *args)
        else:
            return range_query_sorted(sorted_cache[cat], *args)
    elif winner == 'learned':
        if cat not in learned_cache:
            learned_cache[cat] = build_learned(arr)
        return point_query_learned(learned_cache[cat], *args)
    return False

# --- 9. Classic DSA Structure Cache ---
def serve_query_classic(cat, qtype, *args):
    if cat not in sorted_cache:
        sorted_cache[cat] = category_products[cat]
    arr = sorted_cache[cat]
    if qtype == 'point':
        return point_query_sorted(arr, *args)
    else:
        return range_query_sorted(arr, *args)

# --- 10. Benchmark: Self-Optimizing Engine ---
print("\n--- Benchmark: Auto-Routing All Queries ---")
t0 = time.perf_counter()
hits = 0
for i in range(NUM_QUERIES):
    cat = request_categories[i]
    if query_types[i] == 'point':
        if serve_query(cat, 'point', request_product_ids[i]):
            hits += 1
    else:
        lo = request_product_ids[i]
        hi = lo + np.random.randint(1, 1000)
        found = serve_query(cat, 'range', lo, hi)
        if found > 0:
            hits += 1
t1 = time.perf_counter()
total_time = t1-t0
latency = 1000 * total_time / NUM_QUERIES

print(f"Total Queries: {NUM_QUERIES}")
print(f"Total Time: {total_time:.4f}s")
print(f"Avg Latency: {latency:.5f} ms")
print(f"Hit Rate: {100*hits/NUM_QUERIES:.2f}%")

# --- 11. Benchmark: Classic DSA (Sorted Array for All Queries) ---
print("\n--- Benchmark: Classic DSA (Sorted Array for All Queries) ---")
t0 = time.perf_counter()
classic_hits = 0
for i in range(NUM_QUERIES):
    cat = request_categories[i]
    if query_types[i] == 'point':
        if serve_query_classic(cat, 'point', request_product_ids[i]):
            classic_hits += 1
    else:
        lo = request_product_ids[i]
        hi = lo + np.random.randint(1, 1000)
        found = serve_query_classic(cat, 'range', lo, hi)
        if found > 0:
            classic_hits += 1
t1 = time.perf_counter()
classic_total_time = t1-t0
classic_latency = 1000 * classic_total_time / NUM_QUERIES

print(f"Total Queries: {NUM_QUERIES}")
print(f"Total Time: {classic_total_time:.4f}s")
print(f"Avg Latency: {classic_latency:.5f} ms")
print(f"Hit Rate: {100*classic_hits/NUM_QUERIES:.2f}%")

# --- 12. Comparison Table ---
speedup = classic_total_time / total_time if total_time > 0 else 0
print("\n========= BENCHMARK COMPARISON =========")
print(f"{'Method':<18}{'Total Time (s)':>18}{'Avg Latency (ms)':>22}{'Hit Rate (%)':>16}")
print('-'*70)
print(f"{'Classic DSA':<18}{classic_total_time:>18.4f}{classic_latency:>22.5f}{100*classic_hits/NUM_QUERIES:>16.2f}")
print(f"{'Self-Optimizing':<18}{total_time:>18.4f}{latency:>22.5f}{100*hits/NUM_QUERIES:>16.2f}")
print('-'*70)
print(f"Speedup: {speedup:.2f}x faster using Self-Optimizing Engine\n")

# --- 13. Visualize ---
labels = ['Classic DSA', 'Self-Optimizing']
times = [classic_total_time, total_time]
plt.bar(labels, times, color=['#6366f1', '#10b981'])
plt.ylabel("Total Query Time (seconds)")
plt.title("Classic vs Self-Optimizing Engine")
plt.show()
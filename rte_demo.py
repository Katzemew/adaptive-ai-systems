import numpy as np
import time
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import psutil
import os

process = psutil.Process(os.getpid())

NUM_PRODUCTS = 2_000_000
NUM_CATEGORIES = 2000
NUM_QUERIES = 500_000
HOT_FRACTION = 0.10
HOT_TRAFFIC = 0.80
LEARNED_MIN_ITEMS = 1000

np.random.seed(42)

product_ids = np.arange(10_000_000, 10_000_000 + NUM_PRODUCTS)
category_map = np.random.choice(range(NUM_CATEGORIES), NUM_PRODUCTS)
category_products = defaultdict(list)
for pid, cat in zip(product_ids, category_map):
    category_products[cat].append(pid)
for cat in category_products:
    category_products[cat] = np.sort(np.array(category_products[cat]))

# Hot categories logic
cat_counts = Counter(category_map)
cat_sorted = [x[0] for x in cat_counts.most_common()]
num_hot = int(NUM_CATEGORIES * HOT_FRACTION)
hot_cats = set(cat_sorted[:num_hot])
cold_cats = set(cat_sorted[num_hot:])

# Queries: 80% hot, 20% cold
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

# Adaptive structures
setup_start = time.perf_counter()
adaptive_structures = {}
hash_count, learned_count, classic_count = 0, 0, 0
for cat in range(NUM_CATEGORIES):
    arr = category_products[cat]
    if len(arr) == 0:
        adaptive_structures[cat] = ('empty', None)
    elif cat in hot_cats:
        adaptive_structures[cat] = ('hash', set(arr))
        hash_count += 1
    elif len(arr) >= LEARNED_MIN_ITEMS:
        X = arr.astype(float)
        y = np.arange(len(arr), dtype=float)
        a, b = np.polyfit(X, y, 1)
        adaptive_structures[cat] = ('learned', (arr, a, b))
        learned_count += 1
    else:
        adaptive_structures[cat] = ('classic', arr)
        classic_count += 1
setup_end = time.perf_counter()
setup_time = setup_end - setup_start

def classic_query(category, product_id):
    arr = category_products[category]
    idx = np.searchsorted(arr, product_id)
    return idx if idx < len(arr) and arr[idx] == product_id else -1

def ai_query(category, product_id):
    mode, data = adaptive_structures[category]
    if mode == 'empty':
        return -1, "empty"
    if mode == 'hash':
        return (product_id if product_id in data else -1), "hash"
    elif mode == 'learned':
        arr, a, b = data
        pred = int(np.clip(round(a*product_id + b), 0, len(arr)-1))
        for delta in range(-20, 21):
            idx = pred + delta
            if 0 <= idx < len(arr) and arr[idx] == product_id:
                return idx, "learned"
        # Fallback
        idx = np.searchsorted(arr, product_id)
        if idx < len(arr) and arr[idx] == product_id:
            return idx, "fallback"
        return -1, "fail"
    else:
        arr = data
        idx = np.searchsorted(arr, product_id)
        return (idx if idx < len(arr) and arr[idx] == product_id else -1), "classic"

def run_benchmark(method="classic"):
    cpu_start = psutil.cpu_percent(interval=None)
    mem_start = process.memory_info().rss
    t0 = time.perf_counter()
    hits = 0
    details = {"hash":0, "learned":0, "classic":0, "fallback":0, "empty":0, "fail":0}
    for cat, pid in zip(request_categories, request_product_ids):
        if method == "classic":
            found = classic_query(cat, pid)
            details["classic"] += 1
        elif method == "ai":
            found, used = ai_query(cat, pid)
            details[used] += 1
        elif method == "dynamic":
            if cat in hot_cats:
                found, used = ai_query(cat, pid)
                details[used] += 1
            else:
                found = classic_query(cat, pid)
                details["classic"] += 1
        else:
            found = -1
        if found != -1:
            hits += 1
    t1 = time.perf_counter()
    mem_end = process.memory_info().rss
    cpu_end = psutil.cpu_percent(interval=None)
    return {
        "total_time": t1-t0,
        "avg_latency": (t1-t0)/NUM_QUERIES,
        "mem_mb": (mem_end-mem_start)/1e6,
        "cpu_percent": cpu_end,
        "hit_rate": hits/NUM_QUERIES,
        "details": details
    }

# Run all benchmarks
print("\n[Pure Classic DSA Benchmark]")
classic_stats = run_benchmark(method="classic")

print("\n[Pure AI-Enhanced Benchmark]")
ai_stats = run_benchmark(method="ai")

print("\n[Dynamic Adaptive Engine Benchmark]")
dynamic_stats = run_benchmark(method="dynamic")

# Results Table
def details_breakdown(stats):
    d = stats["details"]
    return f"hash: {d.get('hash',0)}, learned: {d.get('learned',0)}, fallback: {d.get('fallback',0)}, classic: {d.get('classic',0)}, empty: {d.get('empty',0)}, fail: {d.get('fail',0)}"

print("\nBenchmark Results (Millions of Products, Retail Skew 80/20):")
print("----------------------------------------------------------")
print(f"{'Method':<16} {'Total Time (s)':<15} {'Avg Latency (ms)':<18} {'Peak Mem (MB)':<13} {'CPU (%)':<8} {'Hit Rate (%)':<12} {'Structure Use (count)'}")
print("----------------------------------------------------------")
print(f"{'Classic DSA':<16} {classic_stats['total_time']:<15.3f} {classic_stats['avg_latency']*1000:<18.5f} {classic_stats['mem_mb']:<13.3f} {classic_stats['cpu_percent']:<8.2f} {classic_stats['hit_rate']*100:<12.2f} {details_breakdown(classic_stats)}")
print(f"{'AI-Enhanced':<16} {ai_stats['total_time']:<15.3f} {ai_stats['avg_latency']*1000:<18.5f} {ai_stats['mem_mb']:<13.3f} {ai_stats['cpu_percent']:<8.2f} {ai_stats['hit_rate']*100:<12.2f} {details_breakdown(ai_stats)}")
print(f"{'Dynamic Eng.':<16} {dynamic_stats['total_time']:<15.3f} {dynamic_stats['avg_latency']*1000:<18.5f} {dynamic_stats['mem_mb']:<13.3f} {dynamic_stats['cpu_percent']:<8.2f} {dynamic_stats['hit_rate']*100:<12.2f} {details_breakdown(dynamic_stats)}")
print("----------------------------------------------------------")
print(f"Setup time: {setup_time:.3f}s | Structures: hash={hash_count}, learned={learned_count}, classic={classic_count}")

# Visualize breakdown if you like
import matplotlib.pyplot as plt
labels = ['Classic DSA', 'AI-Enhanced', 'Dynamic Eng.']
times = [classic_stats["total_time"], ai_stats["total_time"], dynamic_stats["total_time"]]
mems = [classic_stats["mem_mb"], ai_stats["mem_mb"], dynamic_stats["mem_mb"]]
hits = [classic_stats["hit_rate"]*100, ai_stats["hit_rate"]*100, dynamic_stats["hit_rate"]*100]
latency = [classic_stats["avg_latency"]*1000, ai_stats["avg_latency"]*1000, dynamic_stats["avg_latency"]*1000]
cpu = [classic_stats["cpu_percent"], ai_stats["cpu_percent"], dynamic_stats["cpu_percent"]]

plt.figure(figsize=(10,6))
plt.subplot(2,2,1)
plt.bar(labels, times, color=['#6366f1', '#10b981', '#f59e42'])
plt.ylabel("Total Query Time (s)")
plt.title("Query Time")
plt.subplot(2,2,2)
plt.bar(labels, mems, color=['#6366f1', '#10b981', '#f59e42'])
plt.ylabel("Peak Mem (MB)")
plt.title("Memory")
plt.subplot(2,2,3)
plt.bar(labels, latency, color=['#6366f1', '#10b981', '#f59e42'])
plt.ylabel("Avg Latency (ms)")
plt.title("Latency")
plt.subplot(2,2,4)
plt.bar(labels, hits, color=['#6366f1', '#10b981', '#f59e42'])
plt.ylabel("Hit Rate (%)")
plt.title("Hit Rate")
plt.tight_layout()
plt.show()
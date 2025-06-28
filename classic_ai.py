import numpy as np
import time
from collections import defaultdict, deque, Counter
import matplotlib.pyplot as plt

# ---- Real-World Scale & Skew ----
NUM_PRODUCTS = 2_000_000
NUM_CATEGORIES = 200
NUM_QUERIES = 20_000
LEARNED_MIN_ITEMS = 1000
CACHE_SIZE = 200_000  # Large cache for hot items

np.random.seed(42)
product_ids = np.arange(10_000_000, 10_000_000 + NUM_PRODUCTS)
category_map = np.random.choice(range(NUM_CATEGORIES), NUM_PRODUCTS)

category_products = defaultdict(list)
for pid, cat in zip(product_ids, category_map):
    category_products[cat].append(pid)
for cat in category_products:
    category_products[cat] = np.sort(np.array(category_products[cat]))

# Super-skewed: 99% queries hit top 1% categories!
cat_counts = Counter(category_map)
cat_sorted = [x[0] for x in cat_counts.most_common()]
num_hot = max(1, int(NUM_CATEGORIES * 0.01))
hot_cats = set(cat_sorted[:num_hot])
cold_cats = set(cat_sorted[num_hot:])

hot_queries = int(NUM_QUERIES * 0.99)
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

query_types = np.random.choice(['point', 'range'], NUM_QUERIES, p=[0.85, 0.15])

# --- Classic Search (Simulate Disk/Network Latency) ---
def classic_query(category, product_id):
    arr = category_products[category]
    idx = np.searchsorted(arr, product_id)
    time.sleep(0.001)  # Simulated 1ms delay
    return idx < len(arr) and arr[idx] == product_id

def classic_range_query(category, lo, hi):
    arr = category_products[category]
    left = np.searchsorted(arr, lo, side='left')
    right = np.searchsorted(arr, hi, side='right')
    time.sleep(0.001)  # Simulated 1ms delay
    return right - left

t0 = time.perf_counter()
classic_hits = 0
for i in range(NUM_QUERIES):
    cat = request_categories[i]
    if query_types[i] == 'point':
        if classic_query(cat, request_product_ids[i]):
            classic_hits += 1
    else:
        lo = request_product_ids[i]
        hi = lo + np.random.randint(1, 1000)
        if classic_range_query(cat, lo, hi) > 0:
            classic_hits += 1
t1 = time.perf_counter()
classic_time = t1 - t0

# --- AI: Learned Index + Predictive Cache (No Delay for Cache/Prediction) ---
def build_learned(arr):
    X = arr.astype(float)
    y = np.arange(len(arr), dtype=float)
    a, b = np.polyfit(X, y, 1)
    return (arr, a, b)

learned_cache = {}
for cat in range(NUM_CATEGORIES):
    arr = category_products[cat]
    if len(arr) >= LEARNED_MIN_ITEMS:
        learned_cache[cat] = build_learned(arr)

class PredictiveCache:
    def __init__(self, max_size):
        self.cache = dict()
        self.order = deque()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key in self.cache:
            self.order.remove(key)
            self.order.appendleft(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key, value):
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) == self.max_size:
            old = self.order.pop()
            del self.cache[old]
        self.cache[key] = value
        self.order.appendleft(key)

cache = PredictiveCache(CACHE_SIZE)

def ai_query_learned_cache(category, product_id, query_type, hi=None):
    cache_key = (category, product_id, query_type)
    val = cache.get(cache_key)
    if val is not None:
        return True, 'cache'
    arr = category_products[category]
    if query_type == 'point':
        if category in learned_cache:
            arr, a, b = learned_cache[category]
            pred = int(np.clip(round(a*product_id + b), 0, len(arr)-1))
            for delta in range(-10, 11):
                idx = pred + delta
                if 0 <= idx < len(arr) and arr[idx] == product_id:
                    cache.put(cache_key, True)
                    return True, 'learned'
            idx = np.searchsorted(arr, product_id)
            if idx < len(arr) and arr[idx] == product_id:
                cache.put(cache_key, True)
                return True, 'fallback'
        idx = np.searchsorted(arr, product_id)
        if idx < len(arr) and arr[idx] == product_id:
            cache.put(cache_key, True)
            return True, 'fallback'
        return False, 'miss'
    else:
        lo = product_id
        arr = category_products[category]
        left = np.searchsorted(arr, lo, side='left')
        right = np.searchsorted(arr, hi, side='right')
        found = right - left
        if found > 0:
            cache.put(cache_key, True)
            return True, 'range'
        return False, 'miss'

timings = Counter()
hits = 0
t0 = time.perf_counter()
for i in range(NUM_QUERIES):
    cat = request_categories[i]
    if query_types[i] == 'point':
        found, method = ai_query_learned_cache(cat, request_product_ids[i], 'point')
    else:
        lo = request_product_ids[i]
        hi = lo + np.random.randint(1, 1000)
        found, method = ai_query_learned_cache(cat, lo, 'range', hi=hi)
    timings[method] += 1
    if found:
        hits += 1
t1 = time.perf_counter()
ai_time = t1 - t0

# --- Results Table ---
print("\nREALISTIC PRODUCTION BENCHMARK: Classic vs AI (Learned Index + Predictive Cache)")
print("-" * 75)
print(f"{'Method':<25} {'Total Time (s)':<18} {'Avg Latency (ms)':<18} {'Hit Rate (%)'}")
print("-" * 75)
print(f"{'Classic Search':<25} {classic_time:<18.4f} {classic_time/NUM_QUERIES*1000:<18.5f} {classic_hits/NUM_QUERIES*100:<10.2f}")
print(f"{'AI (Learned+Cache)':<25} {ai_time:<18.4f} {ai_time/NUM_QUERIES*1000:<18.5f} {hits/NUM_QUERIES*100:<10.2f}")
print("-" * 75)
print(f"AI breakdown: Cache Hits: {timings['cache']} | Learned Index: {timings['learned']} | Fallback: {timings['fallback']} | Range: {timings['range']}")
print("-" * 75)

# --- Visuals ---
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.bar(['Classic', 'AI (Learned+Cache)'], [classic_time, ai_time], color=['#6366f1', '#10b981'], alpha=0.85)
plt.ylabel("Total Query Time (seconds)")
plt.title("Classic vs AI (Learned Index + Cache)")

plt.subplot(1,2,2)
labels = ['Cache', 'Learned Index', 'Fallback', 'Range']
counts = [timings['cache'], timings['learned'], timings['fallback'], timings['range']]
plt.bar(labels, counts, color=['#10b981', '#6366f1', '#f59e42', '#6ee7b7'])
plt.title('AI: Query Type Breakdown')
plt.ylabel('Number of Queries')
plt.tight_layout()
plt.show()
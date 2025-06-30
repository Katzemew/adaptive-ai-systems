import numpy as np
import time
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

np.random.seed(42)

NOISE = True

environments = [
    {
        "name": "Amazon",
        "cat_count": 500,
        "query_types": ['point', 'range'],
        "point_prob": 0.7,
        "range_prob": 0.3,
    },
    {
        "name": "Google",
        "cat_count": 10000,
        "query_types": ['point', 'semantic'],
        "point_prob": 0.6,
        "range_prob": 0.4,
    },
    {
        "name": "WhatsApp",
        "cat_count": 2000,
        "query_types": ['time', 'point'],
        "point_prob": 0.5,
        "range_prob": 0.5,
    }
]

sizes = [10_000, 100_000, 1_000_000]
blocks_per_env = 6   # for 5+ shifts
queries_per_block = 2500
results = []

def build_learned(arr):
    X = arr.astype(float)
    y = np.arange(len(arr), dtype=float)
    a, b = np.polyfit(X, y, 1)
    return (arr, a, b)

def describe_category_products(category_products, NUM_PRODUCTS, NUM_CATEGORIES, note=""):
    num_cats = len(category_products)
    avg_per_cat = np.mean([len(category_products[cat]) for cat in category_products])
    unique_vals = len(np.unique(np.concatenate(list(category_products.values()))))
    print(f"    [Meta] {note} Categories: {num_cats}/{NUM_CATEGORIES} | Avg products/cat: {avg_per_cat:.2f} | Unique products: {unique_vals}/{NUM_PRODUCTS}")

def shift_data(category_products, NUM_CATEGORIES, NUM_PRODUCTS):
    print("    [Data Drift] *** Category assignment and product order fully randomized! ***")
    new_assignment = np.random.choice(range(NUM_CATEGORIES), NUM_PRODUCTS)
    all_products = np.concatenate(list(category_products.values()))
    np.random.shuffle(all_products)
    new_cp = defaultdict(list)
    for pid, cat in zip(all_products, new_assignment):
        new_cp[cat].append(pid)
    for cat in new_cp:
        new_cp[cat] = np.sort(np.array(new_cp[cat]))
    return new_cp

def rebalance_structures(category_products, last_assignment=None):
    best_path = {}
    learned_models = {}
    hash_tables = {}
    changes = []
    for cat in category_products:
        arr = category_products[cat]
        if len(arr) == 0:
            continue
        if len(arr) > 20:
            learned_models[cat] = build_learned(arr)
        hash_tables[cat] = set(arr)
        for qtype in ['point', 'range','time','semantic']:
            timings = Counter()
            for _ in range(2):
                pid = np.random.choice(arr)
                hi = pid + np.random.randint(1, 1000) if qtype in ('range','time','semantic') else None
                ts = bench_structure(arr, pid, qtype, hi)
                for k,v in ts.items():
                    timings[k] += v
            if qtype == 'point':
                options = [k for k in ['hash','sorted','learned'] if k in timings]
                best = min(options, key=lambda x: timings[x])
                best_path[(cat, 'point')] = best
            else:
                best_path[(cat, qtype)] = 'sorted'
            # Track change
            if last_assignment is not None:
                prev = last_assignment.get((cat, qtype))
                if prev and prev != best_path[(cat, qtype)]:
                    changes.append((cat, qtype, prev, best_path[(cat, qtype)]))
    return best_path, learned_models, hash_tables, changes

def bench_structure(arr, pid, qtype, hi=None):
    timings = {}
    s = time.perf_counter()
    _ = pid in set(arr)
    timings['hash'] = (time.perf_counter() - s) * 1000
    s = time.perf_counter()
    idx = np.searchsorted(arr, pid)
    _ = idx < len(arr) and arr[idx] == pid
    timings['sorted'] = (time.perf_counter() - s) * 1000
    if len(arr) > 20:
        arr_, a, b = build_learned(arr)
        s = time.perf_counter()
        pred = int(np.clip(round(a * pid + b), 0, len(arr) - 1))
        found = False
        for delta in range(-10, 11):
            idx = pred + delta
            if 0 <= idx < len(arr) and arr[idx] == pid:
                found = True
                break
        timings['learned'] = (time.perf_counter() - s) * 1000
    if qtype in ('range','time','semantic'):
        hi = pid + np.random.randint(1, 1000)
        s = time.perf_counter()
        left = np.searchsorted(arr, pid, side='left')
        right = np.searchsorted(arr, hi, side='right')
        _ = right - left
        timings['range'] = (time.perf_counter() - s) * 1000
    return timings

for size in sizes:
    print(f"\n=== DATA SIZE: {size:,} ===")
    for env in environments:
        print(f"\n[ENV: {env['name']}]")
        NUM_PRODUCTS = size
        NUM_CATEGORIES = env['cat_count']
        NUM_BLOCKS = blocks_per_env

        if NOISE:
            products = np.arange(10_000_000, 10_000_000 + NUM_PRODUCTS) + np.random.randint(-50, 50, NUM_PRODUCTS)
        else:
            products = np.arange(10_000_000, 10_000_000 + NUM_PRODUCTS)
        products = np.unique(products)
        if len(products) < NUM_PRODUCTS:
            pad = np.arange(100_000_000, 100_000_000 + (NUM_PRODUCTS-len(products)))
            products = np.concatenate([products, pad])
        cats = np.random.choice(range(NUM_CATEGORIES), NUM_PRODUCTS)
        category_products = defaultdict(list)
        for pid, cat in zip(products, cats):
            category_products[cat].append(pid)
        for cat in category_products:
            category_products[cat] = np.sort(np.array(category_products[cat]))

        classic_sorted_products = {cat: arr.copy() for cat, arr in category_products.items()}
        classic_hash_tables = {cat: set(arr) for cat, arr in classic_sorted_products.items()}

        block_classic_sorted = []
        block_classic_hash = []
        block_classic_combo = []
        block_ai = []
        block_meta = []
        block_classic_sorted_acc = []
        block_classic_hash_acc = []
        block_classic_combo_acc = []
        block_ai_acc = []
        block_meta_acc = []

        meta_assignment = None

        for block in range(NUM_BLOCKS):
            print(f"  [Block {block+1}/{NUM_BLOCKS}]")
            if block > 0:
                print("   >>> DATA DRIFT OCCURRED <<<")
                describe_category_products(category_products, NUM_PRODUCTS, NUM_CATEGORIES, note="Before shift")
                category_products = shift_data(category_products, NUM_CATEGORIES, NUM_PRODUCTS)
                describe_category_products(category_products, NUM_PRODUCTS, NUM_CATEGORIES, note="After shift")
            else:
                describe_category_products(category_products, NUM_PRODUCTS, NUM_CATEGORIES, note="Initial state")
            # Meta-Learning assignment and changes tracking
            best_path, learned_models, hash_tables, changes = rebalance_structures(category_products, last_assignment=meta_assignment)
            if meta_assignment is None:
                print("    [Meta] Structure assignment for first block (showing first 5):")
            elif changes:
                print(f"    [Meta] Structure reassigned for {len(changes)} cat/qtype due to drift (showing first 5):")
            else:
                print("    [Meta] No reassignment needed after this drift.")
            shown = 0
            showlist = changes if changes else list(best_path.keys())
            for entry in showlist:
                if shown >= 5: break
                if changes:
                    cat, qtype, prev, now = entry
                    print(f"     - Cat {cat} ({qtype}): {prev} → {now}")
                else:
                    cat, qtype = entry if isinstance(entry, tuple) else (entry, 'point')
                    print(f"     - Cat {cat} ({qtype}): {best_path[(cat,qtype)]}")
                shown += 1
            meta_assignment = best_path.copy()

            qtypes = np.random.choice(
                env["query_types"], queries_per_block,
                p=[env["point_prob"], env["range_prob"]]
            )
            nonempty_cats = [cat for cat in range(NUM_CATEGORIES) if len(category_products[cat]) > 0]
            cats = np.random.choice(nonempty_cats, queries_per_block)
            pids = np.array([np.random.choice(category_products[cat]) for cat in cats])

            def classic_sorted(cat, pid, qtype, hi=None):
                arr = classic_sorted_products.get(cat, np.array([]))
                if qtype == 'point':
                    idx = np.searchsorted(arr, pid)
                    return idx < len(arr) and arr[idx] == pid
                else:
                    left = np.searchsorted(arr, pid, side='left')
                    right = np.searchsorted(arr, hi, side='right')
                    return right - left

            def classic_hash(cat, pid, qtype, hi=None):
                htable = classic_hash_tables.get(cat, set())
                if qtype == 'point':
                    return pid in htable
                else:
                    return None

            def classic_combo(cat, pid, qtype, hi=None):
                arr = classic_sorted_products.get(cat, np.array([]))
                htable = classic_hash_tables.get(cat, set())
                if qtype == 'point' and len(arr) > 800:
                    return pid in htable
                elif qtype == 'point':
                    idx = np.searchsorted(arr, pid)
                    return idx < len(arr) and arr[idx] == pid
                else:
                    left = np.searchsorted(arr, pid, side='left')
                    right = np.searchsorted(arr, hi, side='right')
                    return right - left

            def ai_fn(cat, pid, qtype, hi=None):
                arr = category_products[cat]
                if qtype == 'point' and cat in learned_models:
                    arr_, a, b = learned_models[cat]
                    pred = int(np.clip(round(a * pid + b), 0, len(arr) - 1))
                    for delta in range(-10, 11):
                        idx = pred + delta
                        if 0 <= idx < len(arr) and arr[idx] == pid:
                            return True
                    return False
                elif qtype == 'point':
                    idx = np.searchsorted(arr, pid)
                    return idx < len(arr) and arr[idx] == pid
                else:
                    left = np.searchsorted(arr, pid, side='left')
                    right = np.searchsorted(arr, hi, side='right')
                    return right - left

            def meta_fn(cat, pid, qtype, hi=None):
                arr = category_products[cat]
                method = best_path[(cat, qtype)]
                if method == 'hash':
                    return pid in hash_tables[cat]
                if method == 'sorted':
                    idx = np.searchsorted(arr, pid)
                    return idx < len(arr) and arr[idx] == pid
                if method == 'learned':
                    arr_, a, b = learned_models[cat]
                    pred = int(np.clip(round(a * pid + b), 0, len(arr) - 1))
                    for delta in range(-10, 11):
                        idx = pred + delta
                        if 0 <= idx < len(arr) and arr[idx] == pid:
                            return True
                    return False
                if method == 'range':
                    left = np.searchsorted(arr, pid, side='left')
                    right = np.searchsorted(arr, hi, side='right')
                    return right - left
                return False

            classic_sorted_acc = 0
            classic_hash_acc = 0
            classic_combo_acc = 0
            ai_acc = 0
            meta_acc = 0
            num_point_queries = 0

            t0 = time.perf_counter()
            for i in range(queries_per_block):
                cat, pid, qtype = cats[i], pids[i], qtypes[i]
                hi = pid + np.random.randint(1, 1000) if qtype != 'point' else None
                res = classic_sorted(cat, pid, qtype, hi)
                if qtype == 'point':
                    num_point_queries += 1
                    if res: classic_sorted_acc += 1
            block_classic_sorted.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            for i in range(queries_per_block):
                cat, pid, qtype = cats[i], pids[i], qtypes[i]
                hi = pid + np.random.randint(1, 1000) if qtype != 'point' else None
                res = classic_hash(cat, pid, qtype, hi)
                if qtype == 'point':
                    if res: classic_hash_acc += 1
            block_classic_hash.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            for i in range(queries_per_block):
                cat, pid, qtype = cats[i], pids[i], qtypes[i]
                hi = pid + np.random.randint(1, 1000) if qtype != 'point' else None
                res = classic_combo(cat, pid, qtype, hi)
                if qtype == 'point':
                    if res: classic_combo_acc += 1
            block_classic_combo.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            for i in range(queries_per_block):
                cat, pid, qtype = cats[i], pids[i], qtypes[i]
                hi = pid + np.random.randint(1, 1000) if qtype != 'point' else None
                res = ai_fn(cat, pid, qtype, hi)
                if qtype == 'point':
                    if res: ai_acc += 1
            block_ai.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            for i in range(queries_per_block):
                cat, pid, qtype = cats[i], pids[i], qtypes[i]
                hi = pid + np.random.randint(1, 1000) if qtype != 'point' else None
                res = meta_fn(cat, pid, qtype, hi)
                if qtype == 'point':
                    if res: meta_acc += 1
            block_meta.append(time.perf_counter() - t0)

            total = num_point_queries if num_point_queries else 1
            block_classic_sorted_acc.append(classic_sorted_acc/total)
            block_classic_hash_acc.append(classic_hash_acc/total)
            block_classic_combo_acc.append(classic_combo_acc/total)
            block_ai_acc.append(ai_acc/total)
            block_meta_acc.append(meta_acc/total)

        results.append({
            "env": env["name"],
            "size": size,
            "classic_sorted": block_classic_sorted,
            "classic_hash": block_classic_hash,
            "classic_combo": block_classic_combo,
            "ai": block_ai,
            "meta": block_meta,
            "classic_sorted_acc": block_classic_sorted_acc,
            "classic_hash_acc": block_classic_hash_acc,
            "classic_combo_acc": block_classic_combo_acc,
            "ai_acc": block_ai_acc,
            "meta_acc": block_meta_acc,
        })

# ==== PLOTS ====
for env in environments:
    plt.figure(figsize=(13,4))
    for size in sizes:
        found = [r for r in results if r['env']==env['name'] and r['size']==size][0]
        plt.subplot(1, len(sizes), sizes.index(size)+1)
        plt.plot(found['classic_sorted'], label="Classic-Sorted", marker='o')
        plt.plot(found['classic_hash'], label="Classic-Hash", marker='s')
        plt.plot(found['classic_combo'], label="Classic-Combo", marker='P')
        plt.plot(found['ai'], label="AI/Hybrid", marker='^')
        plt.plot(found['meta'], label="Meta-Learning", marker='*', color='#10b981')
        plt.title(f"{env['name']} — {size:,} Products\nData Drift: Yes")
        plt.xlabel("Block (Data Drift Step)")
        plt.ylabel("Time (s) per 2500 queries")
        plt.legend()
    plt.tight_layout()
    plt.show()

# ==== SUMMARY TABLES ====
for env in environments:
    print(f"\n--- {env['name']} ---")
    for size in sizes:
        found = [r for r in results if r['env']==env['name'] and r['size']==size][0]
        def avg(lst): return sum(lst)/len(lst)
        print(f"  Size {size:,} (Data Drift: Yes):")
        print(f"    Classic-Sorted: {avg(found['classic_sorted']):.5f}s  Acc: {avg(found['classic_sorted_acc']):.3f}")
        print(f"    Classic-Hash  : {avg(found['classic_hash']):.5f}s  Acc: {avg(found['classic_hash_acc']):.3f}")
        print(f"    Classic-Combo : {avg(found['classic_combo']):.5f}s  Acc: {avg(found['classic_combo_acc']):.3f}")
        print(f"    AI/Hybrid     : {avg(found['ai']):.5f}s  Acc: {avg(found['ai_acc']):.3f}")
        print(f"    Meta-Learning : {avg(found['meta']):.5f}s  Acc: {avg(found['meta_acc']):.3f}")
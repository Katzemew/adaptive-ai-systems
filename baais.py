import numpy as np
import random
import time
import matplotlib.pyplot as plt
import pandas as pd
from numba import njit, prange

# --- Data Generator ---
def generate_streaming_data(size, trend='mixed'):
    arr = np.zeros(size, dtype=np.int64)
    val = random.randint(0, 100)
    for i in range(size):
        if trend == 'up':
            val += random.randint(0, 3)
        elif trend == 'down':
            val -= random.randint(0, 3)
        elif trend == 'mixed':
            val += random.choice([-2, -1, 0, 1, 2])
        elif trend == 'burst':
            if i % 100 == 0:
                val += random.randint(-100, 100)
            else:
                val += random.choice([-1, 0, 1])
        arr[i] = val
    return arr

# --- Numba-Accelerated Sorting Algorithms ---

@njit
def selection_sort(a):
    n = len(a)
    res = a.copy()
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if res[j] < res[min_idx]:
                min_idx = j
        res[i], res[min_idx] = res[min_idx], res[i]
    return res

@njit
def insertion_sort(a):
    n = len(a)
    res = a.copy()
    for i in range(1, n):
        key = res[i]
        j = i - 1
        while j >= 0 and res[j] > key:
            res[j+1] = res[j]
            j -= 1
        res[j+1] = key
    return res

@njit
def merge(a, b):
    n, m = len(a), len(b)
    res = np.empty(n + m, dtype=a.dtype)
    i = j = k = 0
    while i < n and j < m:
        if a[i] < b[j]:
            res[k] = a[i]
            i += 1
        else:
            res[k] = b[j]
            j += 1
        k += 1
    while i < n:
        res[k] = a[i]
        i += 1; k += 1
    while j < m:
        res[k] = b[j]
        j += 1; k += 1
    return res

@njit
def merge_sort(a):
    n = len(a)
    if n <= 1:
        return a.copy()
    mid = n // 2
    left = merge_sort(a[:mid])
    right = merge_sort(a[mid:])
    return merge(left, right)

@njit
def quick_sort(a):
    n = len(a)
    if n <= 1:
        return a.copy()
    pivot = a[0]
    cnt_left = 0
    for x in a[1:]:
        if x < pivot:
            cnt_left += 1
    left = np.empty(cnt_left, dtype=a.dtype)
    right = np.empty(n - 1 - cnt_left, dtype=a.dtype)
    i_left = i_right = 0
    for x in a[1:]:
        if x < pivot:
            left[i_left] = x
            i_left += 1
        else:
            right[i_right] = x
            i_right += 1
    left_sorted = quick_sort(left) if cnt_left else np.empty(0, dtype=a.dtype)
    right_sorted = quick_sort(right) if n - 1 - cnt_left else np.empty(0, dtype=a.dtype)
    out = np.empty(n, dtype=a.dtype)
    out[:len(left_sorted)] = left_sorted
    out[len(left_sorted)] = pivot
    out[len(left_sorted)+1:] = right_sorted
    return out

@njit
def heapify(a, n, i):
    largest = i
    l = 2*i+1
    r = 2*i+2
    if l < n and a[l] > a[largest]:
        largest = l
    if r < n and a[r] > a[largest]:
        largest = r
    if largest != i:
        a[i], a[largest] = a[largest], a[i]
        heapify(a, n, largest)

@njit
def heap_sort(a):
    n = len(a)
    res = a.copy()
    for i in range(n//2-1, -1, -1):
        heapify(res, n, i)
    for i in range(n-1, 0, -1):
        res[i], res[0] = res[0], res[i]
        heapify(res, i, 0)
    return res

@njit
def find_runs(a):
    n = len(a)
    starts = np.zeros(n, dtype=np.int64)
    ends = np.zeros(n, dtype=np.int64)
    run_count = 0
    i = 0
    while i < n:
        start = i
        if i + 1 >= n:
            ends[run_count] = i
            starts[run_count] = start
            run_count += 1
            break
        if a[i] <= a[i+1]:
            while i+1 < n and a[i] <= a[i+1]:
                i += 1
        else:
            while i+1 < n and a[i] >= a[i+1]:
                i += 1
        ends[run_count] = i
        starts[run_count] = start
        run_count += 1
        i += 1
    out = a.copy()
    temp = np.empty_like(a)
    curr_starts = starts[:run_count].copy()
    curr_ends = ends[:run_count].copy()
    curr_run_count = run_count
    while curr_run_count > 1:
        next_starts = np.zeros_like(curr_starts)
        next_ends = np.zeros_like(curr_ends)
        write_ptr = 0
        i = 0
        new_run_count = 0
        while i < curr_run_count:
            s1, e1 = curr_starts[i], curr_ends[i]
            if i+1 < curr_run_count:
                s2, e2 = curr_starts[i+1], curr_ends[i+1]
                merged = merge(out[s1:e1+1], out[s2:e2+1])
                temp[write_ptr:write_ptr+len(merged)] = merged
                next_starts[new_run_count] = write_ptr
                next_ends[new_run_count] = write_ptr + len(merged) - 1
                write_ptr += len(merged)
                new_run_count += 1
                i += 2
            else:
                temp[write_ptr:write_ptr+e1-s1+1] = out[s1:e1+1]
                next_starts[new_run_count] = write_ptr
                next_ends[new_run_count] = write_ptr + (e1-s1)
                write_ptr += e1-s1+1
                new_run_count += 1
                i += 1
        out, temp = temp, out
        curr_starts = next_starts[:new_run_count].copy()
        curr_ends = next_ends[:new_run_count].copy()
        curr_run_count = new_run_count
    return out

# --- Searching Algorithms (Numba) ---
@njit
def linear_search(a, target):
    for i in range(len(a)):
        if a[i] == target:
            return i
    return -1

@njit
def binary_search(a, target):
    left, right = 0, len(a) - 1
    while left <= right:
        mid = (left + right) // 2
        if a[mid] == target:
            return mid
        elif a[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

@njit
def bisect_search(a, target):
    left, right = 0, len(a)
    while left < right:
        mid = (left + right) // 2
        if a[mid] < target:
            left = mid + 1
        else:
            right = mid
    if left < len(a) and a[left] == target:
        return left
    return -1

# --- Pattern Index & Fast Pattern-Aware Search (Numba, Numpy Arrays) ---
@njit
def build_pattern_index_with_min_max_fast(a):
    n = len(a)
    starts = np.zeros(n, dtype=np.int64)
    ends   = np.zeros(n, dtype=np.int64)
    pats   = np.zeros(n, dtype=np.int64)  # 1: inc, 2: dec, 0: const
    mins   = np.zeros(n, dtype=a.dtype)
    maxs   = np.zeros(n, dtype=a.dtype)
    run_count = 0

    i = 0
    while i < n:
        start = i
        if i + 1 >= n:
            pat = 0
        elif a[i] < a[i+1]:
            pat = 1
            while i+1 < n and a[i] <= a[i+1]:
                i += 1
        elif a[i] > a[i+1]:
            pat = 2
            while i+1 < n and a[i] >= a[i+1]:
                i += 1
        else:
            pat = 0
            while i+1 < n and a[i] == a[i+1]:
                i += 1
        end = i
        seg = a[start:end+1]
        starts[run_count] = start
        ends[run_count] = end
        pats[run_count] = pat
        mins[run_count] = np.min(seg)
        maxs[run_count] = np.max(seg)
        run_count += 1
        i += 1

    return starts[:run_count], ends[:run_count], pats[:run_count], mins[:run_count], maxs[:run_count]

@njit
def pattern_search_fast(a, starts, ends, pats, mins, maxs, target):
    for idx in range(len(starts)):
        if not (mins[idx] <= target <= maxs[idx]):
            continue
        start = starts[idx]
        end   = ends[idx]
        pat   = pats[idx]
        if pat == 1:
            l, r = start, end
            while l <= r:
                m = (l + r) // 2
                if a[m] == target:
                    return m
                elif a[m] < target:
                    l = m + 1
                else:
                    r = m - 1
        elif pat == 2:
            l, r = start, end
            while l <= r:
                m = (l + r) // 2
                if a[m] == target:
                    return m
                elif a[m] > target:
                    l = m + 1
                else:
                    r = m - 1
        else:
            if a[start] == target:
                return start
    return -1

# --- Batch Search (Numba Parallel) ---
@njit(parallel=True)
def batch_linear_search(a, targets):
    results = np.empty(len(targets), dtype=np.int64)
    for i in prange(len(targets)):
        results[i] = linear_search(a, targets[i])
    return results

@njit(parallel=True)
def batch_binary_search(sorted_a, targets):
    results = np.empty(len(targets), dtype=np.int64)
    for i in prange(len(targets)):
        results[i] = binary_search(sorted_a, targets[i])
    return results

@njit(parallel=True)
def batch_bisect_search(sorted_a, targets):
    results = np.empty(len(targets), dtype=np.int64)
    for i in prange(len(targets)):
        results[i] = bisect_search(sorted_a, targets[i])
    return results

@njit(parallel=True)
def batch_pattern_search_fast(a, starts, ends, pats, mins, maxs, targets):
    results = np.empty(len(targets), dtype=np.int64)
    for i in prange(len(targets)):
        results[i] = pattern_search_fast(a, starts, ends, pats, mins, maxs, targets[i])
    return results

# --- Benchmarking ---
def benchmark_all(a, batch_targets=None):
    size = len(a)
    target = a[size // 2]

    sort_results = {}
    search_results = {}

    t0 = time.time(); selection_sort(a); sort_results['Selection Sort'] = time.time() - t0
    t = time.time(); insertion_sort(a); sort_results['Insertion Sort'] = time.time() - t
    t = time.time(); merge_sort(a); sort_results['Merge Sort'] = time.time() - t
    t = time.time(); quick_sort(a); sort_results['Quick Sort'] = time.time() - t
    t = time.time(); heap_sort(a); sort_results['Heap Sort'] = time.time() - t
    t = time.time(); find_runs(a); sort_results['Pattern-Aware Sort'] = time.time() - t

    sorted_arr = merge_sort(a)
    starts, ends, pats, mins, maxs = build_pattern_index_with_min_max_fast(a)

    t = time.time(); linear_search(a, target); search_results['Linear Search'] = time.time() - t
    t = time.time(); binary_search(sorted_arr, target); search_results['Binary Search'] = time.time() - t
    t = time.time(); bisect_search(sorted_arr, target); search_results['Bisect Search'] = time.time() - t
    t = time.time(); pattern_search_fast(a, starts, ends, pats, mins, maxs, target); search_results['Pattern-aware Search'] = time.time() - t

    # Batch search for all algorithms (parallel, 1000 queries)
    if batch_targets is not None:
        t = time.time()
        _ = batch_linear_search(a, batch_targets)
        search_results['Batch Linear Search (parallel)'] = time.time() - t

        t = time.time()
        _ = batch_binary_search(sorted_arr, batch_targets)
        search_results['Batch Binary Search (parallel)'] = time.time() - t

        t = time.time()
        _ = batch_bisect_search(sorted_arr, batch_targets)
        search_results['Batch Bisect Search (parallel)'] = time.time() - t

        t = time.time()
        _ = batch_pattern_search_fast(a, starts, ends, pats, mins, maxs, batch_targets)
        search_results['Batch Pattern-aware Search (parallel)'] = time.time() - t

    return sort_results, search_results

def run_benchmarks(trend='burst'):
    sizes = [1000, 10000, 100000]
    batch_size = 1000  # number of targets in batch search
    sort_algos = ['Selection Sort', 'Insertion Sort', 'Merge Sort', 'Quick Sort', 'Heap Sort', 'Pattern-Aware Sort']
    search_algos = [
        'Linear Search', 'Binary Search', 'Bisect Search', 'Pattern-aware Search',
        'Batch Linear Search (parallel)', 'Batch Binary Search (parallel)',
        'Batch Bisect Search (parallel)', 'Batch Pattern-aware Search (parallel)'
    ]

    sort_times = {algo:[] for algo in sort_algos}
    search_times = {algo:[] for algo in search_algos}

    # warmup Numba to avoid JIT compile time in measurement
    _ = generate_streaming_data(32)
    dummy = np.arange(32)
    selection_sort(dummy)
    insertion_sort(dummy)
    merge_sort(dummy)
    quick_sort(dummy)
    heap_sort(dummy)
    find_runs(dummy)
    build_pattern_index_with_min_max_fast(dummy)
    pattern_search_fast(dummy, *build_pattern_index_with_min_max_fast(dummy), 5)
    batch_pattern_search_fast(dummy, *build_pattern_index_with_min_max_fast(dummy), np.arange(5))
    batch_linear_search(dummy, np.arange(5))
    batch_binary_search(dummy, np.arange(5))
    batch_bisect_search(dummy, np.arange(5))
    linear_search(dummy, 5)
    binary_search(dummy, 5)
    bisect_search(dummy, 5)

    for size in sizes:
        print(f"Benchmarking for size={size}")
        arr = generate_streaming_data(size, trend=trend)
        batch_targets = arr[np.random.randint(0, size, batch_size)]  # random valid targets
        sort_res, search_res = benchmark_all(arr, batch_targets)
        for algo in sort_algos:
            sort_times[algo].append(sort_res.get(algo, None))
        for algo in search_algos:
            search_times[algo].append(search_res.get(algo, None))

    # Plotting
    plt.figure(figsize=(14,6))
    for algo in sort_algos:
        times = sort_times[algo]
        if any(t is not None for t in times):
            plt.plot(sizes, [t if t is not None else float('nan') for t in times], marker='o', label=algo)
    plt.title('Sorting Algorithm Benchmark (Numba Optimized)')
    plt.xlabel('Array Size')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14,6))
    for algo in search_algos:
        times = search_times[algo]
        if any(t is not None for t in times):
            plt.plot(sizes, [t if t is not None else float('nan') for t in times], marker='o', label=algo)
    plt.title('Searching Algorithm Benchmark (Numba Optimized, Batch & Single)')
    plt.xlabel('Array Size')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    sort_df = pd.DataFrame(sort_times, index=[str(s) for s in sizes])
    print("\n--- Sorting Algorithm Benchmarks (time in seconds) ---")
    print(sort_df.to_string(float_format="{:0.6f}".format))

    search_df = pd.DataFrame(search_times, index=[str(s) for s in sizes])
    print("\n--- Searching Algorithm Benchmarks (time in seconds) ---")
    print(search_df.to_string(float_format="{:0.6f}".format))

if __name__ == '__main__':
    run_benchmarks(trend='burst')
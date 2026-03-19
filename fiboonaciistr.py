import time
import sys
import functools
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
 
sys.setrecursionlimit(10000)
 
# ──────────────────────────────────────────────
# Implementations
# ──────────────────────────────────────────────
 
def fib_recursive(n):
    """Naive recursive — exponential O(2^n) time."""
    if n <= 1:
        return n
    return fib_recursive(n - 1) + fib_recursive(n - 2)
 
 
@functools.lru_cache(maxsize=None)
def fib_memoized(n):
    """Memoized recursive — O(n) time, O(n) space."""
    if n <= 1:
        return n
    return fib_memoized(n - 1) + fib_memoized(n - 2)
 
 
def fib_iterative(n):
    """Bottom-up iterative — O(n) time, O(1) space."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
 
 
# ──────────────────────────────────────────────
# Benchmarking
# ──────────────────────────────────────────────
 
def benchmark(func, n, repeats=5):
    """Run func(n) multiple times and return mean elapsed time in seconds."""
    times = []
    for _ in range(repeats):
        # Clear lru_cache between runs for memoized so each call is fair
        if hasattr(func, 'cache_clear'):
            func.cache_clear()
        start = time.perf_counter()
        func(n)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return np.mean(times)
 
 
def run_benchmarks():
    # Recursive gets slow fast — cap at 35
    recursive_ns  = list(range(5, 36, 5))
    fast_ns        = list(range(5, 501, 25))
 
    print("\n" + "═" * 58)
    print("  Fibonacci Algorithm Benchmarker")
    print("═" * 58)
 
    # ── Recursive
    print("\n📌 Recursive (naive)  [capped at n=35]")
    print(f"  {'n':>6}  {'time (ms)':>12}  {'fib(n)':>20}")
    print("  " + "─" * 44)
    rec_times = []
    for n in recursive_ns:
        t = benchmark(fib_recursive, n, repeats=3) * 1000  # ms
        val = fib_recursive(n)
        rec_times.append(t)
        print(f"  {n:>6}  {t:>11.4f}ms  {val:>20,}")
 
    # ── Memoized
    print("\n📌 Memoized (lru_cache)")
    print(f"  {'n':>6}  {'time (µs)':>12}  {'fib(n)':>20}")
    print("  " + "─" * 44)
    memo_times = []
    for n in fast_ns:
        t = benchmark(fib_memoized, n, repeats=5) * 1_000_000  # µs
        fib_memoized.cache_clear()
        val = fib_memoized(n)
        memo_times.append(t)
        if n % 100 == 5 or n <= 30:
            print(f"  {n:>6}  {t:>10.2f}µs  {val:>20,}")
 
    # ── Iterative
    print("\n📌 Iterative (bottom-up)")
    print(f"  {'n':>6}  {'time (µs)':>12}  {'fib(n)':>20}")
    print("  " + "─" * 44)
    iter_times = []
    for n in fast_ns:
        t = benchmark(fib_iterative, n, repeats=5) * 1_000_000  # µs
        val = fib_iterative(n)
        iter_times.append(t)
        if n % 100 == 5 or n <= 30:
            print(f"  {n:>6}  {t:>10.2f}µs  {val:>20,}")
 
    print("\n" + "═" * 58)
    print(f"  Fastest for n=500:  Iterative  ({iter_times[-1]:.2f} µs)")
    print(f"  Memoized overhead:  {memo_times[-1]:.2f} µs  (cache cold)")
    print("═" * 58 + "\n")
 
    return recursive_ns, rec_times, fast_ns, memo_times, iter_times
 
 
# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────
 
def plot_results(recursive_ns, rec_times, fast_ns, memo_times, iter_times):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#0d1117')
 
    PURPLE = '#a78bfa'
    TEAL   = '#34d399'
    CORAL  = '#fb7185'
    BG     = '#0d1117'
    PANEL  = '#161b22'
    GRID   = '#21262d'
    TEXT   = '#c9d1d9'
    MUTED  = '#8b949e'
 
    for ax in axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(color=GRID, linewidth=0.5, linestyle='--')
 
    # ── Left: Recursive exponential blowup
    ax1 = axes[0]
    ax1.plot(recursive_ns, rec_times, color=CORAL, linewidth=2,
             marker='o', markersize=5, label='Recursive O(2ⁿ)')
 
    # Overlay theoretical 2^n curve (scaled)
    theory_x = np.linspace(5, 35, 200)
    scale = rec_times[-1] / (2 ** 35)
    theory_y = scale * 2 ** theory_x
    ax1.plot(theory_x, theory_y, color=CORAL, linewidth=1,
             linestyle=':', alpha=0.4, label='2ⁿ (theoretical)')
 
    ax1.set_title('Recursive — Exponential Growth', fontsize=11, pad=10)
    ax1.set_xlabel('n', fontsize=10)
    ax1.set_ylabel('Time (ms)', fontsize=10)
    ax1.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
 
    # ── Right: Memoized vs Iterative
    ax2 = axes[1]
    ax2.plot(fast_ns, memo_times, color=PURPLE, linewidth=2,
             marker='o', markersize=3, label='Memoized O(n)')
    ax2.plot(fast_ns, iter_times, color=TEAL, linewidth=2,
             marker='s', markersize=3, label='Iterative O(n)')
 
    ax2.set_title('Memoized vs Iterative (n up to 500)', fontsize=11, pad=10)
    ax2.set_xlabel('n', fontsize=10)
    ax2.set_ylabel('Time (µs)', fontsize=10)
    ax2.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
 
    # Annotation
    ax2.annotate('Iterative uses\nO(1) space →',
                 xy=(fast_ns[-1], iter_times[-1]),
                 xytext=(fast_ns[-1] - 120, iter_times[-1] + max(iter_times) * 0.35),
                 fontsize=8, color=TEAL,
                 arrowprops=dict(arrowstyle='->', color=TEAL, lw=1))
 
    fig.suptitle('Fibonacci Algorithm Benchmarker', fontsize=14,
                 color=TEXT, y=1.01)
    plt.tight_layout()
 
    output_path = 'benchmark_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    print(f"  Chart saved → {output_path}\n")
    plt.show()
 
 
# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
 
if __name__ == '__main__':
    recursive_ns, rec_times, fast_ns, memo_times, iter_times = run_benchmarks()
    plot_results(recursive_ns, rec_times, fast_ns, memo_times, iter_times)
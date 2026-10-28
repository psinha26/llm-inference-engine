#!/usr/bin/env python3
"""Benchmark the speedup ladder, thread scaling and memory on this machine.

Each configuration runs N times (default 5) and the median is reported. Every step adds one
optimisation on top of the previous one, so the table shows what each change is worth.

  python3 bench/run_bench.py --engine build/engine --model-dir models/SmolLM2-360M [--threads 4]

The model dir must contain model.bin, model-tiled.bin and model-int8-tiled.bin
(scripts/setup.sh makes all three; bench/make_shaped_model.py + convert.py work too).
Writes bench/results/<machine>.json, .md and two charts (if matplotlib is installed).

Every finished run is cached in bench/results/<machine>.cache.json, so an interrupted benchmark
resumes where it stopped. --budget N stops starting new runs after N seconds (exit code 3 means
"not finished yet, run me again").

Optional llama.cpp comparison on the same model and quantization (Q8_0 = 32-value int8 blocks):
  --llama-bench /path/to/llama-bench --gguf models/SmolLM2-360M/model-q8_0.gguf
"""
import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPT = ",".join(str(t) for t in range(1, 9))  # 8 prompt tokens (ids are irrelevant to speed)
GEN = 16                                         # generated tokens per run (keeps the O(n^2) baseline tractable)
STATS = re.compile(r"prefill ([\d.]+) tok/s \| decode ([\d.]+) tok/s.*peak mem (\d+) MB")


def machine_info():
    cpu = platform.processor() or platform.machine()
    ram = ""
    try:
        if sys.platform == "darwin":
            cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            ram = f"{int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'])) / 2**30:.0f} GB"
        else:
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
            ram = f"{os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30:.0f} GB"
    except Exception:
        pass
    return {"cpu": cpu, "cores": os.cpu_count(), "ram": ram, "os": f"{platform.system()} {platform.release()}"}


class OutOfBudget(Exception):
    pass


CACHE, CACHE_PATH, DEADLINE = {}, None, float("inf")


def run(engine, model, args, runs):
    results = []
    for i in range(runs):
        key = f"{Path(model).name} {' '.join(args)} #{i}"
        if key not in CACHE:
            if time.time() > DEADLINE:
                raise OutOfBudget()
            cmd = [engine, "--model", model, "--tokens", PROMPT, "--max-tokens", str(GEN), "--quiet", "--ignore-eos",
                   *args]
            out = subprocess.run(cmd, capture_output=True, text=True, check=True).stderr
            m = STATS.search(out)
            CACHE[key] = (float(m.group(1)), float(m.group(2)), int(m.group(3)))
            CACHE_PATH.write_text(json.dumps(CACHE, indent=1))
        results.append(CACHE[key])
    return {
        "prefill": statistics.median(r[0] for r in results),
        "decode": statistics.median(r[1] for r in results),
        "decode_runs": [r[1] for r in results],
        "peak_mb": max(r[2] for r in results),
    }


def llama_cpp(binary, gguf, threads, runs):
    out = subprocess.run([binary, "-m", gguf, "-p", "0", "-n", str(GEN), "-t", str(threads), "-r", str(runs),
                          "-o", "json"], capture_output=True, text=True, check=True).stdout
    return json.loads(out)[0]["avg_ts"]


def charts(res, out_base):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping charts")
        return
    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    blue, orange = "#2a78d6", "#eb6834"  # categorical slots 1-2, validated for CVD separation
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": grid, "axes.labelcolor": ink2, "xtick.color": ink2,
                         "ytick.color": ink2, "figure.facecolor": surface, "axes.facecolor": surface})

    # Speedup ladder: one series, so one hue and no legend; the speedup is labeled on each bar.
    ladder = res["ladder"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    names = [s["name"] for s in ladder][::-1]
    vals = [s["decode"] for s in ladder][::-1]
    base = ladder[0]["decode"]
    ax.barh(names, vals, color=blue, height=0.62)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, i, f"{v:.1f} tok/s  ({v / base:.0f}×)", va="center", color=ink, fontsize=10)
    ax.set_xlim(0, max(vals) * 1.5)
    ax.set_xlabel("decode tokens/sec (median of runs)")
    ax.grid(axis="x", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title("Speedup ladder", loc="left", color=ink, fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{out_base}-ladder.png", dpi=160)

    # Thread scaling: two series, legend plus direct labels at the line ends.
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for key, color, label in (("int8", blue, "int8, tiled"), ("fp32", orange, "fp32, tiled")):
        pts = res["scaling"][key]
        xs, ys = [p["threads"] for p in pts], [p["decode"] for p in pts]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=7, label=label)
        ax.annotate(f"{ys[-1]:.0f}", (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points", va="center",
                    color=ink, fontsize=10)
    ax.set_xticks([p["threads"] for p in res["scaling"]["int8"]])
    ax.set_xlabel("threads")
    ax.set_ylabel("decode tokens/sec")
    ax.set_ylim(0, None)
    ax.grid(axis="y", color=grid, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, labelcolor=ink)
    ax.set_title("Thread scaling", loc="left", color=ink, fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{out_base}-threads.png", dpi=160)
    print(f"charts: {out_base}-ladder.png, {out_base}-threads.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=str(ROOT / "build" / "engine"))
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--threads", type=int, default=0, help="threads for steps 2+ (default: all cores, max 8)")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--baseline-runs", type=int, default=3, help="runs for the slow no-KV-cache baseline")
    ap.add_argument("--machine", default="", help="label for the results file")
    ap.add_argument("--budget", type=float, default=0, help="seconds before pausing (resume by re-running)")
    ap.add_argument("--llama-bench")
    ap.add_argument("--gguf")
    a = ap.parse_args()
    info = machine_info()
    print(f"machine: {info}")
    t_start = time.time()
    global CACHE, CACHE_PATH, DEADLINE
    label = a.machine or re.sub(r"[^a-z0-9]+", "-", info["cpu"].lower()).strip("-")
    out_dir = ROOT / "bench" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    CACHE_PATH = out_dir / f"{label}.cache.json"
    if CACHE_PATH.exists():
        CACHE = json.loads(CACHE_PATH.read_text())
    if a.budget:
        DEADLINE = t_start + a.budget
    try:
        res = measure(a, info)
    except OutOfBudget:
        print(f"paused after {time.time() - t_start:.0f}s with {len(CACHE)} runs cached; run again to continue")
        sys.exit(3)
    report(a, res, info, out_dir / label)


def measure(a, info):
    md = Path(a.model_dir)
    fp32, tiled, int8 = str(md / "model.bin"), str(md / "model-tiled.bin"), str(md / "model-int8-tiled.bin")

    # Thread scaling first: it decides the thread count for the ladder.
    max_t = os.cpu_count() or 4
    counts = [t for t in (1, 2, 4, 8, 16) if t <= max_t]
    scaling = {}
    for key, model in (("int8", int8), ("fp32", tiled)):
        scaling[key] = []
        for t in counts:
            r = run(a.engine, model, ["--kernel", "blocked", "--threads", str(t)], a.runs)
            scaling[key].append({"threads": t, **r})
            print(f"  scaling {key} {t} threads: {r['decode']:.1f} tok/s")
    threads = a.threads or min(os.cpu_count() or 4, 8)

    steps = [
        ("0. Naive: fp32, 1 thread, no KV cache", fp32, ["--kernel", "naive", "--no-kv-cache"], a.baseline_runs),
        ("1. + KV cache", fp32, ["--kernel", "naive"], a.runs),
        (f"2. + thread pool ({threads} thread{'s' if threads > 1 else ''})", fp32, ["--kernel", "naive", "--threads", str(threads)], a.runs),
        (f"3. + SIMD ({'NEON' if platform.machine() in ('arm64', 'aarch64') else 'AVX2'})", fp32,
         ["--kernel", "simd", "--threads", str(threads)], a.runs),
        ("4. + tiled weight layout", tiled, ["--kernel", "blocked", "--threads", str(threads)], a.runs),
        ("5. + int8 weights", int8, ["--kernel", "blocked", "--threads", str(threads)], a.runs),
    ]
    ladder = []
    for name, model, args, runs in steps:
        r = run(a.engine, model, args, runs)
        ladder.append({"name": name, "runs": runs, **r})
        print(f"  {name}: decode {r['decode']:.2f} tok/s, prefill {r['prefill']:.2f} tok/s, peak {r['peak_mb']} MB")

    llama = None
    if a.llama_bench and a.gguf:
        llama = llama_cpp(a.llama_bench, a.gguf, threads, a.runs)
        print(f"  llama.cpp Q8_0, {threads} threads: {llama:.1f} tok/s")

    return {"machine": info, "prompt_tokens": 8, "generated_tokens": GEN, "threads": threads, "ladder": ladder,
            "scaling": scaling, "llama_cpp_q8_0": llama, "counts": counts}


def report(a, res, info, base):
    ladder, scaling, threads, llama, counts = (res["ladder"], res["scaling"], res["threads"], res["llama_cpp_q8_0"],
                                               res.pop("counts"))
    base.with_suffix(".json").write_text(json.dumps(res, indent=2))

    b = ladder[0]["decode"]
    lines = [f"Machine: {info['cpu']}, {info['cores']} cores, {info['ram']} RAM, {info['os']}. "
             f"8-token prompt, {GEN} generated tokens, median of {a.runs} runs "
             f"({a.baseline_runs} for the baseline).", "",
             "| Step | Decode tok/s | Prefill tok/s | Speedup vs naive | Peak memory |", "|---|---|---|---|---|"]
    for s in ladder:
        lines.append(f"| {s['name']} | {s['decode']:.1f} | {s['prefill']:.1f} | {s['decode'] / b:.1f}× | "
                     f"{s['peak_mb']} MB |")
    if llama:
        lines.append(f"| llama.cpp Q8_0 (same model, {threads} threads) | {llama:.1f} | | | |")
    best = max(scaling["int8"], key=lambda p: p["decode"])
    lines += ["", f"Best configuration: int8 + tiled at {best['threads']} thread{'s' if best['threads'] > 1 else ''}, "
              f"**{best['decode']:.1f} tok/s** decode ({best['decode'] / b:.0f}× the naive baseline), "
              f"{best['peak_mb']} MB peak memory vs {ladder[0]['peak_mb']} MB for fp32."]
    lines += ["", "| Threads | " + " | ".join(str(p["threads"]) for p in scaling["int8"]) + " |",
              "|---" * (len(counts) + 1) + "|"]
    for key in ("int8", "fp32"):
        lines.append(f"| {key} decode tok/s | " + " | ".join(f"{p['decode']:.1f}" for p in scaling[key]) + " |")
    base.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    charts(res, str(base))


if __name__ == "__main__":
    main()

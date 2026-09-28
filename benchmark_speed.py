# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.2",
#     "transformers>=4.51",
#     "accelerate>=0.30",
#     "jinja2",
# ]
# ///
"""
Latency of the four toy decision models on the same request.

Every model answers the same support ticket with N questions (1, 10, 100 by
default). Each timed call uses a fresh ticket, so nothing is reused from the
previous call; 04 is timed both cold (empty cache) and warm (option texts
already cached, state still new). Reported: median wall time per request after
one warm-up call.

These are toy implementations on one machine: they compare the designs, not
the production systems they're inspired by.

usage: uv run benchmark_speed.py [--questions 1 10 100] [--trials 5] [--only 01-4b 04]
"""

import argparse
import gc
import importlib.util
import itertools
import statistics
import time
from pathlib import Path

import torch

ROOT = Path(__file__).parent
TICKET = ("My payouts have failed three times. The bank says everything is fine. "
          "Can someone please fix this? definetely an account issue.")
TEMPLATES = [
    {"type": "choice", "instructions": "Which team should handle this ticket?",
     "criteria": {"payments": "Payout failures and payment processing",
                  "account": "Login and account access",
                  "other": "Something else"}},
    {"type": "yes_no", "instructions": "Does this message require urgent human attention?"},
]
counter = itertools.count()


def load(folder):
    spec = importlib.util.spec_from_file_location(f"dm_{folder}", ROOT / folder / "decision_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_questions(n):
    """n distinct questions, alternating between the two templates."""
    return {f"q{i}": {**TEMPLATES[i % 2], "instructions": f"{TEMPLATES[i % 2]['instructions']} (check {i})"}
            for i in range(n)}


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


def median_ms(fn, trials):
    fn()  # warm-up
    times = []
    for _ in range(trials):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def params(*modules):
    return sum(p.numel() for m in modules for p in m.parameters())


def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def builders():
    """(id, label, build) -> build returns (predict_fn(state, questions), n_params, before_call)."""
    def decoder(name):
        def build():
            dm = load("01_decoder").DecisionModel(model_name=name)
            return dm.predict, params(dm.model), None
        return build

    def encoder_mask():
        dm = load("02_encoder_mask").DecisionModel()
        return dm.predict, params(dm.model), None

    def encoder_markers():
        dm = load("03_encoder_markers").DecisionModel()
        return dm.predict, params(dm), None

    def contrastive(cold):
        def build():
            m = load("04_contrastive")
            dm = m.DecisionModel(m.Embedder())
            before = dm.embed.cache.clear if cold else None
            return dm.predict, params(dm.embed.model, dm), before
        return build

    return [
        ("01-0.8b", "01 decoder · Qwen3.5-0.8B", decoder("Qwen/Qwen3.5-0.8B")),
        ("01-4b", "01 decoder · Qwen3.5-4B", decoder("Qwen/Qwen3.5-4B")),
        ("02", "02 encoder + mask · ModernBERT-Large-Instruct", encoder_mask),
        ("03", "03 encoder + markers · ModernBERT-base", encoder_markers),
        ("04", "04 contrastive (cold) · Qwen3-Embedding-0.6B", contrastive(cold=True)),
        ("04", "04 contrastive (warm) · Qwen3-Embedding-0.6B", contrastive(cold=False)),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, nargs="+", default=[1, 10, 100])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--only", nargs="+", help="model ids to run: 01-0.8b 01-4b 02 03 04")
    args = ap.parse_args()

    device = ("cuda: " + torch.cuda.get_device_name() if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    rows = []
    for model_id, label, build in builders():
        if args.only and model_id not in args.only:
            continue
        predict, n_params, before = build()
        cells = []
        for n in args.questions:
            questions = make_questions(n)

            def call():
                if before:
                    before()
                predict(f"{TICKET} (ticket {next(counter)})", questions)

            ms = median_ms(call, args.trials if n < 100 else max(1, args.trials // 2))
            cells.append(ms)
            print(f"{label:48s} {n:4d} questions: {ms:9.1f} ms", flush=True)
        rows.append((label, n_params, cells))
        del predict, before
        free()

    print(f"\nDevice: {device}\n")
    head = " | ".join(f"{n} question{'s' if n > 1 else ''}" for n in args.questions)
    print(f"| Model | Params | {head} | per question @ {args.questions[-1]} |")
    print("|---|---|" + "---|" * (len(args.questions) + 1))
    for label, n_params, cells in rows:
        per_q = cells[-1] / args.questions[-1]
        fmt = " | ".join(f"{c:,.0f} ms" for c in cells)
        print(f"| {label} | {n_params / 1e6:,.0f}M | {fmt} | {per_q:,.1f} ms |")


if __name__ == "__main__":
    main()

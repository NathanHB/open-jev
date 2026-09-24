# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.2",
#     "transformers>=4.45",
#     "accelerate>=0.30",
#     "datasets",
#     "jinja2",
# ]
# ///
"""
Benchmark the decision model on the multiple-choice split of HLE (cais/hle).

Each HLE question is converted to the decision-model request schema:

    {
      "state": "",
      "questions": {
        "answer": {
          "type": "choice",
          "instructions": "<question stem>",
          "criteria": {"A": "<option A text>", "B": "<option B text>", ...}
        }
      }
    }

The stem is the question itself, so it goes in `instructions` and `state` is
left empty. The criteria keys are the original HLE letters, so the predicted
key can be compared to the gold `answer` directly even when options are
rotated. Options that point at other options ("Both A and C", "None of the
above") only make sense in the original order, so those questions are never
rotated. Image questions are skipped (the model is text-only).

usage: uv run bench_hle.py [--limit N] [--permutations K] [--device mps] [--model ID]
"""

import argparse
import json
import re
import time
from pathlib import Path

from datasets import load_dataset

import decision_model as open_jev

OPTION_RE = re.compile(r"^([A-Z])\.\s?(.*)$")
POSITIONAL_RE = re.compile(r"\b(above|below|previous|following)\b", re.IGNORECASE)


def order_dependent(criteria):
    """True if some option refers to another option by letter or position."""
    for text in criteria.values():
        if POSITIONAL_RE.search(text):
            return True
        if any(re.search(rf"\b{k}\b", text) for k in criteria):
            return True
    return False


def to_request(row):
    """Convert one HLE row to a decision-model request, or None if unparseable."""
    stem, sep, choices = row["question"].rpartition("Answer Choices:")
    if not sep:
        return None

    criteria, current = {}, None
    for line in choices.strip().splitlines():
        m = OPTION_RE.match(line)
        # A new option must be the next letter in sequence; anything else is a
        # continuation line of the current option (multi-line LaTeX etc.).
        if m and m.group(1) == open_jev.LETTERS[len(criteria)]:
            current = m.group(1)
            criteria[current] = m.group(2)
        elif current:
            criteria[current] += "\n" + line
    criteria = {k: v.strip() for k, v in criteria.items()}

    if len(criteria) < 2 or row["answer"] not in criteria:
        return None
    return {
        "state": "",
        "questions": {
            "answer": {
                "type": "choice",
                "instructions": stem.strip(),
                "criteria": criteria,
            }
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--permutations", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--model", default=open_jev.MODEL_NAME)
    ap.add_argument("--out", default="results/hle_mc.jsonl")
    args = ap.parse_args()

    ds = load_dataset("cais/hle", split="test")
    ds = ds.remove_columns(["image_preview", "rationale_image"])
    rows = [r for r in ds if r["answer_type"] == "multipleChoice" and not r["image"]]

    items, skipped = [], 0
    for r in rows:
        req = to_request(r)
        if req is None:
            skipped += 1
        else:
            items.append((r, req))
    items = items[: args.limit]
    print(f"{len(rows)} text-only MC questions, {skipped} unparseable, "
          f"evaluating {len(items)} with {args.model}")

    dm = open_jev.DecisionModel(model_name=args.model, device=args.device)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    correct, chance, t0 = 0, 0.0, time.time()
    with out.open("w") as f:
        for i, (r, req) in enumerate(items, 1):
            fixed = order_dependent(req["questions"]["answer"]["criteria"])
            n_perm = 1 if fixed else args.permutations
            dist = dm.predict(req["state"], req["questions"],
                              permutations=n_perm)["answer"]
            pred = max(dist, key=dist.get)
            correct += pred == r["answer"]
            chance += 1 / len(dist)
            f.write(json.dumps({
                "id": r["id"], "category": r["category"], "gold": r["answer"],
                "pred": pred, "p_gold": dist[r["answer"]], "dist": dist,
                "rotated": n_perm > 1,
            }) + "\n")
            if i % 25 == 0 or i == len(items):
                print(f"[{i}/{len(items)}] acc={correct / i:.3f} "
                      f"chance={chance / i:.3f} ({time.time() - t0:.0f}s)")

    n = len(items)
    print(f"\naccuracy {correct}/{n} = {correct / n:.3f} "
          f"(random baseline {chance / n:.3f})")


if __name__ == "__main__":
    main()

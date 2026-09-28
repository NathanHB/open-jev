# open-jev

**Four minimal ways to build a decision model**: typed questions in, a probability for every allowed answer out, and no text generation.

Decision models ("System One models") became a category in September 2026 when TypeSafe AI launched [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), a closed API. Open alternatives followed within days, and they're built in very different ways. This repo implements the four main designs from scratch, each in about 150 lines of PyTorch, with a small runnable demo.

## The request

Every example takes the same input: a **state** and some named **questions**, each with a fixed set of options.

```python
state = ("My payouts have failed three times. The bank says everything is fine. "
         "Can someone please fix this? definetely an account issue.")
questions = {
    "queue": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {
            "payments": "Payout failures and payment processing",
            "account": "Login and account access",
            "other": "Something else",
        },
    },
    "escalate": {"type": "yes_no", "instructions": "Does this message require urgent human attention?"},
}

dm.predict(state, questions)
# {'queue': {'payments': 0.989, 'account': 0.005, 'other': 0.006},
#  'escalate': {'yes': 0.937, 'no': 0.063}}          ← 01, Qwen3.5-4B
```

The only difference between the four examples is **where those probabilities come from**.

## The four architectures

| | Backbone | Answer read from | Training needed | Inspired by |
|---|---|---|---|---|
| [**01 · Decoder**](01_decoder) | Causal LLM (Qwen3.5) | Next-token logits for `A`/`B`/`C` | None | Jev ([as inferred](https://archerhume.com/posts/jevs-architecture-unmasked/)) |
| [**02 · Encoder + mask**](02_encoder_mask) | Masked LM (ModernBERT-Instruct) | `[MASK]` logits for `A`/`B`/`C` | None | [ModernBERT-Instruct](https://arxiv.org/abs/2502.03793) |
| [**03 · Encoder + markers**](03_encoder_markers) | Masked LM (ModernBERT) | A learned score at a `[MASK]` before each option | Yes (whole model) | [Laya](https://github.com/NandhaKishorM/laya) |
| [**04 · Contrastive**](04_contrastive) | Frozen embedding model (Qwen3-Embedding) | Cosine between state and option vectors | Yes (two small heads) | [CLM](https://github.com/Contrastive-LM/CLM) |

What changes from one design to the next:

| | 01 | 02 | 03 | 04 |
|---|---|---|---|---|
| Works zero-shot | ✅ | ✅ | ❌ | ⚠️ (raw embeddings only) |
| World knowledge | High, grows with size | Low | Low | Medium |
| State encoded once for all questions | ✅ KV cache | ❌ | ❌ | ✅ via embedding cache |
| Options see each other | ✅ | ✅ | ✅ | ❌ |
| Position bias | Rotate options at inference | Rotate options at inference | Shuffle options in training | None, by construction |
| Speed ([measured](#speed)) | Slowest (branches run one by one) | Fast | Fast | Fastest with many or reused options |

## Quickstart

Every script is a self-contained [uv](https://docs.astral.sh/uv/) script, so its dependencies install on first run.

```bash
cd 01_decoder && uv run decision_model.py
cd 02_encoder_mask && uv run decision_model.py
cd 03_encoder_markers && uv run decision_model.py   # trains ~1 min
cd 04_contrastive && uv run decision_model.py       # trains ~3 min
```

They run on CUDA, Apple Silicon (MPS) or CPU. The numbers in each folder's README come from an Apple Silicon laptop and can differ slightly on other hardware.

## Speed

Each model ran in its own job on an **A100 80GB** (`benchmark_speed.py`). The numbers are median latency per request, with the same ticket and N distinct questions, a fresh ticket every call, and one warm-up call first.

| Model | Params | 1 question | 10 questions | 100 questions | Per question @ 100 |
|---|---|---|---|---|---|
| 01 decoder · Qwen3.5-0.8B | 752M | 111 ms | 618 ms | 5,732 ms | 57.3 ms |
| 01 decoder · Qwen3.5-4B | 4,206M | 150 ms | 837 ms | 7,864 ms | 78.6 ms |
| 02 encoder + mask · ModernBERT-Large-Instruct | 396M | 23 ms | 27 ms | 109 ms | 1.1 ms |
| 03 encoder + markers · ModernBERT-base | 149M | 16 ms | 20 ms | 124 ms | 1.2 ms |
| 04 contrastive, cold · Qwen3-Embedding-0.6B | 600M | 77 ms | 83 ms | 140 ms | 1.4 ms |
| 04 contrastive, warm (options cached) | 600M | 41 ms | 43 ms | 96 ms | 1.0 ms |
| *Jev (API, measured by [Hume](https://archerhume.com/posts/jevs-architecture-unmasked/))* | *undisclosed* | *~87 ms* | | | *~0.4 ms @ 1,500 (~610 ms total)* |
| *Jev ([TypeSafe's claim](https://typesafe.ai/blog/introducing-system-one-models-and-jev))* | *undisclosed* | *70–500 ms per request* | | | |

- **01 grows linearly** because our toy runs each question's branch as its own forward pass. Batching the branches would make it nearly flat, and that's presumably what gives Jev about 0.4 ms per question.
- **02 and 03** batch every question into one forward pass, so 10 questions cost barely more than 1.
- **04** has the highest fixed cost (two embedding calls) but grows the least, and caching the options saves about 40 ms per request.
- The Jev rows are external measurements through the API (network included, unknown hardware), so they're not directly comparable with ours.
- Setup: 01 ran in bf16 with `flash-linear-attention` (`causal_conv1d` not installed); 02, 03 and 04 ran in fp32 with default attention.

```bash
uv run benchmark_speed.py --only 02 03    # model ids: 01-0.8b 01-4b 02 03 04
```

## Which one when?

- **The decision needs knowledge or reasoning** (medicine, law, puzzles): use **01**. Nothing else knows enough.
- **Many questions about one long document:** **01** reuses the document through the KV cache, but only pays off once its question branches are batched (see [Speed](#speed)).
- **A narrow, high-volume task with labelled data** (routing, moderation): use **03**. It's tiny and fast once trained.
- **Picking among many or reusable candidates** (tools, next actions, best-of-N answers): use **04**. Candidates are embedded once and cached.

## A note on calibration

A probability is only useful if "80%" means right 80% of the time. None of these toy examples is calibrated out of the box. Production systems fit at least a temperature on held-out data, and Jev, Laya and CLM all train specifically for it.

## References

- TypeSafe AI, [Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) (Sep 15, 2026)
- Archer Hume, [Jev's Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked/) (Sep 17, 2026): the black-box reconstruction that 01 is based on
- [Laya](https://github.com/NandhaKishorM/laya) (Convai Innovations, Sep 18, 2026)
- [CLM: Contrastive Language Models](https://github.com/Contrastive-LM/CLM) (Kwok et al., 2026)
- [autojev](https://github.com/denis-pplx/autojev): a fine-tuned Qwen decision model, 01 plus training
- Clavié et al., [It's All in The [MASK]](https://arxiv.org/abs/2502.03793) (ModernBERT-Instruct)

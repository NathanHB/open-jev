# 01 · Decoder + letter logits

A causal LLM reads the options as a lettered list. We **don't generate anything**. We read the probability of `A`, `B`, `C`… as the next token.

This follows the architecture Archer Hume [inferred for Jev](https://archerhume.com/posts/jevs-architecture-unmasked/): a causal backbone, a shared state encoded once, isolated question branches, and a direct probability readout.

## How it works

```
state ──► [KV cache] ──┬── copy ──► "Which team?"   ──► P(A, B, C)
                       ├── copy ──► "Is it urgent?" ──► P(A, B)
                       └── copy ──► …
```

1. **Encode the state once.** In a decoder, a token's keys and values never change when text is added after it, so the state is encoded into a KV cache once.
2. **One branch per question.** Each question runs only its own suffix on a **copy** of that cache, so questions can't see each other.
3. **Read the letters.** Take the next-token logits at the last position, keep only the option letters, and softmax them.

```python
logits = model(suffix_ids, past_key_values=branch).logits[0, -1]
probs = logits[[id_A, id_B, id_C]].softmax(-1)
```

## The prompt

This uses the model's chat template, with thinking turned off:

```
<|im_start|>user
My payouts have failed three times. The bank says everything is fine. …

Question: Which team should handle this ticket?

A. Payout failures and payment processing
B. Login and account access
C. Something else
Answer with the letter of the best option only.<|im_end|>
<|im_start|>assistant
<think>

</think>

      ← logits are read here
```

## Run

```bash
uv run decision_model.py          # ticket demo (Qwen3.5-4B by default)
uv run ask.py MODEL [MODEL ...]   # bat-and-ball question on one or more models
uv run bench_hle.py --limit 10    # multiple-choice HLE (gated dataset, needs HF login)
```

Output of the ticket demo:

```
queue {'payments': 0.989, 'account': 0.005, 'other': 0.006}
escalate {'yes': 0.937, 'no': 0.063}
```

## Toy example: the bat and the ball

*"A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"* The correct answer is $0.05; the intuitive trap is $0.10.

| Model | $0.05 (correct) | $0.10 (trap) |
|---|---|---|
| Qwen3.5-0.8B | 0.20 | **0.47** |
| Qwen3.5-4B | **0.95** | 0.03 |
| Qwen3.5-27B | **1.00** | 0.00 |

No training is needed, and a bigger model makes better decisions.

## Things to know

- **Position bias.** Qwen3.5-4B gives $0.05 a probability of 0.50 when it's option A, and 0.95 when it's option D. `predict(..., permutations=k)` averages over rotated option orders to cancel this.
- **Hybrid attention.** Qwen3.5 mixes full attention with linear-attention layers, which keep a running state. You can't rewind that state by "cropping" the cache, so the cache is copied per question, and it needs `DynamicCache(config=model.config)`.
- **Up to 26 options** (one letter each).
- **HLE:** on 10 multiple-choice questions, 0.8B, 4B and 27B all scored 1/10 (random ≈ 1.65/10). Reading the answer immediately, with no reasoning, isn't enough for HLE.

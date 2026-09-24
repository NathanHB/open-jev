# 02 · Encoder + one `[MASK]` answer slot

An encoder can't predict a *next* token, but it can fill in a blank. So we put the blank exactly where the answer letter goes and read the masked-LM logits for `A`, `B`, `C`…

## How it works

```
"… CHOICES: - A: … - B: … - C: … ANSWER: [unused0] [MASK]"
                                                     │
                                  MLM head ──► P(A, B, C)
```

1. Write the options as a lettered list and end the prompt with `[MASK]`.
2. Run the encoder once. Every token sees every other token, in both directions.
3. At the `[MASK]` position, keep only the letter tokens and softmax them.

```python
logits = mlm(**inputs).logits[0]
pos = (inputs.input_ids[0] == tok.mask_token_id).nonzero()[-1, 0]
probs = logits[pos, [id_A, id_B, id_C]].softmax(-1)
```

We use [ModernBERT-Large-Instruct](https://huggingface.co/answerdotai/ModernBERT-Large-Instruct) (~0.4B), which was instruction-tuned for exactly this template ([paper](https://arxiv.org/abs/2502.03793)). A plain BERT would spread its guesses over the whole vocabulary.

## The prompt

```
You will be given a question and options. Select the right answer.
QUESTION: My payouts have failed three times. …

Which team should handle this ticket?
CHOICES:
- A: Payout failures and payment processing
- B: Login and account access
- C: Something else
ANSWER: [unused0] [MASK]      ← logits are read here
```

## Run

```bash
uv run decision_model.py
```

```
permutations=1 queue {'payments': 0.789, 'account': 0.137, 'other': 0.073}
permutations=1 escalate {'yes': 0.881, 'no': 0.119}
permutations=3 queue {'payments': 0.832, 'account': 0.111, 'other': 0.058}
permutations=3 escalate {'yes': 0.615, 'no': 0.385}
```

With the original order, `escalate` says yes 0.88. Averaged over both orders it drops to 0.62: "Yes" was benefiting from always being option A.

## Things to know

- **No KV cache.** Attention goes both ways, so adding a question changes how the state is encoded. Each question is encoded in full, but all questions (and option orders) go through **one batched forward pass**.
- **Small and fast, but knows little.** On the bat-and-ball question it picks $0.55 (41%) and gives the correct $0.05 only 12%.
- **Position bias** exists here too, so `permutations=k` averages over rotated orders, as in 01.
- The letters it predicts are space-prefixed tokens (` A`, ` B`, …).

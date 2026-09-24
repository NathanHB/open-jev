# 03 · Encoder + a marker per option (Laya-style)

No letters and no vocabulary. We put a `[MASK]` **in front of every option**, let the encoder read everything, and train a tiny scorer that turns each marker's vector into one number.

This is a stripped-down version of [Laya](https://github.com/NandhaKishorM/laya).

## How it works

```
[CLS] question [SEP] [MASK] option 0  [MASK] option 1  [MASK] option 2 [SEP] state [SEP]
                        │                │                │
                        h₀               h₁               h₂      ← encoder output at each marker
                        └──────── Linear(d, 1) ──────────┘
                                       softmax ──► P(option)
```

1. Build one row per question: the question, then a `[MASK]` before each option, then the state (only the state is ever truncated).
2. The encoder reads the row in both directions, so each marker sees its own option, the other options, the question and the state.
3. `scorer = nn.Linear(d, 1)` turns each marker's vector into a score, and a softmax over the markers gives the answer.

```python
h = encoder(input_ids).last_hidden_state     # [batch, tokens, d]
m = h[:, marker_positions]                   # one vector per option
probs = scorer(m).squeeze(-1).softmax(-1)
```

## It must be trained

Nothing in pretraining taught the model what these markers mean, so the scorer starts random. `train_on` runs plain cross-entropy and **shuffles the options every step**, so the model can't learn to prefer a position. That replaces rotation averaging at inference time.

The demo trains [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) (149M) on 800 [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) articles (4 topics) and tests on 200 others.

## Run

```bash
uv run decision_model.py
```

```
untrained accuracy: 0.240 (random = 0.25)
epoch 0 step 25 loss 1.043
epoch 0 step 50 loss 0.480
epoch 0 step 75 loss 0.197
epoch 0 step 100 loss 0.109
trained accuracy:   0.875
{'world': 0.01, 'sports': 0.004, 'business': 0.98, 'sci_tech': 0.005}
```

The model is at chance before training and at 87.5% after 100 steps, which takes about a minute on a laptop GPU. The last line classifies the payout ticket as a news topic; "business" is the sensible answer.

## Things to know

- **Useless zero-shot, strong once trained** on your task, and very fast. Laya's own README calls its base checkpoints near chance zero-shot: *"a fast base to specialise, not a zero-shot decision engine."*
- **Simplified from Laya:** there are no extra head layers, no question-type embedding, no token budget for the options, and no RL training with proper scoring rules.
- `transformers` prints `UNEXPECTED` warnings for the checkpoint's masked-LM head. That's expected: we load the bare encoder and use our own scorer.

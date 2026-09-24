# 04 · Contrastive two-tower (CLM-style)

The state and each option are embedded **separately**, and each option is scored by how close its vector is to the state's. Options never see each other.

This is a stripped-down version of [CLM](https://github.com/Contrastive-LM/CLM) (Contrastive Language Models).

## How it works

```
"state + question" ──► frozen LLM ──► state head  ──► s  ─┐
"option 0"         ──► frozen LLM ──► option head ──► a₀ ─┼─► softmax(scale · cos(s, aᵢ))
"option 1"         ──► frozen LLM ──► option head ──► a₁ ─┘
```

1. **Frozen backbone.** [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), a decoder used as an encoder: the last token's hidden state is the embedding of the whole text. Every embedding is computed once and cached.
2. **Two small heads**, one for states and one for options. They're residual MLPs that start as the identity, and they're the only trained part.
3. **Score** = `scale · cos(state, option)`, then a softmax.

```python
s = F.normalize(state_head(embed([state_text])), dim=-1)[0]
a = F.normalize(action_head(embed(options)), dim=-1)
probs = (scale * a @ s).softmax(-1)
```

## Training: InfoNCE

Each state is pulled toward the action that was taken and pushed away from the other actions in the batch. **Stage 2** adds *hard negatives*: plausible but wrong actions. Because the backbone is frozen, the heads train on cached vectors in seconds.

The demo is a toy **best-of-4 verifier** on [SciQ](https://huggingface.co/datasets/allenai/sciq): each science question has 1 correct answer and 3 plausible wrong ones, and the model must pick the right one. It's the same shape as CLM's DeepSWE evaluation, where the candidates are 4 agent trajectories and the tests decide which ones are correct.

## Run

```bash
uv run decision_model.py      # ~3 min on a laptop GPU, almost all of it embedding
```

```
random pick:                 0.250
raw embeddings (no heads):   0.662
after stage 1:               0.690
after stage 2:               0.710

Q: Compounds that are capable of accepting electrons, such as o 2 or f2, are called what?
  0.732  oxidants
  0.045  antioxidants
  0.220  Oxygen
  0.003  residues
same scores after shuffling: True
log-odds(0 vs 1) before / after adding an option: 2.785 / 2.785
```

*(Training loss lines omitted.)* Each stage helps: 0.66 with raw embeddings, 0.69 after InfoNCE, and 0.71 once hard negatives are added. The last two lines check the two structural properties below.

## Things to know

- **No position bias, by construction.** Shuffling the options changes nothing.
- **No option interaction.** Adding an option can't change the odds between the existing ones. Hume's black-box tests found that Jev's options *do* interact, so Jev isn't built like this.
- **Options can be cached and reused** across requests, which makes it very fast with many candidates. Candidates can also be anything: tool calls, code patches, whole trajectories.
- **It can't handle** options like "both A and C" or "none of the above", and it squeezes the whole state into one vector.
- **Heads start as the identity** because Qwen3-Embedding is already contrastively trained. Randomly initialised heads threw that away and scored *below* the raw embeddings.

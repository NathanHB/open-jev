# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.2",
#     "transformers>=4.51",
#     "datasets",
# ]
# ///
"""
A minimal CLM-style (contrastive, two-tower) decision model.

  1. A FROZEN embedding model turns text into one vector (last-token pooling).
     The state (+ question) and every option are embedded SEPARATELY.
  2. Two small trainable heads (residual MLPs that start as the identity)
     adjust state and option vectors; an option's score is
     scale * cos(state, option), softmaxed.
  3. The heads are trained with InfoNCE: pull each state toward the action that
     was taken, push it away from the others. Stage 2 adds hard negatives
     (plausible but wrong actions).

Because options never see each other, the scores don't depend on option order,
and adding an option can't change the odds between the existing ones. Because
the backbone is frozen, every embedding is computed once and cached, so
training the heads takes seconds.

Toy task: SciQ science questions, pick the right answer among 4 candidates,
the same shape as a best-of-4 verifier.
"""

import random
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


def normalise_options(question):
    """Return a list of (key, description) pairs for a question."""
    if question["type"] == "yes_no":
        return [("yes", "Yes"), ("no", "No")]
    if question["type"] == "choice":
        return list(question["criteria"].items())
    raise ValueError(f"Unknown question type: {question['type']}")


class Embedder:
    """Frozen decoder used as an encoder: the last token's hidden state is the
    embedding of the whole text. Embeddings are cached by text."""

    def __init__(self, model_name=MODEL_NAME, device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.cache = {}

    @torch.no_grad()
    def __call__(self, texts, batch_size=64):
        missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
        for i in range(0, len(missing), batch_size):
            chunk = missing[i : i + batch_size]
            batch = self.tok(chunk, padding=True, truncation=True, max_length=512,
                             return_tensors="pt").to(self.device)
            h = self.model(**batch).last_hidden_state[:, -1]  # left padding: last = real token
            for text, vec in zip(chunk, F.normalize(h.float(), dim=-1).cpu()):
                self.cache[text] = vec
        return torch.stack([self.cache[t] for t in texts])


class Head(torch.nn.Module):
    """x + MLP(x), with the MLP's last layer zeroed: it starts as the identity,
    so training begins from the embedding model's own similarity and refines it."""

    def __init__(self, dim, width=1024):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, width), torch.nn.GELU(), torch.nn.Linear(width, dim)
        )
        torch.nn.init.zeros_(self.mlp[-1].weight)
        torch.nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        return x + self.mlp(x)


class DecisionModel(torch.nn.Module):
    def __init__(self, embedder):
        super().__init__()
        self.embed = embedder
        dim = embedder.model.config.hidden_size
        self.state_head = Head(dim)
        self.action_head = Head(dim)
        self.logit_scale = torch.nn.Parameter(torch.tensor(1 / 0.07).log())

    def _project(self, head, vecs):
        return F.normalize(head(vecs), dim=-1)

    def scale(self):
        return self.logit_scale.exp().clamp(max=100)

    # ---------- inference ----------

    @torch.no_grad()
    def scores(self, state_text, candidates):
        """Logits for each candidate: scale * cos(state, candidate)."""
        s = self._project(self.state_head, self.embed([state_text]))[0]
        a = self._project(self.action_head, self.embed(candidates))
        return self.scale() * (a @ s)

    def predict(self, state, questions):
        """
        questions: {question_id: {"type": ..., "instructions": ..., "criteria": ...}}
        Returns {question_id: {option_key: probability}}. The state tower reads
        state + question; each option is embedded on its own.
        """
        out = {}
        for qid, q in questions.items():
            opts = normalise_options(q)
            state_text = f"{state}\n\n{q['instructions']}" if state else q["instructions"]
            probs = self.scores(state_text, [desc for _, desc in opts]).softmax(-1)
            out[qid] = {key: p.item() for (key, _), p in zip(opts, probs)}
        return out

    # ---------- training ----------

    def train_on(self, pairs, hard_negatives=None, epochs=3, batch_size=256, lr=1e-4, seed=0):
        """
        pairs: list of (state_text, action_text), the action that was taken.
        hard_negatives: optional list (same length) of lists of wrong actions.
        Bidirectional InfoNCE with in-batch negatives; hard negatives are added
        as extra columns in the state -> action direction.
        """
        S = self.embed([s for s, _ in pairs])       # computed once, cached
        A = self.embed([a for _, a in pairs])
        H = None
        if hard_negatives is not None:
            k = min(len(h) for h in hard_negatives)
            H = self.embed([n for h in hard_negatives for n in h[:k]]).view(len(pairs), k, -1)

        opt = torch.optim.AdamW(self.parameters(), lr=lr)
        g = torch.Generator().manual_seed(seed)
        for epoch in range(epochs):
            order = torch.randperm(len(pairs), generator=g)
            total = 0.0
            for i in range(0, len(pairs), batch_size):
                idx = order[i : i + batch_size]
                s = self._project(self.state_head, S[idx])
                a = self._project(self.action_head, A[idx])
                logits = self.scale() * s @ a.T                  # [B, B], diagonal = positives
                target = torch.arange(len(idx))
                s2a = logits
                if H is not None:
                    h = self._project(self.action_head, H[idx])  # [B, k, d]
                    hard = self.scale() * (h @ s[:, :, None]).squeeze(-1)
                    s2a = torch.cat([logits, hard], dim=1)
                loss = (F.cross_entropy(s2a, target) + F.cross_entropy(logits.T, target)) / 2
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            print(f"  epoch {epoch} loss {total / len(pairs):.3f}")


# ---------- demo ----------

if __name__ == "__main__":
    from datasets import load_dataset

    rng = random.Random(0)
    sciq = load_dataset("allenai/sciq")
    train = list(sciq["train"])
    test = list(sciq["test"])

    def candidates(r):
        return [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]

    emb = Embedder()
    print("embedding train + test texts once (frozen backbone)...")
    emb([r["question"] for r in train + test] + [c for r in train + test for c in candidates(r)])

    def best_of_4(pick):
        """Fraction of test questions where the picked candidate is the right one."""
        return sum(pick(r["question"], candidates(r)) == 0 for r in test) / len(test)

    raw = best_of_4(lambda q, c: (emb(c) @ emb([q])[0]).argmax().item())
    print(f"random pick:                 0.250")
    print(f"raw embeddings (no heads):   {raw:.3f}")

    torch.manual_seed(0)  # the heads' first layers start random: seed for repeatable numbers
    dm = DecisionModel(emb)
    print("stage 1: InfoNCE on (question, correct answer) pairs")
    dm.train_on([(r["question"], r["correct_answer"]) for r in train])
    print(f"after stage 1:               {best_of_4(lambda q, c: dm.scores(q, c).argmax().item()):.3f}")

    print("stage 2: + distractors as hard negatives")
    dm.train_on([(r["question"], r["correct_answer"]) for r in train],
                hard_negatives=[candidates(r)[1:] for r in train])
    print(f"after stage 2:               {best_of_4(lambda q, c: dm.scores(q, c).argmax().item()):.3f}")

    # No position bias and no option interaction, by construction.
    r = test[0]
    q, c = r["question"], candidates(r)
    z = dm.scores(q, c)
    shuffled = c[1:] + c[:1]
    z_shuffled = dm.scores(q, shuffled)
    z_more = dm.scores(q, c + ["The color blue."])
    print(f"\nQ: {q}")
    for text, p in zip(c, z.softmax(-1)):
        print(f"  {p:.3f}  {text}")
    print("same scores after shuffling:", torch.allclose(z, torch.cat([z_shuffled[-1:], z_shuffled[:-1]])))
    print("log-odds(0 vs 1) before / after adding an option:",
          f"{(z[0] - z[1]).item():.3f} / {(z_more[0] - z_more[1]).item():.3f}")

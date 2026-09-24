# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.2",
#     "transformers>=4.48",
#     "datasets",
# ]
# ///
"""
A minimal Laya-style decision model: an encoder with a [MASK] marker in front
of every option and a small learned scorer on those markers.

Input layout (one row per question):

    [CLS] question [SEP] [MASK] option 0 [MASK] option 1 ... [SEP] state [SEP]

  1. The encoder reads the whole row bidirectionally, so each [MASK] marker
     sees its own option, the other options, the question and the state.
  2. A linear scorer turns each marker's hidden state into one logit; a
     softmax over the markers gives the distribution. No letters, no
     vocabulary: options are identified by position.
  3. The scorer starts random, so the model must be trained. `train` does
     plain cross-entropy with options shuffled every step (no position bias).

Simplified from Laya: no extra head layers, no question-type embedding, no
token budgets, no proper-scoring-rule RL.
"""

import random
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "answerdotai/ModernBERT-base"


def normalise_options(question):
    """Return a list of (key, description) pairs for a question."""
    if question["type"] == "yes_no":
        return [("yes", "Yes"), ("no", "No")]
    if question["type"] == "choice":
        return list(question["criteria"].items())
    raise ValueError(f"Unknown question type: {question['type']}")


class DecisionModel(torch.nn.Module):
    def __init__(self, model_name=MODEL_NAME, device=None, max_len=256):
        super().__init__()
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.max_len = max_len
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.scorer = torch.nn.Linear(self.encoder.config.hidden_size, 1)
        self.to(self.device)

    # ---------- input layout ----------

    def _row(self, state, instructions, options):
        """Token ids for one question, plus the position of each [MASK] marker."""
        tok = self.tok
        ids = [tok.cls_token_id]
        ids += tok.encode(instructions, add_special_tokens=False)
        ids.append(tok.sep_token_id)
        markers = []
        for _, desc in options:
            markers.append(len(ids))
            ids.append(tok.mask_token_id)
            ids += tok.encode(" " + desc, add_special_tokens=False)
        ids.append(tok.sep_token_id)
        # The state goes last and is the only part that gets truncated.
        room = max(0, self.max_len - len(ids) - 1)
        ids += tok.encode(state, add_special_tokens=False)[:room]
        ids.append(tok.sep_token_id)
        return ids, markers

    # ---------- core computation ----------

    def _logits(self, rows):
        """rows: list of (ids, markers). Returns [n_rows, max_options] logits."""
        n, length = len(rows), max(len(ids) for ids, _ in rows)
        k = max(len(m) for _, m in rows)
        input_ids = torch.full((n, length), self.tok.pad_token_id)
        attention = torch.zeros((n, length), dtype=torch.long)
        marker_pos = torch.zeros((n, k), dtype=torch.long)
        marker_ok = torch.zeros((n, k), dtype=torch.bool)
        for i, (ids, markers) in enumerate(rows):
            input_ids[i, : len(ids)] = torch.tensor(ids)
            attention[i, : len(ids)] = 1
            marker_pos[i, : len(markers)] = torch.tensor(markers)
            marker_ok[i, : len(markers)] = True

        h = self.encoder(
            input_ids=input_ids.to(self.device),
            attention_mask=attention.to(self.device),
        ).last_hidden_state                                   # [n, length, d]
        idx = marker_pos.to(self.device)[:, :, None].expand(-1, -1, h.size(-1))
        logits = self.scorer(torch.gather(h, 1, idx)).squeeze(-1)  # [n, k]
        return logits.masked_fill(~marker_ok.to(self.device), -1e4)

    @torch.no_grad()
    def predict(self, state, questions):
        """
        questions: {question_id: {"type": ..., "instructions": ..., "criteria": ...}}
        Returns {question_id: {option_key: probability}}. All questions run in
        one batched forward pass.
        """
        self.eval()
        qids = list(questions)
        opts = [normalise_options(questions[q]) for q in qids]
        rows = [self._row(state, questions[q]["instructions"], o) for q, o in zip(qids, opts)]
        probs = F.softmax(self._logits(rows).float(), dim=-1).cpu()
        return {
            q: {key: probs[i, j].item() for j, (key, _) in enumerate(o)}
            for i, (q, o) in enumerate(zip(qids, opts))
        }

    # ---------- training ----------

    def train_on(self, examples, epochs=1, batch_size=8, lr=3e-5, seed=0):
        """
        examples: list of (state, question, correct_key).
        Cross-entropy over the markers; options are shuffled every time so the
        model can't learn to prefer a position.
        """
        rng = random.Random(seed)
        opt = torch.optim.AdamW(self.parameters(), lr=lr)
        self.train()
        for epoch in range(epochs):
            order = examples[:]
            rng.shuffle(order)
            for start in range(0, len(order), batch_size):
                rows, targets = [], []
                for state, q, correct in order[start : start + batch_size]:
                    options = normalise_options(q)
                    rng.shuffle(options)
                    rows.append(self._row(state, q["instructions"], options))
                    targets.append([k for k, _ in options].index(correct))
                loss = F.cross_entropy(self._logits(rows).float(),
                                       torch.tensor(targets, device=self.device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                step = start // batch_size + 1
                if step % 25 == 0:
                    print(f"epoch {epoch} step {step} loss {loss.item():.3f}")


# ---------- demo ----------

if __name__ == "__main__":
    from datasets import load_dataset

    # AG News: 4 topics. Train briefly on 800 articles, test on 200 others.
    topic = {
        "type": "choice",
        "instructions": "What is the topic of this news article?",
        "criteria": {
            "world": "World news and politics",
            "sports": "Sports",
            "business": "Business and economy",
            "sci_tech": "Science and technology",
        },
    }
    keys = list(topic["criteria"])
    ds = load_dataset("fancyzhx/ag_news", split="train").shuffle(seed=0)
    data = [(r["text"], topic, keys[r["label"]]) for r in ds.select(range(1000))]
    train, test = data[:800], data[800:]

    def accuracy(dm):
        hits = 0
        for state, q, correct in test:
            dist = dm.predict(state, {"topic": q})["topic"]
            hits += max(dist, key=dist.get) == correct
        return hits / len(test)

    torch.manual_seed(0)  # the scorer starts random: seed it for repeatable numbers
    dm = DecisionModel()
    print(f"untrained accuracy: {accuracy(dm):.3f} (random = 0.25)")
    dm.train_on(train)
    print(f"trained accuracy:   {accuracy(dm):.3f}")

    ticket = ("My payouts have failed three times. The bank says everything "
              "is fine. Can someone please fix this?")
    dist = dm.predict(ticket, {"topic": topic})["topic"]
    print({k: round(v, 3) for k, v in dist.items()})

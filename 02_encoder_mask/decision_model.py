# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.2",
#     "transformers>=4.48",
# ]
# ///
"""
The same Jev-style decision model as open-jev.py, but on a bidirectional
encoder (masked language model) instead of a causal LLM.

What changes compared to the decoder version:
  1. The answer slot is a [MASK] token at the end of the prompt. The MLM head
     predicts what token belongs there; we read the logits for the option
     letters (A, B, C...) at that position and softmax them.
  2. No KV cache: every token attends to every other token, so the question
     changes how the state is encoded. Each question is encoded in full, with
     all questions batched into one padded forward pass.
  3. Optionally average over rotated option orders (reduces order bias).

Uses ModernBERT-Large-Instruct, which was instruction-tuned to answer this
exact "ANSWER: [unused0] [MASK]" template (arXiv:2502.03793).
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM

MODEL_NAME = "answerdotai/ModernBERT-Large-Instruct"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def normalise_options(question):
    """Return a list of (key, description) pairs for a question."""
    if question["type"] == "yes_no":
        return [("yes", "Yes"), ("no", "No")]
    if question["type"] == "choice":
        return list(question["criteria"].items())
    raise ValueError(f"Unknown question type: {question['type']}")


class DecisionModel:
    def __init__(self, model_name=MODEL_NAME, device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.temperature = 1.0

        # Token id for " A", " B", ... — what the MLM head predicts at [MASK].
        self.letter_ids = []
        for letter in LETTERS:
            ids = self.tok.encode(" " + letter, add_special_tokens=False)
            assert len(ids) == 1, f"' {letter}' is not a single token"
            self.letter_ids.append(ids[0])

    def _prompt(self, state, instructions, options):
        question = f"{state}\n\n{instructions}" if state else instructions
        lines = [
            "You will be given a question and options. Select the right answer.",
            f"QUESTION: {question}",
            "CHOICES:",
        ]
        for i, (_, desc) in enumerate(options):
            lines.append(f"- {LETTERS[i]}: {desc}")
        lines.append(f"ANSWER: [unused0] {self.tok.mask_token}")
        return "\n".join(lines)

    @torch.no_grad()
    def _logits(self, state, jobs):
        """
        jobs: list of (instructions, options). Returns one logit tensor per job.
        All jobs go through the encoder in a single padded batch.
        """
        for _, options in jobs:
            if len(options) > len(LETTERS):
                raise ValueError("This prototype supports at most 26 options")

        texts = [self._prompt(state, ins, opts) for ins, opts in jobs]
        batch = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
        logits = self.model(**batch).logits

        results = []
        for row, (_, options) in enumerate(jobs):
            pos = (batch.input_ids[row] == self.tok.mask_token_id).nonzero()[-1, 0]
            z = logits[row, pos].float()
            results.append(z[self.letter_ids[: len(options)]].cpu())
        return results

    def predict(self, state, questions, permutations=1):
        """
        questions: {question_id: {"type": ..., "instructions": ..., "criteria": ...}}
        permutations: >1 averages over cyclic rotations of the option order.
            Capped at the number of options; at that cap every option sits in
            every letter slot exactly once, cancelling position bias.
        Returns {question_id: {option_key: probability}}.
        """
        jobs, index = [], []  # index maps each job back to (qid, key order)
        n_orders = {}
        for qid, q in questions.items():
            base = normalise_options(q)
            n_orders[qid] = min(permutations, len(base))
            for p in range(n_orders[qid]):
                opts = base[p:] + base[:p]
                jobs.append((q["instructions"], opts))
                index.append((qid, [k for k, _ in opts]))

        logits = self._logits(state, jobs)

        totals = {qid: {} for qid in questions}
        for (qid, keys), z in zip(index, logits):
            probs = F.softmax(z / self.temperature, dim=-1)
            for key, p in zip(keys, probs.tolist()):
                totals[qid][key] = totals[qid].get(key, 0.0) + p / n_orders[qid]
        return totals


# ---------- demo ----------

if __name__ == "__main__":
    dm = DecisionModel()
    request = {
        "state": "My payouts have failed three times. The bank says everything "
                 "is fine. Can someone please fix this? definetely an account issue.",
        "questions": {
            "queue": {
                "type": "choice",
                "instructions": "Which team should handle this ticket?",
                "criteria": {
                    "payments": "Payout failures and payment processing",
                    "account": "Login and account access",
                    "other": "Something else",
                },
            },
            "escalate": {
                "type": "yes_no",
                "instructions": "Does this message require urgent human attention?",
            },
        },
    }
    for k in (1, 3):
        result = dm.predict(request["state"], request["questions"], permutations=k)
        for qid, dist in result.items():
            print(f"permutations={k}", qid, {o: round(p, 3) for o, p in dist.items()})

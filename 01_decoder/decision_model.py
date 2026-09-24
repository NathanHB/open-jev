# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.2",
#     "transformers>=4.45",
#     "accelerate>=0.30",
# ]
# ///
"""
A minimal Jev-style "decision model" built on an open causal LLM.

Follows the architecture inferred for Jev in Archer Hume's "Jev's Architecture
Unmasked" (https://archerhume.com/posts/jevs-architecture-unmasked/):
  1. Encode the shared STATE once into a KV cache.
  2. For each question, run only its suffix (instructions + options) on top of
     a copy of that cache -> questions can't see each other.
  3. Read the next-token logits for the option letters (A, B, C...) at the
     final position and softmax them -> a probability distribution, no text
     generation.
  4. Optionally average over rotated option orders (reduces order bias).
"""

import copy
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

MODEL_NAME = "Qwen/Qwen3.5-4B"
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
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype
        ).to(self.device)
        self.model.eval()
        self.temperature = 1.0

        # Token id for "A", "B", ... — these act as the K "option slots". The
        # assistant turn starts right after the chat template, so no leading space.
        self.letter_ids = []
        for letter in LETTERS:
            ids = self.tok.encode(letter, add_special_tokens=False)
            assert len(ids) == 1, f"'{letter}' is not a single token"
            self.letter_ids.append(ids[0])

    # ---------- prompt pieces ----------

    def _state_text(self, state):
        return state

    def _suffix_text(self, instructions, options):
        lines = [f"Question: {instructions}", ""]
        for i, (_, desc) in enumerate(options):
            lines.append(f"{LETTERS[i]}. {desc}")
        lines.append("Answer with the letter of the best option only.")
        return "\n".join(lines)

    def _prompt_parts(self, state, instructions, options):
        """
        Render the full chat prompt (one user turn: state + question), then
        split it right after the state -> (shared prefix, per-question suffix).
        Thinking is disabled so the answer letter is the very next token.
        """
        state_text = self._state_text(state)
        suffix = self._suffix_text(instructions, options)
        # An empty state is allowed: the question then stands on its own.
        content = f"{state_text}\n\n{suffix}" if state_text else suffix
        full = self.tok.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        cut = full.index(content) + len(state_text)
        return full[:cut], full[cut:]

    # ---------- core computation ----------

    @torch.no_grad()
    def _branch_logits(self, state, jobs):
        """
        jobs: list of (instructions, options). Returns one logit tensor per job.
        The state is encoded ONCE; each job runs only its own suffix.
        """
        # The prefix is the same for every job; render it via any job.
        prefix_text, _ = self._prompt_parts(state, *jobs[0])
        state_ids = self.tok(
            prefix_text, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self.device)

        # config= lets the cache allocate the right layer types (Qwen3.5 mixes
        # full attention with linear attention).
        cache = DynamicCache(config=self.model.config)
        self.model(state_ids, past_key_values=cache, use_cache=True)

        results = []
        for instructions, options in jobs:
            if len(options) > len(LETTERS):
                raise ValueError("This prototype supports at most 26 options")

            _, suffix_text = self._prompt_parts(state, instructions, options)
            suffix_ids = self.tok(
                suffix_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids.to(self.device)

            # Branch off a copy of the state cache -> questions can't see each
            # other. (crop() can't rewind linear-attention recurrent state.)
            branch = copy.deepcopy(cache)
            out = self.model(suffix_ids, past_key_values=branch, use_cache=True)
            last = out.logits[0, -1].float()
            results.append(last[self.letter_ids[: len(options)]].cpu())
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

        logits = self._branch_logits(state, jobs)

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
    result = dm.predict(request["state"], request["questions"], permutations=3)
    for qid, dist in result.items():
        print(qid, {k: round(v, 3) for k, v in dist.items()})


# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.2",
#     "transformers>=4.45",
#     "accelerate>=0.30",
#     "jinja2",
# ]
# ///
"""
Ask the bat-and-ball question to one or more models.

usage: uv run ask.py [MODEL ...]
"""

import gc
import sys

import torch

import decision_model as open_jev

REQUEST = {
    "state": "",
    "questions": {
        "ball": {
            "type": "choice",
            "instructions": "A bat and a ball cost $1.10 in total. The bat costs "
                            "$1.00 more than the ball. How much does the ball cost?",
            "criteria": {
                "$0.10": "$0.10",
                "$0.55": "$0.55",
                "$1.00": "$1.00",
                "$0.05": "$0.05",
            },
        }
    },
}

if __name__ == "__main__":
    for model in sys.argv[1:] or [open_jev.MODEL_NAME]:
        dm = open_jev.DecisionModel(model_name=model)
        for k in (1, 4):
            dist = dm.predict(REQUEST["state"], REQUEST["questions"],
                              permutations=k)["ball"]
            print(f"{model} permutations={k}:",
                  {o: round(p, 3) for o, p in dist.items()}, flush=True)
        del dm
        gc.collect()
        torch.cuda.empty_cache()

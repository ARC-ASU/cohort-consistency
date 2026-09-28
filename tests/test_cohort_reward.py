"""
CPU-only smoke test of the cohort reward manager against a mock OpenAI-compatible endpoint.

    PYTHONPATH=training/verl python tests/test_cohort_reward.py --tokenizer Qwen/Qwen2.5-Coder-7B-Instruct

The mock retriever answers True to every atomic query and "idk" to queries containing "and then";
the mock judge returns score 7 and the unchanged program. Expected rewards follow the formulas in
verl/workers/reward_manager/cohort.py.
"""
import argparse
import json
import os
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "training", "verl"))


class MockLLM(BaseHTTPRequestHandler):

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        messages = body["messages"]
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if system.startswith("You are a fact-lookup assistant"):
            answer = '"idk"' if "and then" in user else "true"
            content = "[Explanation] mock.```json{\"answer\": %s}```" % answer
        else:  # judge: echo the program under evaluation
            program = user.split("(to be evaluated and then improved)\n\n", 1)[1].split("\nNOTE:", 1)[0]
            content = "```json\n" + json.dumps({"score": 7, "program": program}) + "\n```"
        payload = {
            "id": "mock", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def start_mock_server():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = HTTPServer(("127.0.0.1", port), MockLLM)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return port


GOOD = '''```python
def answer(Fruit: str) -> int:
    is_red = retrieve(f"Is {Fruit} red?", bool)
    is_round = retrieve(f"Is {Fruit} round?", bool)
    if is_red and is_round:
        return 1
    return 0
```'''

REJECTED = '''```python
def answer(Fruit: str) -> int:
    x = retrieve(f"Find the color of {Fruit} and then check whether it is red", bool)
    return 1 if x else 0
```'''

NO_CODE = "I think the answer is Yes."


def make_batch(tokenizer, responses, cohort):
    from verl import DataProto
    prompt = tokenizer("Masked Question: Is Fruit red and round?", return_tensors="pt")["input_ids"][0]
    prompt_len, resp_len = 32, 256
    prompts, resps, masks = [], [], []
    for text in responses:
        r = tokenizer(text, return_tensors="pt")["input_ids"][0][:resp_len]
        p_pad = torch.full((prompt_len,), tokenizer.pad_token_id or 0)
        p_pad[-len(prompt):] = prompt
        r_pad = torch.full((resp_len,), tokenizer.pad_token_id or 0)
        r_pad[:len(r)] = r
        mask = torch.zeros(prompt_len + resp_len, dtype=torch.long)
        mask[prompt_len - len(prompt):prompt_len] = 1
        mask[prompt_len:prompt_len + len(r)] = 1
        prompts.append(p_pad), resps.append(r_pad), masks.append(mask)
    n = len(responses)
    non_tensors = {
        "data_source": np.array(["conceptual"] * n, dtype=object),
        "similar_questions": np.array([json.dumps(cohort)] * n, dtype=object),
        "abstraction": np.array([json.dumps({"masked_question": "Is Fruit red and round?",
                                             "parameters": {"Fruit": "apple"}})] * n, dtype=object),
        "domain": np.array(["StrategyQA"] * n, dtype=object),
        "reasoning": np.array(["Check color, then shape."] * n, dtype=object),
        "choices": np.array([["No", "Yes"]] * n, dtype=object),
    }
    return DataProto.from_dict(tensors={"prompts": torch.stack(prompts), "responses": torch.stack(resps),
                                        "attention_mask": torch.stack(masks)},
                               non_tensors=non_tensors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    args = parser.parse_args()

    port = start_mock_server()
    os.environ["COHORT_LLM_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
    os.environ["COHORT_LLM_MODEL"] = "mock"

    from transformers import AutoTokenizer
    from verl.workers.reward_manager import CohortRewardManager

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # the mock retriever says True to everything, so GOOD returns 1 ("B" = Yes) on every question:
    # 4 of the 6 cohort questions are labelled "B"
    cohort = [{"question": f"q{i}", "answer": "B" if i < 4 else "A", "parameters": {"Fruit": f"fruit{i}"}}
              for i in range(6)]
    batch = make_batch(tokenizer, [GOOD, REJECTED, NO_CODE], cohort)

    def rewards(**kwargs):
        rm = CohortRewardManager(tokenizer=tokenizer, num_examine=0, **kwargs)
        details = {}
        scores = rm(batch, details).sum(-1).tolist()
        return [round(s, 4) for s in scores], details

    checks = [
        # GOOD: R_acc 4*0.2 + R_ret 6*0.1 + R_fc 7*0.06 + R_sa 1.0*0.6 = 2.42
        # REJECTED: R_ret 0 (one call) + R_rej 6*(-0.1) + R_fc 0.42 + R_sa 0.6 = 0.42
        (dict(gate_k=3), [2.42, 0.42, 0.0]),
        (dict(gate_k=5), [1.62, 0.42, 0.0]),                       # 4 correct < 5: no accuracy reward
        (dict(gate_k=0, use_judge=False), [1.4, -0.6, 0.0]),      # Exec only
        (dict(gate_k=0, use_judge=False, use_exec_shaping=False), [0.8, 0.0, 0.0]),  # accuracy only
    ]
    ok = True
    for kwargs, expected in checks:
        got, details = rewards(**kwargs)
        passed = np.allclose(got, expected, atol=1e-4)
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'} {kwargs}: got {got}, expected {expected}")
    print("details (last run):", {k: round(v, 4) for k, v in details.items()})
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

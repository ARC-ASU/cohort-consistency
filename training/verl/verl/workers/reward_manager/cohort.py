"""
Cohort reward manager.

For every rollout the policy emits one program ``def answer(...) -> int``. The program is
executed *unchanged* on every question of its cohort (1 original + 5 similar questions);
``retrieve(...)`` calls inside the program are answered by the retriever LLM, which replies
"idk" (raised as ``ValueError("idk")``) to non-atomic queries.

Composite reward (paper Sec. 3.3 and App. "Reward Design"):

    R = R_acc + R_ret + R_rej + R_fc + R_sa

    R_acc  accuracy reward     acc_weight * n_correct              (0.2 per correct question)
           with cohort gating: granted only if n_correct >= gate_k  (gate_k=3 -> RL_Consistency,
                                                                     gate_k=0 -> RL_Normal)
    R_ret  retrieve usage      per question: -0.1 (0 calls) / 0 (1 call) / +0.1 (>1 calls)
    R_rej  rejection penalty   per question: -0.1 if a retrieve call was rejected ("idk")
    R_fc   factor-complete     fc_weight * judge score s_fc in [1, 10]
    R_sa   structural align.   sa_weight * AST similarity(p, p+) in [0, 1]

R_ret / R_rej are the execution-based shaping terms ("Exec"); R_fc / R_sa come from the
frozen judge ("Crit").
"""

import ast
import json
import multiprocessing
import os
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import torch

from verl import DataProto
from verl.trainer.judge import batch_judge

_DEFAULT_HEADER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "cohort", "retriever_header.py"))

# Cache of executed program environments, keyed by hash(code)
_CODE_ENV_CACHE = {}


def analyze_code(code_text):
    """Count the ``retrieve(...)`` calls in a program."""
    try:
        tree = ast.parse(code_text)
        n_calls = sum(1 for node in ast.walk(tree)
                      if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'retrieve')
    except SyntaxError:
        n_calls = 0
    return {"number_of_retrieve_batch_calls": n_calls}


class TimeoutError(Exception):
    pass


def run_with_timeout_thread(func, args=(), kwargs={}, timeout_sec=60):
    """Thread-safe timeout implementation."""
    result_container = []
    exception_container = []
    completed = threading.Event()

    def target():
        try:
            result_container.append(func(*args, **kwargs))
        except Exception as e:
            exception_container.append(e)
        finally:
            completed.set()

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()

    if not completed.wait(timeout_sec):
        raise TimeoutError("Thread execution exceeded allowed time")
    if exception_container:
        raise exception_container[0]
    if result_container:
        return result_container[0]
    return None


def extract_code_from_text(text):
    """Extract the last Python code block from markdown code blocks."""
    matches = re.findall(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    return matches[-1] if matches else None


def prepare_execution_environment(code, header_path):
    """Execute the retriever header followed by the program; return (env, has_answer_fn)."""
    cache_key = hash(code)
    if cache_key in _CODE_ENV_CACHE:
        return _CODE_ENV_CACHE[cache_key], True

    try:
        try:
            with open(header_path, "r") as f:
                header_code = f.read()
        except Exception:
            header_code = ""

        env = {}
        env.update(globals())
        run_with_timeout_thread(exec, (header_code + "\n" + code, env), timeout_sec=60)

        if "answer" not in env:
            return env, False

        _CODE_ENV_CACHE[cache_key] = env
        return env, True
    except Exception:
        return {}, False


class CohortRewardManager:
    """Executes each program on its cohort and combines execution and judge rewards."""

    def __init__(self,
                 tokenizer,
                 num_examine,
                 compute_score=None,
                 gate_k=3,
                 use_exec_shaping=True,
                 use_judge=True,
                 acc_weight=0.2,
                 fc_weight=0.06,
                 sa_weight=0.6,
                 judge_batch_size=30,
                 header_path=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.gate_k = int(gate_k)
        self.use_exec_shaping = bool(use_exec_shaping)
        self.use_judge = bool(use_judge)
        self.acc_weight = float(acc_weight)
        self.fc_weight = float(fc_weight)
        self.sa_weight = float(sa_weight)
        self.judge_batch_size = int(judge_batch_size)
        self.header_path = header_path or os.environ.get("COHORT_RETRIEVER_HEADER", _DEFAULT_HEADER)
        assert os.path.exists(self.header_path), f"retriever header not found: {self.header_path}"

        self.max_workers = min(40, multiprocessing.cpu_count() + 4)
        self.print_lock = Lock()
        self.printed_sources = defaultdict(int)

    def __call__(self, data: DataProto, detail_rewards=None):
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # Step 1: one job per (rollout, cohort question)
        all_jobs = []
        for data_idx in range(len(data)):
            data_item = data[data_idx]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            if valid_response_length <= 0:
                continue

            sequences_str = self.tokenizer.decode(torch.cat((valid_prompt_ids, valid_response_ids)))

            data_source = data_item.non_tensor_batch['data_source']
            with self.print_lock:
                if self.printed_sources[data_source] < self.num_examine:
                    print(sequences_str)
                    self.printed_sources[data_source] += 1

            code = extract_code_from_text(sequences_str)
            if not code:
                continue

            try:
                similar_questions = json.loads(data_item.non_tensor_batch['similar_questions'])
                masked_question = json.loads(data_item.non_tensor_batch['abstraction'])["masked_question"]
                domain = data_item.non_tensor_batch['domain']
            except Exception:
                continue

            env, env_valid = prepare_execution_environment(code, self.header_path)
            if not env_valid:
                continue

            for q_idx, question in enumerate(similar_questions):
                all_jobs.append({
                    'data_idx': data_idx,
                    'q_idx': q_idx,
                    'env': env,
                    'question': question,
                    'domain': domain,
                    'valid_response_length': valid_response_length,
                    'code': code,
                    'masked_question': masked_question,
                    'reasoning': data_item.non_tensor_batch['reasoning'],
                    'choices': data_item.non_tensor_batch['choices'],
                })

        # Step 2: execute all (program, question) pairs in one thread pool
        results_by_data_idx = defaultdict(lambda: {
            'flags': [],
            'num_retrieve': [],
            'ret_count': [],
            'simple_score': [],
            'results': [],
            'extra': {},
            'valid_response_length': 0,
        })

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_job = {
                executor.submit(self._process_single_question, job['env'], job['question'], job['domain'], job['code']):
                job for job in all_jobs
            }
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                data_idx = job['data_idx']
                try:
                    flag, num_retrieve, ret_count, simple_score = future.result()
                except Exception:
                    continue
                res = results_by_data_idx[data_idx]
                res['flags'].append(flag)
                res['num_retrieve'].append(num_retrieve)
                res['ret_count'].append(ret_count)
                res['simple_score'].append(simple_score)

                gold = job["question"]["answer"]
                pred = gold if flag else ("B" if gold == "A" else "A")
                res['results'].append((job["question"]["question"], gold, pred))
                res['extra'] = {
                    'masked_question': job['masked_question'],
                    'reasoning': job['reasoning'],
                    'choices': job['choices'],
                    'code': job['code'],
                    'domain': job['domain'],
                }
                res['valid_response_length'] = job['valid_response_length']

        # Step 3: judge critique (batched), then combine
        data_indices = [i for i, r in results_by_data_idx.items() if r["extra"]]
        if not data_indices:
            return reward_tensor

        if self.use_judge:
            batch_inputs = [{
                "masked_question": results_by_data_idx[i]["extra"]["masked_question"],
                "similar_questions": results_by_data_idx[i]["results"],
                "reasoning": results_by_data_idx[i]["extra"]["reasoning"],
                "program": results_by_data_idx[i]["extra"]["code"],
                "choices": results_by_data_idx[i]["extra"]["choices"],
            } for i in data_indices]
            judge_results = batch_judge(batch_inputs, batch_size=self.judge_batch_size)
        else:
            judge_results = [(0, 0.0, None)] * len(data_indices)

        for data_idx, (judge_score_raw, program_score_raw, _new_program) in zip(data_indices, judge_results):
            res = results_by_data_idx[data_idx]
            valid_response_length = res['valid_response_length']
            if valid_response_length <= 0:
                continue

            n_correct = sum(res['flags'])
            acc_score = self.acc_weight * n_correct
            if self.gate_k > 0 and n_correct < self.gate_k:
                acc_score = 0.0

            ret_score = sum(res['num_retrieve']) if self.use_exec_shaping else 0.0
            rej_score = sum(res['simple_score']) if self.use_exec_shaping else 0.0
            judge_score = judge_score_raw * self.fc_weight if self.use_judge else 0.0
            program_score = program_score_raw * self.sa_weight if self.use_judge else 0.0

            total_score = acc_score + ret_score + rej_score + judge_score + program_score

            if detail_rewards is not None:
                ret_count = res['ret_count']
                detail_rewards["accuracy"] = detail_rewards.get("accuracy", 0) + acc_score
                detail_rewards["num_correct"] = detail_rewards.get("num_correct", 0) + n_correct
                detail_rewards["num_retrieve"] = detail_rewards.get("num_retrieve", 0) + ret_score
                detail_rewards["exact_number_retrieve"] = detail_rewards.get(
                    "exact_number_retrieve", 0) + (sum(ret_count) / len(ret_count) if ret_count else 0)
                detail_rewards["simple_score"] = detail_rewards.get("simple_score", 0) + rej_score
                detail_rewards["judge_score"] = detail_rewards.get("judge_score", 0) + judge_score
                detail_rewards["program_score"] = detail_rewards.get("program_score", 0) + program_score

            reward_tensor[data_idx, valid_response_length - 1] = total_score

        return reward_tensor

    def _process_single_question(self, env, question, domain, code):
        """Run the program on one cohort question.

        Returns (correct, retrieve_usage_reward, n_retrieve_calls, rejection_penalty).
        """
        try:
            flag = False
            num_retrieve = 0
            simple_score = 0

            ret_count = analyze_code(code)["number_of_retrieve_batch_calls"]
            if ret_count == 0:
                num_retrieve = -0.1
                return flag, num_retrieve, ret_count, simple_score
            num_retrieve = 0 if ret_count == 1 else 0.1

            if "answer" not in env:
                return flag, num_retrieve, ret_count, simple_score

            try:
                result = run_with_timeout_thread(env["answer"], kwargs=question["parameters"], timeout_sec=60)
            except ValueError as e:
                if str(e) == "idk":  # rejected by the retriever
                    simple_score = -0.1
                return flag, num_retrieve, ret_count, simple_score
            except Exception:
                return flag, num_retrieve, ret_count, simple_score

            options = ["A", "B"]
            if isinstance(result, int) and result in [0, 1] and options[result] == question["answer"]:
                flag = True
            return flag, num_retrieve, ret_count, simple_score
        except Exception:
            return False, 0, 0, 0

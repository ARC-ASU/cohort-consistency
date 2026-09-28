# Learning Verifiable Reasoning Programs through Cohort Consistency

GRPO training with the cohort reward from the AACL-IJCNLP 2026 paper *Learning Verifiable Reasoning
Programs through Cohort Consistency* (Xiao Ye, Shaswat Shrivastava, Zhaonan Li, Jacob Dineen, Shijie Lu,
Avneet Ahuja, Ming Shen, Zhikun Xu, Ben Zhou).

The policy writes one short executable program `def answer(...) -> int` for a masked abstraction of a
question. The program may only call an atomic `retrieve(question, type)` plus simple control flow, and it is
executed **unchanged** on a cohort of factually varied questions (the original question + 5 similar
questions). Cohort-level consistency, combined with judge critique, is the reward.

```
training/verl/                       verl v0.2 (commit 0c32cf7) with the cohort reward
  verl/workers/reward_manager/cohort.py   executes programs on the cohort, computes the composite reward
  verl/trainer/judge.py                   frozen judge for the critique-based rewards
cohort/retriever_header.py           retrieve() with the rejection prompt, prepended to every program
training/scripts/                    serve the retriever/judge, launch GRPO
tests/test_cohort_reward.py          CPU-only check of the reward against a mock LLM endpoint
```

## Setup

The experiments used Python 3.9, CUDA 12.4, `torch==2.4.0`, `vllm==0.6.3`, `transformers==4.51.3`, `ray==2.43.0`.

```bash
conda create -n cohort python=3.9 -y && conda activate cohort
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e training/verl --no-deps
```

Smoke test (no GPU needed; checks the composite reward against hand-computed values):

```bash
PYTHONPATH=training/verl python tests/test_cohort_reward.py --tokenizer Qwen/Qwen2.5-Coder-7B-Instruct
```

## Retriever and judge

`retrieve(...)` calls inside programs and the judge are served by one OpenAI-compatible endpoint. The paper
uses Qwen2.5-7B-Instruct (Qwen2.5-3B-Instruct for the 3B policy).

```bash
CUDA_VISIBLE_DEVICES=0 bash training/scripts/serve_llm.sh Qwen/Qwen2.5-7B-Instruct 5500
export COHORT_LLM_BASE_URL=http://0.0.0.0:5500/v1
export COHORT_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct   # the name passed to `vllm serve`
```

The retriever is prompted to answer only single-step fact lookups and to reply `idk` otherwise
(`cohort/retriever_header.py`); a rejected call raises `ValueError("idk")`.

## Training data

`train.parquet` / `val.parquet`, one row per cohort, with the columns read by verl and by the reward manager:

| Column | Content |
|---|---|
| `prompt` | chat messages for the policy (verl applies the chat template): masked question, parameter names, options and the function header `def answer(<param>: <type>, ...) -> int` |
| `similar_questions` | JSON string: the cohort, `[{"question", "answer", "parameters"}, ...]`, original question first, then 5 similar questions; `answer` is `"A"`/`"B"` and `parameters` are the keyword arguments passed to `answer(...)` |
| `abstraction` | JSON string: `{"masked_question", "parameters"}` |
| `choices` | the two answer options; the program returns the index of the chosen option |
| `reasoning` | a cohort-level reasoning path, shown to the judge |
| `domain`, `data_source`, `reward_model` | bookkeeping (`reward_model = {"style": "rule", "ground_truth": ...}`) |

## Training

```bash
POLICY_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct DATA_DIR=data/rl CKPT_DIR=checkpoints/consistency_7b \
VARIANT=consistency_exec_crit N_GPUS=2 bash training/scripts/run_grpo.sh
```

GRPO with 5 rollouts per prompt, batch size 128, learning rate 1e-5, KL coefficient 0.001 (low-variance KL
loss), rollout temperature 1.0, on two H200 GPUs. Checkpoints are saved every 5 steps as FSDP shards; convert
one to a Hugging Face model with
`python training/verl/scripts/model_merger.py --local_dir <CKPT_DIR>/global_step_<N>/actor`.

| `VARIANT` | Paper | Reward |
|---|---|---|
| `consistency_exec_crit` | RL<sub>Consistency</sub> (Exec+Crit) | cohort-gated R<sub>acc</sub> + R<sub>ret</sub> + R<sub>rej</sub> + R<sub>fc</sub> + R<sub>sa</sub> |
| `normal_exec_crit` | RL<sub>Normal</sub> (Exec+Crit) | per-question R<sub>acc</sub> + R<sub>ret</sub> + R<sub>rej</sub> + R<sub>fc</sub> + R<sub>sa</sub> |
| `consistency_exec` | RL<sub>Consistency</sub> (Exec) | cohort-gated R<sub>acc</sub> + R<sub>ret</sub> + R<sub>rej</sub> |
| `normal_acc` | RL<sub>Normal</sub> (Acc) | per-question R<sub>acc</sub> |

## Cohort reward

`training/verl/verl/workers/reward_manager/cohort.py`. Each rollout's program is executed on all 6 cohort
questions; `retrieve` calls go to the retriever endpoint.

| Term | Definition |
|---|---|
| R<sub>acc</sub> | 0.2 × number of correct cohort questions (out of 6); with cohort gating it is granted only if at least `gate_k` questions are correct (`GATE_K`, default 3) |
| R<sub>ret</sub> | per question: −0.1 if the program makes no `retrieve` call, 0 for one call, +0.1 for more than one |
| R<sub>rej</sub> | per question: −0.1 if a `retrieve` call is rejected (`idk`) |
| R<sub>fc</sub> | 0.06 × judge score (1–10) for factor-complete decomposition |
| R<sub>sa</sub> | 0.6 × AST similarity between the program and the judge's improved program |

The weights and switches are in the `reward_model.cohort` block of
`training/verl/verl/trainer/config/ppo_trainer.yaml`.

Changes to verl v0.2 (`0c32cf7`): `reward_manager=cohort` and its config block
(`verl/trainer/main_ppo.py`, `verl/trainer/config/ppo_trainer.yaml`), the reward manager
(`verl/workers/reward_manager/cohort.py`), the judge (`verl/trainer/judge.py`), and per-component reward
logging in `verl/trainer/ppo/ray_trainer.py`.

## Citation

```bibtex
@inproceedings{ye2026cohort,
  title     = {Learning Verifiable Reasoning Programs through Cohort Consistency},
  author    = {Ye, Xiao and Shrivastava, Shaswat and Li, Zhaonan and Dineen, Jacob and Lu, Shijie and
               Ahuja, Avneet and Shen, Ming and Xu, Zhikun and Zhou, Ben},
  booktitle = {Proceedings of the Asia-Pacific Chapter of the Association for Computational Linguistics (AACL-IJCNLP)},
  year      = {2026}
}
```

## Acknowledgments

This work is supported in part by the National Cancer Institute of the National Institutes of Health under
Award Number R01CA323939, as part of the NSF/NIH Smart Health and Biomedical Research in the Era of
Artificial Intelligence and Advanced Data Science Program. The content is solely the responsibility of the
authors and does not necessarily represent the official views of the National Institutes of Health.

The RL implementation builds on [verl](https://github.com/volcengine/verl) (Apache-2.0).

## License

Apache-2.0 (see `LICENSE`).

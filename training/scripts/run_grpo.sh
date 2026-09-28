#!/usr/bin/env bash
# GRPO training with the cohort reward (paper Sec. 3.3). Requires a running retriever/judge
# endpoint (serve_llm.sh) and train.parquet / val.parquet in the format described in README.md.
#
#   POLICY_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct DATA_DIR=data/rl VARIANT=consistency_exec_crit \
#   CKPT_DIR=checkpoints/consistency_7b bash training/scripts/run_grpo.sh
#
# VARIANT (paper Tables 1-2)
#   consistency_exec_crit  RL_Consistency (Exec+Crit): cohort-gated R_acc + R_ret + R_rej + R_fc + R_sa
#   normal_exec_crit       RL_Normal (Exec+Crit):      per-question R_acc + R_ret + R_rej + R_fc + R_sa
#   consistency_exec       RL_Consistency (Exec):      cohort-gated R_acc + R_ret + R_rej
#   normal_acc             RL_Normal (Acc):            per-question R_acc only
# For the RL_Org ablation use cohorts that contain only the original question and
# VARIANT=normal_exec_crit (a single-question cohort cannot pass the cohort gate).
# Checkpoints are written every 5 steps (trainer.save_freq); extra hydra overrides can be appended,
# e.g. trainer.total_training_steps=60.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
POLICY_MODEL=${POLICY_MODEL:?set POLICY_MODEL}
DATA_DIR=${DATA_DIR:?set DATA_DIR (contains train.parquet / val.parquet)}
CKPT_DIR=${CKPT_DIR:?set CKPT_DIR}
VARIANT=${VARIANT:-consistency_exec_crit}
N_GPUS=${N_GPUS:-2}
GATE_K=${GATE_K:-3}

case "$VARIANT" in
  consistency_exec_crit) GATE=$GATE_K; EXEC=True;  JUDGE=True ;;
  normal_exec_crit)      GATE=0;       EXEC=True;  JUDGE=True ;;
  consistency_exec)      GATE=$GATE_K; EXEC=True;  JUDGE=False ;;
  normal_acc)            GATE=0;       EXEC=False; JUDGE=False ;;
  *) echo "unknown VARIANT=$VARIANT" >&2; exit 1 ;;
esac

export HYDRA_FULL_ERROR=1
export COHORT_RETRIEVER_HEADER=${COHORT_RETRIEVER_HEADER:-$ROOT/cohort/retriever_header.py}
cd "$ROOT/training/verl"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/val.parquet" \
  data.train_batch_size=128 \
  data.val_batch_size=8 \
  data.max_prompt_length=2048 \
  data.max_response_length=512 \
  actor_rollout_ref.model.path="$POLICY_MODEL" \
  actor_rollout_ref.actor.optim.lr=1e-5 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.grad_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.n=5 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  reward_model.reward_manager=cohort \
  reward_model.cohort.gate_k=$GATE \
  reward_model.cohort.use_exec_shaping=$EXEC \
  reward_model.cohort.use_judge=$JUDGE \
  trainer.critic_warmup=0 \
  trainer.logger=['console'] \
  trainer.project_name=cohort_consistency \
  trainer.experiment_name="$VARIANT" \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.n_gpus_per_node=$N_GPUS \
  trainer.nnodes=1 \
  +trainer.val_before_train=False \
  trainer.save_freq=5 \
  trainer.test_freq=5 \
  trainer.total_epochs=${EPOCHS:-5} "$@"

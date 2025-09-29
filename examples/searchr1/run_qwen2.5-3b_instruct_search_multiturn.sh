# run on 8xH20
# make sure your current working directory is the root of the project

set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"


TRAIN_DATA="$HOME/data/searchR1_processed_direct/train.parquet"
VAL_DATA="$HOME/data/searchR1_processed_direct/test.parquet"

TOOL_CONFIG="./examples/searchr1/search_tool_config.yaml"

BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_batch_size=512 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.shuffle=True \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.max_model_len=15000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.top_p=1 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=500 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.015 \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size_per_gpu=32 \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    trainer.critic_warmup=0 \
    algorithm.use_kl_in_reward=False \
    trainer.val_before_train=False \
    trainer.val_only=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='search_r1_like_async_rl' \
    trainer.experiment_name='qwen2.5-3b-instruct_function_rm-search-async-vllm-multi-w-searchtool-verify-n16' \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_training_steps=1005 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA"  \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" $@

#actor_rollout_ref.actor.kl_loss_coef=0.001 \ # True for GRPO
#actor_rollout_ref.actor.kl_loss_type=low_var_kl \


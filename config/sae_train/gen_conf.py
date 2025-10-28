import yaml

template = """deploy: True
tag: {tag}
seed: {seed}

device : "cuda:0"
bf16: False
epochs: 1

model_name: '{model_name}'

data:
  path: '../data/labeled_sentences_large_deduped_train.jsonl'
  dev_path: '../data/labeled_sentences_large_deduped_dev.jsonl'
  corr_config: {corr_config}
  batch_size: {batch_size}
  num_workers: 2
  num_iters: {num_iters}
  max_sample_length: 64
  run_validation: False

sae:
  sae_type: {sae_type} # ['relu', 'jumprelu', 'topk', 'sparsemax_dist']
  exp_factor: 8
  kval_topk: 128
  mp_kval: 128
  gamma_reg: 0.01 # ['default' <-- stick with this in general]
  encoder_reg: True

optimizer:
  learning_rate: 1.0e-3
  weight_decay: 1.0e-4
  beta1: 0.9
  beta2: 0.95
  grad_clip: 1.0
  decay_lr: True
  warmup_iters: 200
  min_lr: 9.0e-4

eval:
  save_tables: False

log: 
  save_multiple: False
  log_interval: 10
  eval_interval: 1000
  save_interval: 1000


# Nested configs. Disable hydra logging
defaults:
  - _self_
  - override hydra/job_logging: disabled
  - override hydra/hydra_logging: disabled

# Disable hydra directory structure
hydra:
  output_subdir: Null
  job:
    chdir: False
  run:
    dir: .

  sweep:
    dir: .
    subdir: .
"""

# default_batch_size = 64
default_batch_size = 128
default_num_steps = 10000

# model_name = "EleutherAI/pythia-70m-deduped"
model_name = "google/gemma-2-2b"

for corr in ("null", 0.1, 0.2, 0.5, 0.9, 1.0):
    for sae_type in ("relu", "topk", "sparsemax_dist"):
    # for sae_type in ("sparsemax_dist",):
        for seed in (42, 2718, 5772):
          if corr == "null":
              tag = f"{sae_type}_vp-tpa_uniform"
              corr_config = "null"
          else:
              tag = f"{sae_type}_corr_vp-tpa_{corr}"
              corr_config = f"\'../configs/corr_vp-tpa_{corr}.jsonl\'"
          
          batch_size = default_batch_size
          num_steps = default_num_steps
          if sae_type in ("sparsemax_dist",):
              batch_size = default_batch_size // 2
          #     num_steps = default_num_steps * 2

          model_basename = model_name.split("/")[-1]
          tag = f"\'{model_basename}/{tag}\'"

          config_str = template.format(tag=tag, corr_config=corr_config,
                                      model_name=model_name, sae_type=f"\'{sae_type}\'",
                                      batch_size=batch_size, num_iters=num_steps,
                                      seed=seed)
          # data = yaml.safe_load(config_str)
          with open(f"conf_{model_basename}_{sae_type}_corr_vp-tpa-{corr}_seed-{seed}.yaml", 'w') as handle:
              handle.write(config_str)
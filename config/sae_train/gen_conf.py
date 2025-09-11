import yaml

template = """deploy: True
tag: {tag}
seed: 42

device : "cuda:0"
bf16: True
epochs: 1

model_name: '{model_name}'

data:
  path: '../data/labeled_sentences_large_deduped_train.jsonl'
  corr_config: {corr_config}
  batch_size: 64
  num_workers: 2
  num_iters: 10000
  max_sample_length: 64

sae:
  sae_type: {sae_type} # ['relu', 'jumprelu', 'topk', 'sparsemax_dist']
  exp_factor: 8
  kval_topk: 100
  mp_kval: 100
  gamma_reg: 0.001 # ['default' <-- stick with this in general]
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
  save_interval: 100


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

model_name = "google/gemma-2-2b"

for corr in ("null", 0.1, 0.2, 0.5, 0.9, 1.0):
    for sae_type in ("relu", "topk", "sparsemax_dist", "MP"):
    # for sae_type in ("MP",):
        if corr == "null":
            tag = f"{sae_type}_uniform"
            corr_config = "null"
        else:
            tag = f"{sae_type}_corr_{corr}"
            corr_config = f"\'../configs/corr_ds-sp_{corr}.jsonl\'"
        model_basename = model_name.split("/")[-1]
        tag = f"\'{model_basename}/{tag}\'"

        config_str = template.format(tag=tag, corr_config=corr_config,
                                     model_name=model_name, sae_type=f"\'{sae_type}\'")
        # data = yaml.safe_load(config_str)
        with open(f"conf_{model_basename}_{sae_type}_corr-{corr}.yaml", 'w') as handle:
            handle.write(config_str)
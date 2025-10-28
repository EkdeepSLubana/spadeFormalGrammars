import hydra
import yaml
import torch
import torch.nn.functional as F
import os
import numpy as np
import pickle as pkl

from model import GPT
from dgp import get_dataloader, get_dataloader_json
from sae_test import SAE, step_fn

from utils import init_wandb, cleanup
from utils import set_seed, save_config, open_log
from utils import update_cosine_warmup_lr
from utils import save_sae, move_to_device, log_sae_train
from transformers import AutoModelForCausalLM, AutoTokenizer


# @hydra.main(config_path="./config/sae_train", config_name="conf_gemma-2-2b_MP.yaml", version_base="1.3")
@hydra.main(config_path="./config/sae_train", config_name="conf_gemma-2-2b_spade.yaml", version_base="1.3")
def main_hf(sae_cfg):
    # Setup Environment
    set_seed(sae_cfg.seed)
    # init_wandb(sae_cfg, project_name=sae_cfg.log.wandb_project_name)
    save_config(sae_cfg)
    # fp = open_log(sae_cfg)

    print(sae_cfg)
    device = sae_cfg.device if torch.cuda.is_available() else 'cpu'

    base_model_name = sae_cfg.model_name
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    base_model_tok = AutoTokenizer.from_pretrained(base_model_name)

    # Load model parameters
    # base_model.load_state_dict(base_model_dict['net'])
    base_model.to(device)
    # base_model_tok.to(device)
    # print("number of parameters: %.2fM" % (base_model.get_num_params()/1e6,))

    # Define dataloader
    dataloader = get_dataloader_json(
        path=sae_cfg.data.path,
        corr_config=sae_cfg.data.corr_config,
        num_iters=sae_cfg.data.num_iters * sae_cfg.data.batch_size,
        max_sample_length=sae_cfg.data.max_sample_length,
        seed=sae_cfg.seed,
        batch_size=sae_cfg.data.batch_size,
        num_workers=sae_cfg.data.num_workers,
        model_name=sae_cfg.model_name
    )

    dev_dataloader = get_dataloader_json(
        path=sae_cfg.data.dev_path,
        corr_config=sae_cfg.data.corr_config,
        num_iters=sae_cfg.data.num_iters * sae_cfg.data.batch_size,
        max_sample_length=sae_cfg.data.max_sample_length,
        seed=sae_cfg.seed,
        batch_size=sae_cfg.data.batch_size // 2,
        num_workers=sae_cfg.data.num_workers,
        model_name=sae_cfg.model_name
    )
    

    # Define SAE
    dimin = base_model.config.hidden_size
    sae = SAE(
        dimin=dimin,
        width=int(dimin*sae_cfg.sae.exp_factor), 
        sae_type=sae_cfg.sae.sae_type,
        kval_topk=sae_cfg.sae.kval_topk if sae_cfg.sae.sae_type=='topk' else None,
        mp_kval=sae_cfg.sae.mp_kval if sae_cfg.sae.sae_type=='MP' else None,
    )
    sae.to(device)
    # print("number of parameters: %.2fM" % (base_model.get_num_params()/1e6,))

    # Optimizer
    optimizer = torch.optim.AdamW(sae.parameters(), lr=sae_cfg.optimizer.learning_rate,
        betas=(sae_cfg.optimizer.beta1, sae_cfg.optimizer.beta2),
        weight_decay=sae_cfg.optimizer.weight_decay)

    # Train
    train(sae_cfg, sae, base_model, base_model_tok, dataloader, optimizer, device, encoder_reg=sae_cfg.sae.encoder_reg, dev_dataloader=dev_dataloader)

    # Close wandb and log file
    # cleanup(sae_cfg, fp)


# @hydra.main(config_path="./config/sae_train", config_name="conf.yaml", version_base="1.3")
def main(sae_cfg):
    # init_wandb(sae_cfg, project_name="pcfg_saes_training")
    set_seed(sae_cfg.seed)
    save_config(sae_cfg)
    fp = open_log(sae_cfg)
    device = sae_cfg.device if torch.cuda.is_available() else 'cpu'

    # Define model
    base_model_dir = sae_cfg.pt_model_dir
    base_model_dict = torch.load(os.path.join(base_model_dir, 'latest_ckpt.pt'), map_location=device)
    base_model_cfg = base_model_dict['config']
    with open(os.path.join(base_model_dir, 'grammar/PCFG.pkl'), 'rb') as f:
        pcfg = pkl.load(f)
    base_model = GPT(base_model_cfg.model, pcfg.vocab_size)

    # Load model parameters
    base_model.load_state_dict(base_model_dict['net'])
    base_model.to(device)
    print("number of parameters: %.2fM" % (base_model.get_num_params()/1e6,))


    # Define dataloader
    dataloader = get_dataloader(
        language=base_model_cfg.data.language,
        config=base_model_cfg.data.config,
        alpha=base_model_cfg.data.alpha,
        prior_type=base_model_cfg.data.prior_type,
        num_iters=base_model_cfg.data.num_iters * sae_cfg.data.batch_size,
        max_sample_length=base_model_cfg.data.max_sample_length,
        seed=base_model_cfg.seed,
        batch_size=sae_cfg.data.batch_size,
        num_workers=sae_cfg.data.num_workers,
    )

    # Define SAE
    dimin = base_model.config.hidden_size
    sae = SAE(
        dimin=dimin,
        width=int(dimin*sae_cfg.sae.exp_factor), 
        sae_type=sae_cfg.sae.sae_type,
        kval_topk=sae_cfg.sae.kval_topk if sae_cfg.sae.sae_type=='topk' else None,
        mp_kval=sae_cfg.sae.mp_kval if sae_cfg.sae.sae_type=='MP' else None,
    )
    sae.to(device)
    print("number of parameters: %.2fM" % (base_model.get_num_params()/1e6,))

    # Optimizer
    optimizer = torch.optim.AdamW(sae.parameters(), lr=sae_cfg.optimizer.learning_rate,
        betas=(sae_cfg.optimizer.beta1, sae_cfg.optimizer.beta2),
        weight_decay=sae_cfg.optimizer.weight_decay)

    # Train
    train(sae_cfg, sae, base_model, dataloader, optimizer, device, encoder_reg=sae_cfg.sae.encoder_reg)

    # Close wandb and log file
    # cleanup(sae_cfg, fp)


def train(sae_cfg, sae, base_model, base_model_tok, dataloader, optimizer, device, encoder_reg=True, dev_dataloader=None):
    """
    Training function
    """
    # Set SAE to train mode and base model to eval mode
    sae.train()
    base_model.eval()
    mid_layer = base_model.config.num_hidden_layers // 2
    
    # Hook to get activations for SAE training
    activations = []
    def getActivation():
        """
        Function to define hooks for storing activations.
        """
        def hook(model, input, output):
            if type(output) is tuple:
                output = output[0]
            activations.append(output.detach().clone())
        return hook

    # Attach hook to get activations
    act_hook = base_model.gpt_neox.layers[mid_layer].register_forward_hook(getActivation())
    # act_hook = base_model.model.layers[mid_layer].register_forward_hook(getActivation())

    # Activation scaling (ensure E[||x||_2] = 1)
    # acts_scaling = 1 / 236.184564 
    acts_scaling = 1 / 64

    # Data type (bf16 for efficiency)
    dt = torch.bfloat16 if sae_cfg.bf16 else torch.float32

    # Configuration
    total_steps = len(dataloader)

    # Initialize SAE training log
    train_log = {
        "total": [],
        "mse": [],
        "nmse": [],
        "reg": [],
        "variance_explained": [],
        "p_sparsity": [],
        "lambda": []
    }
    val_log = {
        "mse": [],
        "nmse": [],
        "variance_explained": []
    }

    # Some hparams variables
    lr, it = 0.0, 0
    if sae_cfg.sae.gamma_reg == 'default':
        gamma_reg = 0.01 if sae_cfg.sae.sae_type == 'jumprelu' else 0.1
    else:
        gamma_reg = sae_cfg.sae.gamma_reg
    print("Total training steps: ", total_steps)
    print("Learning rate warmup steps: ", sae_cfg.optimizer.warmup_iters)

    # Save initial SAE
    save_sae(sae_cfg, sae, optimizer, it)

    # Training loop
    for epoch in range(sae_cfg.epochs):
        for sequences, _, _ in dataloader:
            if it > 7000 and sae_cfg.sae.sae_type == 'sparsemax_dist': 
                print("Training destabilizes after a certain point when the loss is too low, so we break")
                save_sae(sae_cfg, sae, optimizer, it)
                break

            # Update LR
            it, lr = update_cosine_warmup_lr(it, sae_cfg.optimizer, optimizer, total_steps)

            # Get activations from input data
            inputs = move_to_device([sequences], device)

            # Loss computation
            optimizer.zero_grad(set_to_none=True) # Set gradients to None
            with torch.amp.autocast(device_type='cuda' if 'cuda' in device else 'cpu', dtype=dt): # Mixed precision

                # Get activations
                activations = [] # Reset activations
                with torch.no_grad():
                    base_model(inputs)
                    activations = activations[0][inputs > 2] # Pull out relevant tokens
                    activations *= acts_scaling # Scale activations to 1/D variance

                # Get SAE output
                pred_activations, latent_code = sae(activations, return_hidden=True) 

                ## Regularization loss
                # ReLU: L1 
                if sae.sae_type == 'relu':
                    loss_reg = torch.norm(latent_code, p=1, dim=-1).mean()

                # Sparsemax: Distance based penalty
                elif sae.sae_type == 'sparsemax_dist':

                    # Optimized using kernel trick: ||x - a||^2 = ||x||^2 + ||a||^2 - 2*x^T*a
                    act_norm_sq = torch.sum(activations**2, dim=-1, keepdim=True)  # (batch_size, 1)
                    ae_norm_sq = torch.sum(sae.Ae**2, dim=-1, keepdim=True).T  # (1, width)
                    dot_product = torch.matmul(activations, sae.Ae.T)  # (batch_size, width)
                    dist_penalty_encoder = act_norm_sq + ae_norm_sq - 2 * dot_product
                    loss_reg = (dist_penalty_encoder * latent_code).sum(dim=-1).mean()

                # JumpReLU: L0 loss
                elif sae.sae_type == 'jumprelu':
                    bandwidth = 1e-3
                    loss_reg = torch.sum(
                        step_fn(latent_code, torch.exp(sae.logthreshold), bandwidth), 
                        dim=-1).mean()

                # TopK: L0 loss
                elif sae.sae_type == 'topk':
                    loss_reg = torch.tensor([0.0], device=device)

                # MP: L0 loss
                elif 'MP' in sae.sae_type:
                    loss_reg = torch.tensor([0.0], device=device)

                else:
                    raise ValueError('Invalid SAE type')

                ## MSE
                loss_mse = F.mse_loss(
                    pred_activations, 
                    activations, 
                    reduction='sum'
                ) / activations.shape[0]

                ## Total loss
                loss = loss_mse + gamma_reg * loss_reg

                ## Update model
                loss.backward() # Compute gradients
                if sae_cfg.optimizer.grad_clip > 0.0: # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(sae.parameters(), sae_cfg.optimizer.grad_clip)
                optimizer.step() # Update weights

                ## Logging
                with torch.no_grad():
                    # Log MSE / variance explained
                    av_norm_activations = activations.norm(dim=-1).pow(2).mean().item()
                    train_log['total'].append(loss.item()) # Total loss
                    train_log['mse'].append(loss_mse.item()) # MSE loss
                    train_log['nmse'].append(loss_mse.item() / av_norm_activations) # Normalized MSE loss
                    train_log['reg'].append(loss_reg.item()) # Reg loss
                    train_log['variance_explained'].append((1 - (pred_activations - activations).var() / activations.var()).item())
                    p_sparsity = (
                        (latent_code.abs() < 1e-5).sum()
                        / latent_code.numel() * 100
                    )  # Sparsity of the latent code
                    train_log["p_sparsity"].append(p_sparsity.item())  # Sparsity
                    train_log['lambda'].append(sae.lambda_val.data.item()) # Lambda
            
                    if dev_dataloader is not None and sae_cfg.data.run_validation:
                        with torch.no_grad():
                            torch.cuda.empty_cache()
                            for sequences, _, _ in dev_dataloader:
                                activations = [] # Reset activations
                                inputs = move_to_device([sequences], device)
                                base_model(inputs)
                                activations = activations[0][inputs > 2]
                                activations *= acts_scaling # Scale activations to 1/D variance
                                pred_activations, _ = sae(activations, return_hidden=True)
                                loss_mse = F.mse_loss(
                                    pred_activations, 
                                    activations, 
                                    reduction='sum'
                                ) / activations.shape[0]
                                av_norm_activations = activations.norm(dim=-1).pow(2).mean().item() 
                                val_log['mse'].append(loss_mse.item()) # MSE loss
                                val_log['nmse'].append(loss_mse.item() / av_norm_activations) # Normalized MSE loss
                                val_log['variance_explained'].append((1 - (pred_activations - activations).var() / activations.var()).item())
                                break # Only use one batch for validation

            # Log train metrics
            if it % sae_cfg.log.log_interval == 0:
                train_log = log_sae_train(it, sae_cfg.deploy, lr, train_log)
                # print(val_log)

            # Save model every few iterations
            if it % sae_cfg.log.save_interval == 0:
                save_sae(sae_cfg, sae, optimizer, it)
    
    # Save one last time
    save_sae(sae_cfg, sae, optimizer, it)

    # Remove hook
    act_hook.remove()


if __name__ == "__main__":
    main_hf()
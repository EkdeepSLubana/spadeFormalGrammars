import torch
import os
from sae import SAE
from types import SimpleNamespace

def load_sae(checkpoint_path, device='cpu'):
    """
    Load SAE model from checkpoint
    
    Args:
        checkpoint_path (str): Path to the checkpoint file (.pt)
        device (str): Device to load the model on ('cpu', 'cuda', etc.)
    
    Returns:
        tuple: (sae_model, optimizer, iteration, config)
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract components
    sae_state_dict = checkpoint['sae']
    optimizer_state_dict = checkpoint['optimizer']
    iteration = checkpoint['iter']
    config = checkpoint['config']
    
    # Reconstruct SAE model from config
    sae = SAE(
        dimin=config.dimin,
        width=config.width,
        sae_type=config.sae_type,
        kval_topk=getattr(config, 'kval_topk', None),
        normalize_decoder=getattr(config, 'normalize_decoder', False),
        lambda_init=getattr(config, 'lambda_init', None)
    )
    
    # Load state dict
    sae.load_state_dict(sae_state_dict)
    sae.to(device)
    
    # Create optimizer (you'll need to specify the optimizer type and params)
    # This assumes Adam optimizer - adjust as needed
    optimizer = torch.optim.Adam(sae.parameters())
    optimizer.load_state_dict(optimizer_state_dict)
    
    return sae, optimizer, iteration, config

def load_sae_inference_only(checkpoint_path, device='cpu', dimin=None):
    """
    Load SAE model for inference only (no optimizer)
    
    Args:
        checkpoint_path (str): Path to the checkpoint file (.pt)
        device (str): Device to load the model on ('cpu', 'cuda', etc.)
    
    Returns:
        tuple: (sae_model, iteration, config)
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract components
    sae_state_dict = checkpoint['sae']
    iteration = checkpoint['iter']
    config = checkpoint['config']
    
    # Reconstruct SAE model from config
    sae = SAE(
        dimin=dimin,
        width=int(dimin*config.sae.exp_factor), 
        sae_type=config.sae.sae_type,
        kval_topk=config.sae.kval_topk if config.sae.sae_type=='topk' else None,
        mp_kval=config.sae.mp_kval if config.sae.sae_type=='MP' else None,
        normalize_weights=True,
    )
    
    # Load state dict
    sae.load_state_dict(sae_state_dict)
    sae.to(device)
    sae.eval()  # Set to evaluation mode
    
    return sae, iteration, config

def load_latest_sae(results_dir, tag, device='cpu'):
    """
    Load the latest SAE checkpoint from a results directory
    
    Args:
        results_dir (str): Base results directory (e.g., 'sae_results/')
        tag (str): Model tag/name
        device (str): Device to load the model on
    
    Returns:
        tuple: (sae_model, optimizer, iteration, config)
    """
    fdir = os.path.join(results_dir, tag)
    checkpoint_path = os.path.join(fdir, 'latest_ckpt.pt')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    return load_sae(checkpoint_path, device)

def load_specific_iteration_sae(results_dir, tag, iteration, device='cpu'):
    """
    Load SAE checkpoint from a specific iteration
    
    Args:
        results_dir (str): Base results directory (e.g., 'sae_results/')
        tag (str): Model tag/name
        iteration (int): Specific iteration to load
        device (str): Device to load the model on
    
    Returns:
        tuple: (sae_model, optimizer, iteration, config)
    """
    fdir = os.path.join(results_dir, tag)
    checkpoint_path = os.path.join(fdir, f'ckpt_{iteration}.pt')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    return load_sae(checkpoint_path, device)

def list_available_checkpoints(results_dir, tag):
    """
    List all available checkpoints for a given tag
    
    Args:
        results_dir (str): Base results directory (e.g., 'sae_results/')
        tag (str): Model tag/name
    
    Returns:
        list: List of available checkpoint files
    """
    fdir = os.path.join(results_dir, tag)
    
    if not os.path.exists(fdir):
        return []
    
    checkpoints = []
    for file in os.listdir(fdir):
        if file.endswith('.pt'):
            checkpoints.append(file)
    
    return sorted(checkpoints)

# Example usage:
if __name__ == "__main__":
    # Load latest checkpoint
    sae, optimizer, iteration, config = load_latest_sae('sae_results/', 'my_model_tag')
    
    # Load for inference only
    sae_inference, iteration, config = load_sae_inference_only('sae_results/my_model_tag/latest_ckpt.pt')
    
    # Load specific iteration
    sae_specific, optimizer, iteration, config = load_specific_iteration_sae('sae_results/', 'my_model_tag', 1000)
    
    # List available checkpoints
    checkpoints = list_available_checkpoints('sae_results/', 'my_model_tag')
    print(f"Available checkpoints: {checkpoints}")
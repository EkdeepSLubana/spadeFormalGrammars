import sys
# sys.path.append("./dictionary_learning/")
# sys.path.append("./spadeFormalGrammars/")

from collections import namedtuple, defaultdict
# from dictionary_learning import AutoEncoder, JumpReluAutoEncoder
# from sae_test import SAE, step_fn
from utils.loading import load_sae_inference_only
from test import load_dataset, score_identification, get_activations

# RelaxedArchetypalAutoEncoder
# from dictionary_learning.dictionary import IdentityDict
from dataset import Submodule, Dataset
from typing import Literal
from nnsight import LanguageModel
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns
import torch as t
import math
import random
import numpy as np
from huggingface_hub import list_repo_files
from tqdm import tqdm
import os
import argparse


DICT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/dictionaries"

DictionaryStash = namedtuple("DictionaryStash", ["embed", "attns", "mlps", "resids"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, default="pythia70m")
    parser.add_argument("--dataset", "-d", type=str, default="../data/labeled_sentences.jsonl")
    parser.add_argument("--arch", "-a", type=str, default="relu")
    parser.add_argument("--id-metric", type=str, choices=["accuracy", "macrof1", "mcc"], default="mcc")
    parser.add_argument("--lamda", type=float, default=0.1)
    parser.add_argument("--randomize-sae", action="store_true")
    parser.add_argument("--identity-baseline", action="store_true")
    args = parser.parse_args()

    device = "cuda"
    dtype = t.float32 if args.model_name == "pythia70m" else t.bfloat16
   
    if args.model_name == "pythia70m":
        model = LanguageModel("EleutherAI/pythia-70m-deduped", device_map=device, dispatch=True, torch_dtype=dtype)
        model_name = "pythia-70m-deduped"
    elif args.model_name == "gemma2":
        model = LanguageModel("google/gemma-2-2b", device_map=device, dispatch=True, attn_implementation="eager", torch_dtype=dtype)
        model_name = "gemma-2-2b"
    else:
        raise NotImplementedError()
    
    mid_layer = model.config.num_hidden_layers // 2

    neurons = args.identity_baseline
    
    submodule = Submodule(
        name = f"resid_{mid_layer}",
        # submodule=model.gpt_neox.layers[mid_layer],
        submodule = model.model.layers[mid_layer] if args.model_name == "gemma2" else model.gpt_neox.layers[mid_layer],
        is_tuple=True,
    )
    # submodules, dictionaries = load_saes_and_submodules(model, dtype=dtype, device=device, thru_layer=mid_layer, neurons=neurons)
    # submodule = get_mid_layer(submodules, mid_layer)
    # Load for inference only

    cc_scores = defaultdict(list)
    cc_std = defaultdict(list)

    for corr in tqdm([0.0, 0.1, 0.2, 0.5, 0.9, 1.0], total=6, desc="Correlations"):
    # for corr in tqdm([0.0]):
        corr_seeds = defaultdict(list)
        for seed in tqdm([42, 2718, 5772], total=3, desc="Seeds", leave=False):
        # for seed in (42,2718,5772):
        # for seed in (42,):
            if corr == 0.0:
                sae_path = f"sae_results/{seed}/{model_name}/{args.arch}_uniform/latest_ckpt.pt"
            else:
                sae_path = f"sae_results/{seed}/{model_name}/{args.arch}_corr_vp-tpa_{corr}/latest_ckpt.pt"
            
            if not os.path.exists(sae_path):
                continue
            dictionary, _, _ = load_sae_inference_only(sae_path, dimin=model.config.hidden_size)
            dictionary.to("cuda:0").to(dtype)
            if args.randomize_sae:
                dictionary.encoder.reset_parameters()
                dictionary.decoder.reset_parameters()
                # dictionaries[submodule].W_enc.data.normal_(mean=0, std=0.1)
                # dictionaries[submodule].W_dec.data.normal_(mean=0, std=0.1)
                # dictionaries[submodule].b_enc.data.normal_(mean=0, std=0.1)
                # dictionaries[submodule].b_dec.data.normal_(mean=0, std=0.1)
            # dictionary = dictionaries[submodule]


            dataset = load_dataset(args.dataset)
            examples = dataset.examples
            labels = dataset.labels_binary
            num_examples = len(examples)
            batch_size = 8

            n_batches = math.ceil(num_examples / batch_size)
            batches = [
                examples[batch * batch_size : (batch + 1) * batch_size]
                for batch in range(n_batches)
            ]
            labels_batched = [
                {l: labels[l][batch * batch_size : (batch + 1) * batch_size]
                for l in labels}
                for batch in range(n_batches)
            ]

            with t.no_grad(), model.trace("test"):
                x = submodule.get_activation()
                x_saved = x.save()
            if "sparsemax_dist" in sae_path or "MP" in sae_path:
                x_saved = x_saved.value.flatten(start_dim=0, end_dim=1)
                x_hat, f = dictionary(x_saved, return_hidden=True)
            else:
                x_saved = x_saved.value
                x_hat, f = dictionary(x_saved[:, 0, :], return_hidden=True)
            num_hidden = f.detach().shape[-1]
            num_hidden_xhat = x_hat.detach().shape[-1]
            # f_saved = f.save()
            # num_hidden = f_saved.value.detach().shape[-1]
            acts = t.zeros((num_examples, num_hidden))
            acts_xhat = t.zeros((num_examples, num_hidden_xhat))

            for idx, batch in tqdm(enumerate(batches), desc="Caching activations", total=len(batches)):
                f, xhat = get_activations(model, submodule, dictionary, batch, sae_path)
                f = f.sum(dim=1)
                xhat = xhat.sum(dim=1)
                len_batch = len(batch)
                start_idx = idx * batch_size
                acts[start_idx : start_idx + len_batch] = f
                acts_xhat[start_idx : start_idx + len_batch] = xhat

            scores, top_features = score_identification(acts, dataset.labels_binary, 
                                                        lamda=args.lamda, metric=args.id_metric)
            top_scores = scores.max(dim=0).values
            prefix_groups = {}
            for i, label in enumerate(list(top_features.keys())):
                prefix = label.split('-')[0] if '-' in label else label
                if prefix not in prefix_groups:
                    prefix_groups[prefix] = []
                prefix_groups[prefix].append(top_scores[i].item())
            
            # Compute average for each prefix group
            prefix_averages = {}
            for prefix, group_scores in prefix_groups.items():
                prefix_averages[prefix] = sum(group_scores) / len(group_scores)
            
            # Compute macro-average across prefix groups
            mcc = sum(prefix_averages.values()) / len(prefix_averages)

            for i, label in enumerate(list(top_features.keys())):
                if label not in ("tense-past", "voice-passive"):
                    continue
                # print(f"{label}: {top_scores[i]} ({top_features[label]})")
                corr_seeds[label].append(top_scores[i].item())
            corr_seeds["overall"].append(mcc)
            # for i, label in enumerate(list(top_features.keys())):
            #     print(f"{label}: {top_scores[i]} ({top_features[label]})")
            # print(f"MCC: {mcc:.3f}") 
            # cc_scores["overall"].append(round(mcc, 2))
        
        for i, label in enumerate(corr_seeds.keys()):
            if len(corr_seeds[label]) > 1:
                cc_scores[label].append(np.mean(corr_seeds[label]))
                cc_std[label].append(np.std(corr_seeds[label]))
            else:
                cc_scores[label].append(corr_seeds[label][0])
                cc_std[label].append(None)

    
    print("Tense-past:")
    print("\t" + ", ".join(f"{x:.3f}" for x in cc_scores["tense-past"]))
    print("\t" + ", ".join(f"{x:.3f}" for x in cc_std["tense-past"]))
    print()

    print("Voice-passive:")
    print("\t" + ", ".join(f"{x:.3f}" for x in cc_scores["voice-passive"]))
    print("\t" + ", ".join(f"{x:.3f}" for x in cc_std["voice-passive"]))
    print()

    print("MCC:")
    print("\t" + ", ".join(f"{x:.3f}" for x in cc_scores["overall"]))
    print("\t" + ", ".join(f"{x:.3f}" for x in cc_std["overall"]))
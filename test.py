import sys
# sys.path.append("./dictionary_learning/")
# sys.path.append("./spadeFormalGrammars/")
sys.path.append("../dictionary_learning/")

from collections import namedtuple
from dictionary_learning import AutoEncoder, JumpReluAutoEncoder
from sae import SAE, step_fn
from utils.loading import load_sae_inference_only

# RelaxedArchetypalAutoEncoder
from dictionary_learning.dictionary import IdentityDict
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
sns.set()
sns.set_style("whitegrid")

DICT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/dictionaries"

DictionaryStash = namedtuple("DictionaryStash", ["embed", "attns", "mlps", "resids"])


def _load_pythia_saes_and_submodules(
    model,
    thru_layer: int | None = None,
    separate_by_type: bool = False,
    include_embed: bool = True,
    neurons: bool = False,
    null: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device = t.device("cpu"),
):
    assert len(model.gpt_neox.layers) == 6, "Not the expected number of layers for pythia-70m-deduped"
    if thru_layer is None:
        thru_layer = len(model.gpt_neox.layers)

    attns = []
    mlps = []
    resids = []
    dictionaries = {}
    if include_embed:
        embed = Submodule(
            name = "embed",
            submodule=model.gpt_neox.embed_in,
        )
        if not neurons and not null:
            dictionaries[embed] = AutoEncoder.from_pretrained(
                f"{DICT_DIR}/pythia-70m-deduped/embed/10_32768/ae.pt",
                dtype=dtype,
                device=device,
            )
        elif neurons:
            dictionaries[embed] = IdentityDict(512)
        else:
            dictionaries[embed] = None
    else:
        embed = None
    for i, layer in enumerate(model.gpt_neox.layers[:thru_layer+1]):
        attns.append(
            attn := Submodule(
                name = f"attn_{i}",
                submodule=layer.attention,
                is_tuple=True,
            )
        )
        mlps.append(
            mlp := Submodule(
                name = f"mlp_{i}",
                submodule=layer.mlp,
            )
        )
        resids.append(
            resid := Submodule(
                name = f"resid_{i}",
                submodule=layer,
                is_tuple=True,
            )
        )
        if not neurons and not null:
            dictionaries[attn] = AutoEncoder.from_pretrained(
                f"{DICT_DIR}/pythia-70m-deduped/attn_out_layer{i}/10_32768/ae.pt",
                dtype=dtype,
                device=device,
            )
            dictionaries[mlp] = AutoEncoder.from_pretrained(
                f"{DICT_DIR}/pythia-70m-deduped/mlp_out_layer{i}/10_32768/ae.pt",
                dtype=dtype,
                device=device,
            )
            dictionaries[resid] = AutoEncoder.from_pretrained(
                f"{DICT_DIR}/pythia-70m-deduped/resid_out_layer{i}/10_32768/ae.pt",
                dtype=dtype,
                device=device,
            )
        elif neurons:
            dictionaries[attn] = IdentityDict(512)
            dictionaries[mlp] = IdentityDict(512)
            dictionaries[resid] = IdentityDict(512)
        else:
            dictionaries[attn] = None
            dictionaries[mlp] = None
            dictionaries[resid] = None

    if separate_by_type:
        return DictionaryStash(embed, attns, mlps, resids), dictionaries
    else:
        submodules = (
            [embed] if include_embed else []
         ) + [
            x for layer_dictionaries in zip(attns, mlps, resids) for x in layer_dictionaries
        ]
        return submodules, dictionaries


def load_gemma_sae(
    submod_type: Literal["embed", "attn", "mlp", "resid"],
    layer: int,
    width: Literal["16k", "65k"] = "16k",
    neurons: bool = False,
    null: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device = t.device("cpu"),
):
    if neurons:
        if submod_type != "attn":
            return IdentityDict(2304)
        else:
            return IdentityDict(2048)

    repo_id = "google/gemma-scope-2b-pt-" + (
        "res" if submod_type in ["embed", "resid"] else
        "att" if submod_type == "attn" else
        "mlp"
    )
    if submod_type != "embed":
        directory_path = f"layer_{layer}/width_{width}"
    else:
        directory_path = "embedding/width_4k"

    files_with_l0s = [
        (f, int(f.split("_")[-1].split("/")[0]))
        for f in list_repo_files(repo_id, repo_type="model", revision="main")
        if f.startswith(directory_path) and f.endswith("params.npz")
    ]
    optimal_file = min(files_with_l0s, key=lambda x: abs(x[1] - 100))[0]
    optimal_file = optimal_file.split("/params.npz")[0]
    return JumpReluAutoEncoder.from_pretrained(
        load_from_sae_lens=True,
        release=repo_id.split("google/")[-1],
        sae_id=optimal_file,
        dtype=dtype,
        device=device,
    )


def _load_gemma_saes_and_submodules(
    model,
    thru_layer: int | None = None,
    separate_by_type: bool = False,
    include_embed: bool = True,
    neurons: bool = False,
    null: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device = t.device("cpu"),
):
    assert len(model.model.layers) == 26, "Not the expected number of layers for Gemma-2-2B"
    if thru_layer is None:
        thru_layer = len(model.model.layers)
    
    attns = []
    mlps = []
    resids = []
    dictionaries = {}
    if include_embed:
        embed = Submodule(
                name = "embed",
                submodule=model.model.embed_tokens,
        )
        if not neurons and not null:
            dictionaries[embed] = load_gemma_sae("embed", 0, neurons=neurons, dtype=dtype, device=device)
        elif neurons:
            dictionaries[embed] = IdentityDict(2304)
        else:
            dictionaries[embed] = None
    else:
        embed = None
    for i, layer in tqdm(enumerate(model.model.layers[:thru_layer+1]), total=thru_layer+1, desc="Loading Gemma SAEs"):
        attns.append(
            attn := Submodule(
                name=f"attn_{i}",
                submodule=layer.self_attn.o_proj,
                use_input=True
            )
        )
        mlps.append(
            mlp := Submodule(
                name=f"mlp_{i}",
                submodule=layer.post_feedforward_layernorm,
            )
        )
        resids.append(
            resid := Submodule(
                name=f"resid_{i}",
                submodule=layer,
                is_tuple=True,
            )
        )
        if not neurons and not null:
            dictionaries[resid] = load_gemma_sae("resid", i, neurons=neurons, dtype=dtype, device=device)
            # dictionaries[attn] = load_gemma_sae("attn", i, neurons=neurons, dtype=dtype, device=device)
            # dictionaries[mlp] = load_gemma_sae("mlp", i, neurons=neurons, dtype=dtype, device=device)
            dictionaries[attn] = None
            dictionaries[mlp] = None
        elif neurons:
            dictionaries[mlp] = IdentityDict(2304)
            dictionaries[attn] = IdentityDict(2304)
            dictionaries[resid] = IdentityDict(2304)
        else:
            dictionaries[mlp] = None
            dictionaries[attn] = None
            dictionaries[resid] = None

    if separate_by_type:
        return DictionaryStash(embed, attns, mlps, resids), dictionaries
    else:
        submodules = (
            [embed] if include_embed else []
        )+ [
            x for layer_dictionaries in zip(attns, mlps, resids) for x in layer_dictionaries
        ]
        return submodules, dictionaries


def load_saes_and_submodules(
    model,
    thru_layer: int | None = None,
    separate_by_type: bool = False,
    include_embed: bool = True,
    neurons: bool = False,
    null: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device = t.device("cpu"),
):
    model_name = model.config._name_or_path

    if model_name == "EleutherAI/pythia-70m-deduped":
        return _load_pythia_saes_and_submodules(model, thru_layer=thru_layer, separate_by_type=separate_by_type, include_embed=include_embed, neurons=neurons, null=null, dtype=dtype, device=device)
    elif model_name == "google/gemma-2-2b":
        return _load_gemma_saes_and_submodules(model, thru_layer=thru_layer, separate_by_type=separate_by_type, include_embed=include_embed, neurons=neurons, null=null, dtype=dtype, device=device)
    else:
        raise ValueError(f"Model {model_name} not supported")


def get_mid_layer(submodules, layer):
    target = f"resid_{layer}"
    for submodule in submodules:
        if submodule.name == target:
            return submodule


def get_activations(model, submodule, dictionary, batch, sae_path, identity=False):
    with t.no_grad(), model.trace(batch):
        x = submodule.get_activation()
        x_saved = x.save()
    x_saved = x_saved.value
    if "sparsemax_dist" in sae_path or "MP" in sae_path:
        x_saved = x_saved[0]
    if not identity:
        # x_hat, f = dictionary(x_saved, return_hidden=True)
        x_hat, f = dictionary(x_saved, output_features=True)
    else:
        f = dictionary(x_saved)
        x_hat = f
    # f_saved = f.save()
    return (f.detach(), x_hat.detach())


def load_dataset(dataset):
    return Dataset(dataset)


def score_identification(acts, labels, lamda=0.1, metric="accuracy"):
    scores = {}
    top_features = {}
    labels = {k: v for k, v in labels.items() if k not in ("formality-high", "formality-neutral", "reading-level-low", "reading-level-high")}
    label_matrix = t.stack([t.Tensor(labels[l]) for l in labels], dim=0)    # N x L

    for label_name in labels:
        if metric == "mcc":
            label_vec = t.Tensor(labels[label_name])   # N
        else:
            label_vec = t.tensor(labels[label_name])
        feature_labels = acts.T > lamda     # F x N
        if metric == "accuracy":
            matches = (feature_labels == label_vec)
            accuracies = matches.sum(dim=1) / label_vec.shape[-1]
            accuracy = accuracies.max()
            top_features[label_name] = accuracies.argmax()
            scores[label_name] = accuracy
        elif metric == "macrof1":
            # Calculate true positives, false positives, false negatives for each feature
            true_positives = (feature_labels & label_vec).sum(dim=1).float()  # F
            false_positives = (feature_labels & ~label_vec).sum(dim=1).float()  # F
            false_negatives = (~feature_labels & label_vec).sum(dim=1).float()  # F
            
            # Calculate precision and recall
            precision = true_positives / (true_positives + false_positives + 1e-10)  # F
            recall = true_positives / (true_positives + false_negatives + 1e-10)  # F
            
            # Calculate F1 scores
            f1_scores = 2 * precision * recall / (precision + recall + 1e-10)  # F
            
            # Find the feature with the max F1 score
            top_feature = f1_scores.argmax()
            max_f1 = f1_scores[top_feature]
            
            top_features[label_name] = top_feature
            scores[label_name] = max_f1
        elif metric == "mcc":
            acts_centered = acts - acts.mean(dim=0, keepdim=True)
            acts_std = acts_centered.norm(dim=0, keepdim=True)
            label_matrix_centered = label_matrix.T - label_matrix.T.mean(dim=0, keepdim=True)
            label_matrix_std = label_matrix_centered.norm(dim=0, keepdim=True)
            # Correct correlation computation
            numerator = acts_centered.T @ label_matrix_centered  # F × L
            denominator = acts_std.T * label_matrix_std  # F × L (broadcasting)

            mask = denominator != 0     # prevent NaNs
            corr_matrix = t.zeros_like(numerator)
            corr_matrix[mask] = numerator[mask] / denominator[mask]

            # Get indices of maximum correlations for each label
            top_feature_indices = corr_matrix.argmax(dim=0)  # Returns indices, shape: (L,)
            top_features = {label_name: top_feature_indices[i].item() for i, label_name in enumerate(list(labels))}

            return corr_matrix, top_features
        else:
            raise ValueError(f"Unrecognized metric: {metric}")

    return scores, top_features


def score_sensitivity(acts, labels, feature_idx, lamda=0.1, target_label="domain-science"):
    # First, find sentences where all labels are the same except the target_label
    print(acts.sum())
    prefix = target_label.split("-")[0]
    label_present = t.Tensor(labels[target_label]).nonzero().squeeze().tolist()
    not_label_present = set(list(range(acts.shape[0]))).difference(label_present)
    pair_indices = []
    for idx1 in label_present:
        label_vec1 = t.Tensor([labels[l][idx1] for l in labels.keys() if not l.startswith(prefix)])
        for idx2 in not_label_present:
            label_vec2 = t.Tensor([labels[l][idx2] for l in labels.keys() if not l.startswith(prefix)])
            if label_vec1.equal(label_vec2):
                pair_indices.append((idx1, idx2))

    sensitive = 0
    total = len(pair_indices)
    for pair in pair_indices:
        idx1, idx2 = pair
        if (acts[idx1][feature_idx] > lamda and acts[idx2][feature_idx] < lamda) or \
            (acts[idx1][feature_idx] < lamda and acts[idx2][feature_idx] > lamda):
            sensitive += 1
    return sensitive / total, total


def reconstruct_means(acts, labels, target_label1="domain-science", target_label2="sentiment-positive"):
    label1_present = set(t.Tensor(labels[target_label1]).nonzero().squeeze().tolist())
    label2_present = set(t.Tensor(labels[target_label2]).nonzero().squeeze().tolist())
    acts1 = np.zeros_like(acts[0])
    acts2 = np.zeros_like(acts[0])
    acts12 = np.zeros_like(acts[0])
    for idx1 in label1_present:     # Find all target_label1 sentences, take mean (v_1)
        if idx1 in label2_present:
            continue
        acts1 = np.add(acts1, acts[idx1])
    acts1 /= len(label1_present)
    for idx2 in label2_present:     # Find all target_label2 sentences, take mean (v_2)
        if idx2 in label1_present:
            continue
        acts2 = np.add(acts2, acts[idx2])
    acts2 /= len(label2_present)

    label12_present = label1_present.intersection(label2_present)
    for idx12 in label12_present:   # Find all target_label1 AND target_label2 sentences, take mean (v_12)
        acts12 = np.add(acts12, acts[idx12])
    acts12 /= len(label12_present)

    proj_acts1 = (np.dot(acts12, acts1) / np.dot(acts1, acts1)) * acts1
    proj_acts2 = (np.dot(acts12, acts2) / np.dot(acts2, acts2)) * acts2
    print(proj_acts1)
    print(proj_acts2)
    acts12_reconstructed = proj_acts1 + proj_acts2

    # Analysis
    resid_vec = acts12 - acts12_reconstructed

    mse = np.square(resid_vec).mean()
    acts12_norm = np.linalg.norm(acts12)
    relative_error = np.linalg.norm(resid_vec) / acts12_norm if acts12_norm > 0 else float('inf')

    ss_tot = np.square(acts12 - acts12.mean()).sum()
    ss_res = np.square(resid_vec).sum()
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    proj_v1_magnitude = np.linalg.norm(proj_acts1)
    proj_v2_magnitude = np.linalg.norm(proj_acts2)
    original_magnitude = np.linalg.norm(acts12)
    reconstructed_magnitude = np.linalg.norm(acts12_reconstructed)
    fraction_in_v1_direction = proj_v1_magnitude / original_magnitude
    fraction_in_v2_direction = proj_v2_magnitude / original_magnitude

    cos_sim = np.dot(acts1, acts2) / (np.linalg.norm(acts1) * np.linalg.norm(acts2))

    return {"proj_v1": proj_acts1,
            "proj_v2": proj_acts2,
            "reconstruction": acts12_reconstructed,
            "MSE": mse,
            "relative_error": relative_error,
            "r_squared": r_squared,
            "cos_sim": cos_sim,
            "fraction_in_v1": fraction_in_v1_direction,
            "fraction_in_v2": fraction_in_v2_direction
            }


def plot_distributions(activations, top_features, labels, bins=30, model_name="pythia70m", lamda=0.1):
    random.seed(12)
    for label_name in labels:
        label_vec = t.Tensor(labels[label_name])
        top_feature = top_features[label_name]
        random_feature = random.randint(0, activations.shape[-1])

        class_0_acts = activations.T[top_feature][label_vec == 0]
        class_1_acts = activations.T[top_feature][label_vec == 1]
        var_0, mean_0 = t.var_mean(class_0_acts, dim=-1)
        var_1, mean_1 = t.var_mean(class_1_acts, dim=-1)

        random_class_0_acts = activations.T[random_feature][label_vec == 0]
        random_class_1_acts = activations.T[random_feature][label_vec == 1]
        random_var_0, random_mean_0 = t.var_mean(random_class_0_acts, dim=-1)
        random_var_1, random_mean_1 = t.var_mean(random_class_1_acts, dim=-1)
        
        print(f"{label_name}: {mean_0} ({var_0}) | {mean_1} ({var_1})")
        print(f"\t- Random: {random_mean_0} ({random_var_0}) | {random_mean_1} ({random_var_1})")

        class_0_kde = stats.gaussian_kde(class_0_acts)
        class_1_kde = stats.gaussian_kde(class_1_acts)
        min_act = min(min(class_0_acts), min(class_1_acts))
        max_act = max(max(class_0_acts), max(class_1_acts))
        xx = np.linspace(min_act, max_act, 1000)

        fig, ax1 = plt.subplots(figsize=(10,6))
        
        ax1.hist(class_0_acts, bins=bins, alpha=0.5, color='blue', label='False')
        ax1.hist(class_1_acts, bins=bins, alpha=0.5, color='red', label='True')
        ax1.set_ylabel("Frequency")
        ax2 = ax1.twinx()
        ax2.plot(xx, class_0_kde(xx), color='blue')
        ax2.plot(xx, class_1_kde(xx), color='red')
        ax2.set_ylabel("Density")
        ax2.grid(False)

        plt.title(label_name)
        ax1.legend()

        out_dir = f"results/activation_plots/{model_name}/{lamda}/"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f"{label_name}.pdf"), format="pdf", bbox_inches="tight")
        plt.cla()
        plt.clf()
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="pythia70m")
    parser.add_argument("--dataset", "-d", type=str, default="../data/labeled_sentences.jsonl")
    parser.add_argument("--sae", "-s", type=str, default="sae_results/relu_uniform/latest_ckpt.pt")
    parser.add_argument("--id-metric", type=str, choices=["accuracy", "macrof1", "mcc"], default="mcc")
    parser.add_argument("--lamda", type=float, default=0.1)
    parser.add_argument("--randomize-sae", action="store_true")
    parser.add_argument("--identity-baseline", action="store_true")
    args = parser.parse_args()

    device = "cuda"
    dtype = t.float32 if args.model_name == "pythia70m" else t.bfloat16
   
    if args.model_name == "pythia70m":
        model = LanguageModel("EleutherAI/pythia-70m-deduped", device_map=device, dispatch=True, torch_dtype=dtype)
    elif args.model_name == "gemma2":
        model = LanguageModel("google/gemma-2-2b", device_map=device, dispatch=True, attn_implementation="eager", torch_dtype=dtype)
    else:
        raise NotImplementedError()
    
    mid_layer = model.config.num_hidden_layers // 2

    neurons = args.identity_baseline
    
    submodule = Submodule(
        name = f"resid_{mid_layer}",
        # submodule=model.gpt_neox.layers[mid_layer],
        submodule = model.model.layers[mid_layer],
        is_tuple=True,
    )
    submodules, dictionaries = load_saes_and_submodules(model, dtype=dtype, device=device, thru_layer=mid_layer, neurons=neurons)
    submodule = get_mid_layer(submodules, mid_layer)

    # Load for inference only
    # dictionary, _, _ = load_sae_inference_only(args.sae, dimin=model.config.hidden_size)
    # dictionary.to("cuda:0")
    # dictionary.requires_grad_(False)

    # Archetypal SAEs
    # dictionaries[submodule] = AutoEncoder.from_pretrained(
    #         "/home/aaron/fromgit/identifiable_language/archetypal/dictionary_learning/weights/pythia-70m/reg/trainer_0/ae.pt",
    #         device=device
    # )
    if args.randomize_sae:
        dictionary._reset_parameters()
        # dictionaries[submodule].W_enc.data.normal_(mean=0, std=0.1)
        # dictionaries[submodule].W_dec.data.normal_(mean=0, std=0.1)
        # dictionaries[submodule].b_enc.data.normal_(mean=0, std=0.1)
        # dictionaries[submodule].b_dec.data.normal_(mean=0, std=0.1)
    dictionary = dictionaries[submodule]


    dataset = load_dataset(args.dataset)
    examples = dataset.examples
    labels = dataset.labels_binary
    num_examples = len(examples)
    batch_size = 1

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

    with t.no_grad(), model.trace("t"):
        x = submodule.get_activation()
        x_saved = x.save()
    if not args.identity_baseline:
        # x_hat, f = dictionary(x_saved.value[:, 0, :], return_hidden=True)
        x_hat, f = dictionary(x_saved.value[:, 0, :], output_features=True)
    else:
        f = dictionary(x_saved.value[:, 0, :])
        x_hat = f

    num_hidden = f.detach().shape[-1]
    num_hidden_xhat = x_hat.detach().shape[-1]
    # f_saved = f.save()
    # num_hidden = f_saved.value.detach().shape[-1]
    acts = t.zeros((num_examples, num_hidden))
    acts_xhat = t.zeros((num_examples, num_hidden_xhat))

    for idx, batch in tqdm(enumerate(batches), desc="Caching activations", total=len(batches)):
        f, xhat = get_activations(model, submodule, dictionary, batch, args.sae, identity=args.identity_baseline)
        if "sparsemax_dist" in args.sae or "MP" in args.sae:
            f = f.unsqueeze(0)
            xhat = xhat.unsqueeze(0)
        f = f.sum(dim=1)
        xhat = xhat.sum(dim=1)
        len_batch = len(batch)
        start_idx = idx * batch_size
        acts[start_idx : start_idx + len_batch] = f
        acts_xhat[start_idx : start_idx + len_batch] = xhat
    
    if args.randomize_sae:
        acts_input = t.rand(acts.shape)
        acts = t.rand(acts.shape)
        acts_hat = t.rand(acts_xhat.shape)


    scores, top_features = score_identification(acts, dataset.labels_binary, 
                                                lamda=args.lamda, metric=args.id_metric)
    if args.id_metric == "mcc":
        top_scores = scores.max(dim=0).values
        # print(top_scores.shape)
        # print(scores, top_features)
        mcc = top_scores.mean().item()
        for i, label in enumerate(list(top_features.keys())):
            if label not in ("domain-science", "sentiment-positive"):
                continue
            print(f"{label}: {top_scores[i]} ({top_features[label]})")
        # for i, label in enumerate(list(top_features.keys())):
        #     print(f"{label}: {top_scores[i]} ({top_features[label]})")
        print(f"MCC: {mcc:.3f}")
    else:
        print(scores)
        print(sum(list(scores.values())))
        print(sum(list(scores.values())) / len(list(scores.keys())))
        print()
        print(top_features)

    # print()
    # rec_metrics = reconstruct_means(acts, dataset.labels_binary)
    # print("Decoded activation reconstruction:")
    # print(rec_metrics)
    # print("\tRelative error:", rec_metrics["relative_error"])
    # print("\tR^2:", rec_metrics["r_squared"])

    # sensitivities = {}
    # for label in tqdm(dataset.labels_binary, total=len(list(dataset.labels_binary.keys())), desc="Sensitivity of label"):
    #     sensitivities[label], N = score_sensitivity(acts, dataset.labels_binary, top_features[label],
    #                                              lamda=args.lamda, target_label=label)
    # print("Sensitivities: ", sensitivities)
    # print("Sensitivity mean: ", sum(sensitivities.values()) / len(sensitivities.values()))

    # Print conditional distributions
    # plot_distributions(acts, top_features, dataset.labels_binary)
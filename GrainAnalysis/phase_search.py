import torch
import torch.nn as nn
import torch.nn.functional as F
import copy, os
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse as parser
from utils import FSNeuronDecoupled, FSNeuronDecoupledSoftMax
import torch.optim as optim
import argparse, sys
from typing import List

parser = argparse.ArgumentParser(description='Phase Search Training')
parser.add_argument('--epoch_outer', type=int, default=20, help='Number of epochs to train')
parser.add_argument('--epoch_inner', type=int, default=50, help='Number of epochs to train')
parser.add_argument('--T', type=int, default=8)
parser.add_argument('--num_grains', type=int, default=3)
parser.add_argument('--log_interval', type=int, default=1)
parser.add_argument('--beta', type=float, default=10.)
parser.add_argument('--step_size', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--start', type=int, default=-1)
parser.add_argument('--end', type=int, default=32)
parser.add_argument('--model_name', type=str, default='Llama-2-7B-hf')
parser.add_argument('--save_name', type=str, default='search_arch-tau1.0.pth',
                    help='Output filename for the final merged/search result')
parser.add_argument('--merge_only', action='store_true', default=False,
                    help='Only merge existing single-layer/multi-layer files, skip training')
parser.add_argument('--merge_source', type=str, default='single',
                    choices=['single', 'multi', 'both'],
                    help='Which layer dir to merge from: single, multi, or both')

args = parser.parse_args()


# ============================================================
# Helper: save / merge layer results
# ============================================================

def get_save_dir(model_name, T, num_grains):
    return f'./arch_dir/{model_name}-T-{T}-grains-{num_grains}'


def save_single_layer(layer_idx, layer_dict, save_dir):
    """Save a single layer's result dict to single-layer/layer_{idx}.pth"""
    single_dir = os.path.join(save_dir, 'single-layer')
    os.makedirs(single_dir, exist_ok=True)
    fpath = os.path.join(single_dir, f'layer_{layer_idx}.pth')
    torch.save(layer_dict, fpath)
    print(f"[save] single-layer: {fpath}  ({len(layer_dict)} keys)")


def save_multi_layer_batch(start, end, full_dict, save_dir):
    """Save the full batch result for a start-end range to multi-layers/"""
    multi_dir = os.path.join(save_dir, 'multi-layers')
    os.makedirs(multi_dir, exist_ok=True)
    fname = f'layers_{start}_{end - 1}.pth'
    fpath = os.path.join(multi_dir, fname)
    torch.save(full_dict, fpath)
    print(f"[save] multi-layers: {fpath}  ({len(full_dict)} keys)")


def merge_layer_files(layer_dir):
    """Load and merge all .pth files from a directory into a single dict."""
    if not os.path.isdir(layer_dir):
        print(f"[merge] directory not found: {layer_dir}")
        return {}
    merged = {}
    fnames = sorted([f for f in os.listdir(layer_dir) if f.endswith('.pth')])
    if not fnames:
        print(f"[merge] no .pth files in {layer_dir}")
        return {}
    for fname in fnames:
        fpath = os.path.join(layer_dir, fname)
        d = torch.load(fpath, map_location='cpu')
        merged.update(d)
        print(f"[merge]   + {fname}  ({len(d)} keys)")
    return merged


def merge_single_layers(save_dir):
    """Merge all single-layer/layer_*.pth into one complete dict."""
    single_dir = os.path.join(save_dir, 'single-layer')
    return merge_layer_files(single_dir)


def merge_multi_layers(save_dir):
    """Merge all multi-layers/layers_*.pth into one complete dict."""
    multi_dir = os.path.join(save_dir, 'multi-layers')
    return merge_layer_files(multi_dir)


def normalize_arch_dict(merged, num_grains):
    """
    Ensure every key's value matches what retrain_decoupled.py expects:
      - model.norm.output           → bare list (genotype)
      - model.layers.{i}.{bk}       → (num_grains, genotype) tuple
    """
    normalized = {}
    for k, v in merged.items():
        if k == "model.norm.output":
            # expect bare list; if saved as tuple, unwrap
            if isinstance(v, tuple):
                normalized[k] = v[-1]  # take the last element (genotype list)
            else:
                normalized[k] = v
        else:
            # layer keys expect (num_grains, list)
            if isinstance(v, tuple):
                normalized[k] = v
            else:
                # bare list → wrap with num_grains
                normalized[k] = (num_grains, v)
    return normalized


def merge_all_results(save_dir, merge_source='single', num_grains=3):
    """
    Top-level merge entry point.
    merge_source: 'single' | 'multi' | 'both'
    Returns a normalized dict ready for retrain_decoupled.py.
    """
    merged = {}
    if merge_source in ('single', 'both'):
        print(f"[merge] scanning single-layer/ ...")
        merged.update(merge_single_layers(save_dir))
    if merge_source in ('multi', 'both'):
        print(f"[merge] scanning multi-layers/ ...")
        merged.update(merge_multi_layers(save_dir))

    merged = normalize_arch_dict(merged, num_grains)
    return merged


# ============================================================
# PhaseSearch model (unchanged)
# ============================================================

class PhaseSearch(nn.Module):
    def __init__(self, X, T: int, num_grains: int = 3, beta: float = 1.0,
                 epoch_inner: int = 1000, device: str = 'cuda:0',
                 tau=None, mode=None):
        super().__init__()
        self.T = T
        self.num_grains = num_grains
        self.genotype = self.__generate_genotype(T, num_grains)
        self.num_edges = len(self.genotype)
        self.beta = beta
        self.epoch = epoch_inner
        self.device = device
        self.fs_neurons = nn.ModuleList()
        self.tau = tau
        self.mode = mode
        self.architecture = self.__get_architecture(self.genotype)
        if self.mode is None:
            for i in range(self.num_edges):
                fs_neuron = FSNeuronDecoupled(T=self.T, num_grains=self.num_grains,
                                              beta=self.beta, genotype=self.genotype[i],
                                              tau=self.tau)
                fs_neuron.to(device)
                self.fs_neurons.append(fs_neuron)
        elif self.mode == "SoftMax":
            for i in range(self.num_edges):
                fs_neuron = FSNeuronDecoupledSoftMax(T=self.T, num_grains=self.num_grains,
                                                     beta=self.beta, genotype=self.genotype[i],
                                                     tau=self.tau)
                fs_neuron.to(device)
                self.fs_neurons.append(fs_neuron)

    def get_params(self):
        if self.mode == "SoftMax":
            h_params, theta_params, d_params, v0_params, arch_params = [], [], [], [], list(self.architecture)
            for fs_neuron in self.fs_neurons:
                h_params.append(fs_neuron.h)
                theta_params.append(fs_neuron.theta)
                d_params.append(fs_neuron.d)
                v0_params.append(fs_neuron.v0)
            return h_params, theta_params, d_params, v0_params, arch_params
        else:
            h_params, theta_params, d_params, arch_params = [], [], [], list(self.architecture)
            for fs_neuron in self.fs_neurons:
                h_params.append(fs_neuron.h)
                theta_params.append(fs_neuron.theta)
                d_params.append(fs_neuron.d)
            return h_params, theta_params, d_params, arch_params

    def get_arch_params(self):
        return list(self.architecture)

    def freeze_params(self, params_to_freeze):
        for param in params_to_freeze:
            param.requires_grad = False

    def unfreeze_params(self, params_to_unfreeze):
        for param in params_to_unfreeze:
            param.requires_grad = True

    def __generate_genotype(self, T: int, num_grains: int) -> List[List[int]]:
        """
        Args: T=8, num_grains=3
        Outputs: [[2, 2, 4], [2, 3, 3], [2, 4, 2], [3, 2, 3], [3, 3, 2], [4, 2, 2]]
        """
        if num_grains * 2 > T:
            raise ValueError()
        genotype = []

        def _generate(current: List[int], remaining_T: int, remaining_grains: int):
            if len(current) == num_grains:
                if remaining_T == 0:
                    result = []
                    for i, count in enumerate(current):
                        result.extend([i] * count)
                    genotype.append(list(current))
                return
            min_size, max_size = 2, remaining_T - 2 * (remaining_grains - 1)
            for size in range(min_size, max_size + 1):
                _generate(current + [size], remaining_T - size, remaining_grains - 1)

        _generate([], T, num_grains)
        assignment = []
        for element in genotype:
            element_list = []
            for grain_idx, steps in enumerate(element):
                element_list.extend([grain_idx] * steps)
            assignment.append(element_list)
        return assignment

    def __get_architecture(self, genotype):
        architecture = nn.ParameterList()
        arch_weights = []
        for i in range(self.num_edges):
            random_offset = torch.normal(mean=0.05, std=0.05, size=(1,))
            random_offset = torch.clamp(random_offset, min=0, max=0.1)
            if self.mode == "SoftMax":
                arch_param = nn.Parameter(
                    0.01 * torch.randn(1).abs(),
                    requires_grad=True
                )
            else:
                arch_param = nn.Parameter(
                    torch.ones(1) * 0.1 + random_offset * 0.02,
                    requires_grad=True
                )
            architecture.append(arch_param)
        for i, arch_param in enumerate(architecture):
            arch_weights.append(arch_param.item())
        print(f"arch_weights:{arch_weights}")
        return architecture

    def get_optimal_adaptive_T(self):
        arch_list = list(self.architecture)
        index = arch_list.index(max(arch_list))
        return self.genotype[index]

    def forward(self, X):
        losses, outputs = [], []
        for i, fs_neuron in enumerate(self.fs_neurons):
            Y, S = fs_neuron(X, None, True)
            loss_i = F.mse_loss(Y, X, reduction='mean')
            losses.append(loss_i)
            outputs.append(Y)
        arch_params = torch.stack(list(self.architecture)).squeeze()
        arch_weights = F.softmax(arch_params, dim=0)
        output = torch.zeros_like(X)
        for i, Y in enumerate(outputs):
            output += arch_weights[i] * Y
        total_loss = F.mse_loss(output, X, reduction='mean') + \
                     torch.sum(torch.stack(losses) * arch_weights) * 10
        return total_loss, self.architecture


# ============================================================
# Training logic
# ============================================================

def train_phase_search(X, args, device, tau=None, mode=None):
    model = PhaseSearch(X, args.T, args.num_grains, args.beta, args.epoch_inner,
                        device, tau=tau, mode=mode)
    model.to(device)

    best_error, best_T = torch.inf, None
    loss, _ = model(X)
    if loss.item() <= best_error:
        best_error = loss.item()
        best_T = model.get_optimal_adaptive_T()

    if mode == "SoftMax":
        theta_params, h_params, d_params, v0_params, arch_params = model.get_params()
        theta_h_optimizer = optim.Adam(h_params + theta_params, lr=args.lr)
        d_optimizer = optim.Adam(d_params, lr=args.lr)
        v0_optimizer = optim.Adam(v0_params, lr=args.lr)
        arch_optimizer = optim.Adam(arch_params, lr=args.lr * 10)

        for epoch in range(args.epoch_outer):
            # optimize h, theta
            model.freeze_params(arch_params + d_params)
            model.unfreeze_params(theta_params + h_params)
            for inner_epoch in range(args.epoch_inner):
                theta_h_optimizer.zero_grad()
                loss, _ = model(X)
                loss.backward()
                theta_h_optimizer.step()
                v0_optimizer.step()

            # optimize d
            model.freeze_params(arch_params + theta_params + h_params)
            model.unfreeze_params(d_params)
            for inner_epoch in range(args.epoch_inner):
                d_optimizer.zero_grad()
                loss, _ = model(X)
                loss.backward()
                d_optimizer.step()
                v0_optimizer.step()

            # optimize architecture
            model.freeze_params(theta_params + h_params + d_params)
            model.unfreeze_params(arch_params)
            arch_optimizer.zero_grad()
            loss, architecture = model(X)
            loss.backward()
            arch_optimizer.step()

            if loss.item() <= best_error:
                best_error = loss.item()
                best_T = model.get_optimal_adaptive_T()

            print(f"Epoch {epoch+1}/{args.epoch_outer}, Loss: {loss.item():.6f}")
            arch_weights = [p.item() for p in architecture]
            print(f"arch_weights:{arch_weights}")
            print(f"optim T:{best_T}")

        return model.num_grains, best_T

    else:
        theta_params, h_params, d_params, arch_params = model.get_params()
        theta_h_optimizer = optim.Adam(h_params + theta_params, lr=args.lr)
        d_optimizer = optim.Adam(d_params, lr=args.lr)
        arch_optimizer = optim.Adam(arch_params, lr=args.lr * 10)

        for epoch in range(args.epoch_outer):
            # optimize h, theta
            model.freeze_params(arch_params + d_params)
            model.unfreeze_params(theta_params + h_params)
            for inner_epoch in range(args.epoch_inner):
                theta_h_optimizer.zero_grad()
                loss, _ = model(X)
                loss.backward()
                theta_h_optimizer.step()

            # optimize d
            model.freeze_params(arch_params + theta_params + h_params)
            model.unfreeze_params(d_params)
            for inner_epoch in range(args.epoch_inner):
                d_optimizer.zero_grad()
                loss, _ = model(X)
                loss.backward()
                d_optimizer.step()

            # optimize architecture
            model.freeze_params(theta_params + h_params + d_params)
            model.unfreeze_params(arch_params)
            arch_optimizer.zero_grad()
            loss, architecture = model(X)
            loss.backward()
            arch_optimizer.step()

            print(f"Epoch {epoch+1}/{args.epoch_outer}, Loss: {loss.item():.6f}")

            if loss.item() <= best_error:
                best_error = loss.item()
                best_T = model.get_optimal_adaptive_T()

            arch_weights = [p.item() for p in architecture]
            print(f"arch_weights:{arch_weights}")
            print(f"optim T:{best_T}")

    return model.num_grains, best_T


# ============================================================
# Main search loop
# ============================================================

def train_search(args):
    base_keys = [
        "post_attention_layernorm.output",
        "input_layernorm.output",
        "self_attn.o_proj.input",
        "mlp.down_proj.input",
        "self_attn.q_Identity.input",
        "self_attn.k_Identity.input",
        "self_attn.v_Identity.input",
        "self_attn.softmax_Identity.input",
        "mlp.up_proj.output",
        "mlp.silu_Identity.input",
    ]
    output_dict = {}

    # tau_dict = torch.load(f'../tau_dict.pth', map_location="cuda:0")
    tau = 1.0  # shared across all keys (global tau)

    save_dir = get_save_dir(args.model_name, args.T, args.num_grains)

    for i in range(args.start, args.end):
        # ---- model.norm.output (i == -1) ----
        if i == -1:
            activation = torch.load(
                f'./activation_dir/{args.model_name}/layers/down_model.norm.output.pth',
                map_location="cuda:0")
            activation_value = activation["model.norm.output"]
            print(f"load down_model.norm.output.pth success")

            X = activation_value.view(-1).to(device="cuda:0").unsqueeze(dim=1)
            num_grains, T_list = train_phase_search(X, args, device, tau, mode=None)

            print(f'num_grains={num_grains}, T_list={T_list}')
            key = "model.norm.output"
            # norm output is bare list (retrain_decoupled expects list, not tuple)
            output_dict[key] = T_list

            # save single-layer immediately
            save_single_layer(-1, {key: output_dict[key]}, save_dir)
            continue

        # ---- regular layer i ----
        activation = torch.load(
            f'./activation_dir/{args.model_name}/layers/down_activation_stat_{i}.pth',
            map_location="cuda:0")
        print(f"load down_activation_stat_{i}.pth success")

        layer_dict = {}

        for bk in base_keys:
            activation_value = activation[f"model.layers.{i}.{bk}"]
            X = activation_value.view(-1).to(device="cuda:0").unsqueeze(dim=1)

            # FIXED: determine mode per-key (not once-and-leaking)
            mode = "SoftMax" if "softmax" in bk else None

            num_grains, T_list = train_phase_search(X, args, device, tau, mode=mode)
            print("----------------------------------------------------------------")
            print(f"model.layers.{i}.{bk}")
            print(f'num_grains={num_grains}, T_list={T_list}')

            key = f"model.layers.{i}.{bk}"
            output_dict[key] = (args.num_grains, T_list)
            layer_dict[key] = (args.num_grains, T_list)

        # save single-layer immediately after each layer finishes
        save_single_layer(i, layer_dict, save_dir)

    # ---- end of loop: save multi-layer batch ----
    save_multi_layer_batch(args.start, args.end, output_dict, save_dir)

    # ---- save final merged result ----
    os.makedirs(save_dir, exist_ok=True)
    final_path = os.path.join(save_dir, args.save_name)
    torch.save(output_dict, final_path)
    print(f"[save] final: {final_path}  ({len(output_dict)} keys)")
    print("save success")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device(f"{args.device}:{args.gpu}")
    elif torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        sys.exit(2)

    if args.merge_only:
        # ---- merge-only mode: no training, just collect existing files ----
        save_dir = get_save_dir(args.model_name, args.T, args.num_grains)
        print(f"[merge_only] collecting from: {save_dir}")
        print(f"[merge_only] merge_source = {args.merge_source}")

        merged = merge_all_results(save_dir, merge_source=args.merge_source,
                                   num_grains=args.num_grains)

        if not merged:
            print("[merge_only] ERROR: no results found. Did you run phase_search first?")
            sys.exit(1)

        final_path = os.path.join(save_dir, args.save_name)
        torch.save(merged, final_path)
        print(f"[merge_only] merged {len(merged)} keys → {final_path}")
        print("merge success")
    else:
        train_search(args)

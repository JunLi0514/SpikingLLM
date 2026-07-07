
# personal reproduction of Official implement for **[Distribution-Aware Multi-Granularity Phase Coding: Towards Lower Conversion Error for Spike-Driven Large Language Models](https://openreview.net/pdf/5255ebcb400a21b8843aecd0adfb598bce09871d.pdf)** [ICLR 2026]

复现流程
1 获得activation范围和tau_dict的取值
2 进入grainanalysis，phase_search 设定T开始搜索给不同T设置什么grain。用到的输入有tau_dict和act。此时会优化 genotype(哪个t用哪个grain)、神经元的3个参数，输出search_arch.pth，记录了哪个t哪个grain；
3 retrain_decoupled 输入act， tau, search_arc几个pth，重新训练神经元获得retrain.pth
4 用自己训练获得的神经元eval数据集。
总之tau=1.0的时候，截至2026.7.7,wikitest2 困惑度3k+。但是tau大部分是min max获得的就没有问题（mlp up_proj output由于多一层comment，所以它们的tau=1.0了，对结果似乎没有影响。）

![Overview](/main_figure.jpg)
## Installation
```
conda create -n prefixquant python==3.9.21

conda activate prefixquant


pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124


git clone git@github.com:Dao-AILab/fast-hadamard-transform.git
cd fast-hadamard-transform
pip install -e .

cd ..
pip install -r requirements.txt
```

pip install datasets==3.5.0 

## ANN-to-SNN Conversion
We provide two example commands to convert Llama-2-7B and Llama-3-8B to spiking neural networks:

```
bash run_scripts/run_phase_ours_llama2.sh
bash run_scripts/run_phase_ours_llama3.sh
```
Optional arguments:
* `--neuron_path` corresponds to the optimized phase coding base described in the paper (path located at `../GrainAnalysis/retrain_dir/`)
* `--T` corresponds to the number of time steps for spiking neurons.

Example with arguments:
```
bash run_scripts/run_phase_ours_llama2.sh 0,1
```
Note: `0,1` specifies the GPU indices (e.g., for A100 GPUs) to be used for the conversion process. 

The log file will be saved at: `run_scripts/log/*.txt`

## GrainAnalysis: Neuron Optimization Pipeline

The spiking neuron parameters used by `--neuron_path` are obtained through a NAS-style search-and-retrain pipeline under `GrainAnalysis/`. There are four stages:

### Stage 1 — Collect Activation Statistics

Run `phase_main.py` (or `phase_debug.py`) with `--stop_after 3` to stop after the activation-collection stage. This saves per-channel max activation values to `GrainAnalysis/activation_dir/{model_name}/layers/`.

At the same time, `phase/phase_util.py` collects each neuron's initial `tau` (membrane time constant, derived from the activation max) into a module-level `tau_dict`. After the activation collection pass, call `save_tau_dict()` to persist it:

```python
from phase.phase_util import save_tau_dict
save_tau_dict("GrainAnalysis/activation_dir/Llama-2-7B-hf/tau_dict.pth")
```

### Stage 2 — Genotype Search (NAS)

`phase_search.py` searches for the optimal **genotype** — how to allocate *T* time steps across *K* grains. Each grain has its own learnable `(d, h, theta)` parameters, and a DARTS-style architecture weight selects among all valid grain-partition candidates (e.g. for T=8, K=3: `[0,0,1,1,1,2,2,2]`). The search objective is to minimize MSE between the phase-neuron output and the original activation.

```bash
cd GrainAnalysis
python phase_search.py --model_name Llama-2-7B-hf --T 8 --num_grains 3
# Output: arch_dir/{model}-T-{T}-grains-{num_grains}/search_arch.pth
```

Use `--start` / `--end` to parallelize across GPUs (e.g. layers 0–7 on GPU 0, 8–15 on GPU 1). Intermediate results are saved per-layer under `single-layer/`. After all layers finish, use `--merge_only` to assemble the final dict:

```bash
python phase_search.py --merge_only --model_name Llama-2-7B-hf --T 8 --num_grains 3
```

### Stage 3 — Decoupled Retrain

`retrain_decoupled.py` fixes the genotype found in Stage 2 and retrains the neuron parameters `(d, h, theta)` (and `v0` for softmax/silu neurons) with alternating optimization — freeze `d`, train `(h, theta)`; freeze `(h, theta)`, train `d`. This step requires both a `tau_dict.pth` (from Stage 1) and a `search_arch.pth` (from Stage 2):

```bash
python retrain_decoupled.py --model_name Llama-2-7B-hf --T 8 --num_grains 3 \
  --tau_dict_path activation_dir/Llama-2-7B-hf/tau_dict.pth \
  --search_arch_path arch_dir/Llama-2-7B-hf-T-8-grains-3/search_arch.pth \
  --epoch_inner 5000
# Output: retrain_dir/{model}-T-{T}-grains-{num_grains}/retrain.pth
```

### Stage 4 — Evaluate

Feed the `retrain.pth` back into `phase_main.py` via `--neuron_path`:

```bash
python phase_main.py ... --neuron_path GrainAnalysis/retrain_dir/Llama-2-7B-hf-T-8-grains-3/retrain.pth --T 8
```  
> Note: The results presented here reflect the latest optimized configuration. 
Compared to the values reported in the original paper, current results show 
improved performance due to refined hyperparameter tuning.

| Model | T | Grain | Wiki2 | Wino | ArcC | ArcE | PiQA |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| LLaMA-2-7B |  8 | 2 | 6.50 | 70.56 | 46.33 | 73.70 | 78.35 |  
| LLaMA-2-7B |  8 | 3 | 6.31 | 70.48 | 46.25 | 73.82 | 78.29 |
| LLaMA-2-7B | 10 | 2 | 5.50 | 70.48 | 46.50 | 73.91 | 78.29 |  
| LLaMA-2-7B | 10 | 3 | 5.50 | 70.48 | 46.33 | 73.86 | 78.35 |  

| Model | T | Grain | Wiki2 | Wino | ArcC | ArcE | PiQA |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| LLaMA-3-8B | 6 | 2 | 7.60 | 69.77 | 48.55 | 74.75 | 79.11 |
| LLaMA-3-8B | 6 | 3 | 7.79 | 70.80 | 48.81 | 74.96 | 79.05 |
| LLaMA-3-8B | 8 | 2 | 6.34 | 72.93 | 54.01 | 77.44 | 80.63 | 
| LLaMA-3-8B | 8 | 3 | 6.33 | 73.72 | 53.41 | 77.36 | 80.36 |

## Reproduction

We have reproduced the results for Llama-2-7B (T=8, grains=3) at two levels:

| # | Method | WikiText-2 ↓ | WinoGrande | ARC-C | ARC-E | PiQA |
|---|--------|:---:|:---:|:---:|:---:|:---:|
| — | Paper (reported) | 6.31 | 70.48 | 46.25 | 73.82 | 78.29 |
| 1 | Use provided `retrain_paper.pth` directly | 6.31 | 70.09 | 46.50 | 73.74 | 78.51 |
| 2 | Full pipeline (tau_dict → search → retrain) | **6.28** | 70.48 | 46.59 | 73.78 | 78.45 |

Both methods match the paper within noise, confirming the GrainAnalysis pipeline is fully reproducible.

> **Note:** Setting all `tau_dict` values to 1.0 (skipping proper tau collection) causes WikiText-2 to collapse to ~6631. The per-channel tau values derived from activation statistics (typically 4.0–4.5 for layernorm outputs) are essential.

To reproduce, follow the GrainAnalysis pipeline above to obtain the neuron parameters, then run:

```bash
python phase_main.py \
  --model_path /path/to/Llama-2-7b \
  --neuron_path GrainAnalysis/retrain_dir/Llama-2-7B-hf-T-8-grains-3/retrain.pth \
  --T 8 --wbits 8 --pre_rotate --down_online_had --qk_online_had --set_prefixed_tokens \
  --eval_ppl --eval_tasks piqa,arc_easy,arc_challenge,winogrande
```

## Citation

If you find our work helpful, please consider citing our paper:

```bibtex
@inproceedings{zhengdistribution,
  title={Distribution-Aware Multi-Granularity Phase Coding: Towards Lower Conversion Error for Spike-Driven Large Language Models},
  author={Zheng, Hanyuan and Zhang, Haozhen and Chen, Tianshuo and Liu, Zhaogeng and Chang, Yi and Gu, Bin},
  booktitle={The Fourteenth International Conference on Learning Representations}
}
```

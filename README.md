# Iterative Vectors

**Iterative Vectors: In-Context Gradient Steering without Backpropagation**  
Yiting Liu, Zhi-Hong Deng  
*ICML 2025*

![Iterative Vectors](./iv.png)

---

## 📌 Overview

Iterative Vectors (IV) is a novel approach for improving **in-context learning (ICL)** in large language models by **directly editing internal activations** using *simulated gradients*.  
Unlike traditional fine-tuning approaches, Iterative Vectors operate entirely at **inference time** and do **not require backpropagation or weight updates**.

Key characteristics of Iterative Vectors:
- ❌ No backpropagation
- ❌ No model weight updates
- ✅ Activation-level adaptation
- ✅ Model-agnostic
- ✅ Works with in-context examples

The method achieves significant performance improvements across multiple NLP benchmarks.

---

## ⚙️ Environment Setup

You can set up the environment using **either of the following two methods**.

---

## 🔹 Method 1: Conda-based Setup (Recommended)

### 1. Create and Activate Conda Environment

conda env create -n <env_name> -f env.yaml
conda activate <env_name>


### Pip Setup

python -m venv iv_env  
source iv_env/bin/activate (Linux / macOS)  
iv_env\Scripts\activate (Windows)  
pip install -r requirements.txt  

## Model Configuration

Create a configuration file at `config/yours.yaml`:

models:  
&nbsp;&nbsp;llama-2-7b: meta-llama/Llama-2-7b  
&nbsp;&nbsp;llama-2-13b: /path/to/llama-2-13b-hf  
&nbsp;&nbsp;llama-2-70b: ...  
&nbsp;&nbsp;llama-3.1-8b: ...  
&nbsp;&nbsp;gpt-j-6b: EleutherAI/gpt-j-6b  

## Running Iterative Vectors

Example command using GPT-J-6B on AG News with GPU 0:

python iv.py -c config/yours.yaml -g 0 -m gpt-j-6b -t ag_news -s 4 --strength 0.5 --ext-strength 0.3 --run-test  

## Command-Line Arguments

-c : Path to configuration file  
-g : GPU ID  
-m : Model name  
-t : Task / dataset  
-s : Number of in-context examples (shots)  
--strength : Iterative Vector update strength  
--ext-strength : External vector strength  
--run-test : Run evaluation on the test set  

## Scripts

For hyperparameter search, Task Vectors, and Function Vectors, see `scripts.sh`.

## Performance and Logging Tips

export HF_DATASETS_OFFLINE=1  
export PYTHONWARNINGS="ignore::DeprecationWarning"  
export DATASETS_VERBOSITY=error  
export TRANSFORMERS_VERBOSITY=error  

Set HF_DATASETS_OFFLINE=1 only after datasets are cached.

## Reference

Iterative Vectors: In-Context Gradient Steering without Backpropagation  
Yiting Liu, Zhi-Hong Deng  
Forty-Second International Conference on Machine Learning (ICML 2025)

## Citation

@inproceedings{liu2025IterativeVectors,  
&nbsp;&nbsp;title={Iterative Vectors: In-Context Gradient Steering without Backpropagation},  
&nbsp;&nbsp;booktitle={Forty-Second International Conference on Machine Learning},  
&nbsp;&nbsp;author={Liu, Yiting and Deng, Zhi-Hong},  
&nbsp;&nbsp;year={2025},  
&nbsp;&nbsp;url={https://openreview.net/forum?id=1v3XEcRMyP}  
}

## Notes

This repository is intended for research and educational purposes. Results may vary depending on the model, dataset, and hyperparameter choices.

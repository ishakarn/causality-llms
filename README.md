# [PAPER TITLE]

Anonymous submission — NeurIPS 2026 Datasets and Benchmarks Track.

---

## What this is

Code and evaluation results for our study on LLM causal reasoning robustness. We build on [CLadder](https://github.com/causalNLP/cladder) (Jin et al., NeurIPS 2023) by applying surface-form perturbations to its questions and measuring how much model accuracy drops. We also fine-tune several open-source models on CLadder with LoRA and test whether fine-tuning improves robustness to these perturbations.

We don't release a new dataset — the underlying questions come from CLadder v1. What we release is the intervention code, fine-tuning pipeline, and evaluation scripts.

---

## Setup

```bash
conda create -n cladder_olmo python=3.10
conda activate cladder_olmo
pip install vllm transformers peft datasets accelerate pandas matplotlib seaborn
```

Get CLadder data:
```bash
# Download from https://huggingface.co/datasets/causalnlp/CLadder
# or use the copy in cladder_interventions/data/
```

Set environment variables before running any scripts:
```bash
export HF_TOKEN=<your_huggingface_token>
export WORKDIR=<path_to_repo_root>          # defaults to script location if unset
export SCRATCH_CACHE=<path_to_cache_dir>    # large model files go here
```

---

## Repo layout

```
cladder_interventions/
    interventions/         # word_replace.py, story_swap.py, nonsense_replace.py, etc.
    cladder-main/          # CLadder codebase (Jin et al. 2023); only assets/stories/ is used
    data/                  # CLadder v1 source splits (gitignored; download separately)

data/
    cladder-v1-q-easy.json
    cladder-v1-q-hard.json
    cladder-v1-q-anticommonsense.json
    cladder-v1-q-noncommonsense.json
    cladder-v1-meta-models.json
    # intervened datasets are gitignored — regenerate with intervention scripts

finetune/
    configs/               # LoRA training configs per model
    train.py               # HF Trainer + LoRA
    infer_vllm_multi.py    # vLLM inference, logit-mode yes/no scoring
    compile_results.py     # aggregates scores → outputs/results_summary.csv
    plots/                 # figure generation scripts
    scripts/               # SLURM job scripts

cladder_score_yesno.py     # scorer: accuracy + per-query-type breakdown
grapher.py                 # timeline plot (model accuracy vs. release date)
```

---

## Interventions

We test 7 interventions. Each preserves the correct answer while changing the surface form of the question:

| ID | Name | What changes |
|----|------|-------------|
| 67 | Word Replace | variable names → random English nouns |
| 68 | Number Replace | variable names → random 3-digit integers |
| 81 | Story Swap | story framing swapped for another with the same causal graph topology |
| 86 | Nonsense Replace | variable names → random 4-letter nonsense strings |
| 94 | Drop Background | removes the background text entirely |
| 96 | Probability Expander | inflates probability values into the millions |
| 100 | Drop Graph Structure | removes the causal relationship description, keeps only the story title |

To regenerate intervened datasets:
```bash
cd cladder_interventions/interventions
python word_replace.py --splits easy hard anticommonsense noncommonsense
python story_swap.py   --splits easy hard anticommonsense noncommonsense
# etc.
```

---

## Models

| Model | Condition | HF ID |
|-------|-----------|-------|
| Qwen2.5-3B-Instruct | base + LoRA | `Qwen/Qwen2.5-3B-Instruct` |
| Llama-3.1-8B-Instruct | base + LoRA | `meta-llama/Llama-3.1-8B-Instruct` |
| OLMo-3.1-32B-Instruct | base + LoRA | `allenai/OLMo-3.1-32B-Instruct` |
| GPT-OSS-20B | base | `openai/gpt-oss-20b` |
| GPT-5-Nano | base | OpenAI API |
| GPT-5.5 | base | OpenAI API |

LoRA fine-tuning used 2,000 CLadder training examples, rank 16, bf16.

---

## Running things

**Baseline inference:**
```bash
python finetune/infer_vllm_multi.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --pairs data/cladder-v1-q-easy.json:outputs/cladder-v1-q-easy/baseline/qwen-baseline/qwen-baseline.jsonl \
    --run_ids qwen-baseline \
    --max_model_len 8192

python cladder_score_yesno.py \
    --pred_jsonl outputs/cladder-v1-q-easy/baseline/qwen-baseline/qwen-baseline.jsonl \
    --out_dir outputs/cladder-v1-q-easy/baseline/qwen-baseline/score
```

**Intervention eval (SLURM):**
```bash
bash finetune/scripts/intervention_evals/eval_100_drop_graph_structure.sh
```

**LoRA fine-tuning:**
```bash
CONFIG=finetune/configs/qwen25-3b-instruct.yaml sbatch --export=ALL finetune/finetune.sh
```

**Compile all results:**
```bash
python finetune/compile_results.py
# writes outputs/results_summary.csv
```

**Plots:**
```bash
python finetune/plots/plot_paper_interventions.py
python finetune/plots/plot_dumbbell_components.py
python finetune/plots/plot_paper_graded.py
```

---

## Results

`outputs/results_summary.csv` has accuracy for every model × intervention × split combination. Columns: `model`, `condition_type`, `intervention`, `split`, `n`, `acc_all`, `delta_acc_all`.

---

## Citation

If you use this work, please also cite the original CLadder paper:

```bibtex
@inproceedings{jin2023cladder,
    author = {Zhijing Jin and Yuen Chen and Felix Leeb and Luigi Gresele and
              Ojasv Kamal and Zhiheng Lyu and Kevin Blin and Fernando Gonzalez and
              Max Kleiman-Weiner and Mrinmaya Sachan and Bernhard Sch{\"o}lkopf},
    title  = {{CL}adder: Assessing Causal Reasoning in Language Models},
    year   = {2023},
    booktitle = {NeurIPS},
}
```

---

## License

Code: [PLACEHOLDER]. CLadder data is subject to its own license; see `cladder_interventions/cladder-main/LICENSE`.

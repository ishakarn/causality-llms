"""
plot_attention.py — Extract and visualize attention scores for CLADDER queries.

Usage (zero-shot base model, one or many queries):
    python finetune/plot_attention.py \
        --config finetune/configs/olmo3-7b-instruct.yaml \
        --query_ids 23190 13239 17897 14947 37522 22573 7399 11827 329 11380 \
        --out_dir finetune/attention_plots

With a LoRA adapter (fine-tuned):
    python finetune/plot_attention.py \
        --config finetune/configs/olmo3-7b-instruct.yaml \
        --adapter finetune/checkpoints/olmo3-7b-instruct-lora/final_adapter \
        --query_ids 23190 13239 17897 14947 37522 22573 7399 11827 329 11380 \
        --out_dir finetune/attention_plots

Options:
    --layers        Which layer indices to plot (default: last 4).
    --heads         Which head indices to plot (default: first 4).
    --avg_heads     Average over all heads per layer (summary grid per layer).
    --max_tokens    Truncate input to this many tokens (default: 256).
    --max_display   Max tokens shown on heatmap axis (default: 64).
    --dtype         bf16 | fp16 | fp32 (default: bf16)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from cladder_infer_yesno import build_chat_input, resolve_system_prompt


DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_queries_by_id(queries_json: str, ids: list[int]) -> dict[int, dict]:
    data = json.loads(Path(queries_json).read_text())
    if isinstance(data, dict) and "queries" in data:
        data = data["queries"]
    wanted = set(ids)
    result = {q["question_id"]: q for q in data if q["question_id"] in wanted}
    missing = wanted - set(result)
    if missing:
        raise ValueError(f"query_ids not found: {sorted(missing)}")
    return result


def tokenize_prompt(tokenizer, query: dict, model_name: str, max_tokens: int):
    system_msg = resolve_system_prompt("AUTO", model_name)
    rendered = build_chat_input(tokenizer, query["text"], system_msg=system_msg)
    enc = tokenizer(
        rendered,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
    )
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    tokens = [t.replace("▁", " ").replace("Ġ", " ").replace("Ċ", "↵") for t in tokens]
    return enc, tokens


# ─────────────────────────────────────────────────────────────────────────────
# Model helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_forward(model, enc, device):
    with torch.no_grad():
        out = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc.get("attention_mask", None),
            output_attentions=True,
        )
    return out.attentions  # tuple: (n_layers,) each (1, n_heads, S, S)


def pick_indices(requested: list[int] | None, total: int, default_n: int, from_end=False) -> list[int]:
    if requested:
        return [i for i in requested if 0 <= i < total]
    if from_end:
        return list(range(max(0, total - default_n), total))
    return list(range(min(default_n, total)))


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_attention_heatmap(
    attn: np.ndarray,
    tokens: list[str],
    title: str,
    out_path: Path,
    max_display: int = 64,
):
    n = min(len(tokens), max_display)
    a = attn[:n, :n]
    toks = tokens[:n]

    fs = max(4, 8 - n // 16)
    size = max(7, n * 0.17)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    im = ax.imshow(a, aspect="auto", cmap="viridis", vmin=0)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_xticks(range(n)); ax.set_xticklabels(toks, rotation=90, fontsize=fs)
    ax.set_yticks(range(n)); ax.set_yticklabels(toks, fontsize=fs)
    ax.set_xlabel("Key (attended to)"); ax.set_ylabel("Query (attending from)")
    ax.set_title(title, fontsize=9, pad=6)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


def plot_layer_avg_grid(
    layer_attns: dict[int, np.ndarray],
    tokens: list[str],
    title_prefix: str,
    out_path: Path,
    max_display: int = 64,
):
    """Grid of per-layer heatmaps (heads already averaged), with shared token axis labels."""
    n_layers = len(layer_attns)
    n = min(len(tokens), max_display)
    toks = tokens[:n]
    fs = max(3, 7 - n // 16)

    cols = min(4, n_layers)
    rows = (n_layers + cols - 1) // cols
    cell = max(3.5, n * 0.13)
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * cell, rows * cell * 0.9),
                             squeeze=False)

    for ax_i, (layer_idx, attn) in enumerate(sorted(layer_attns.items())):
        r, c = divmod(ax_i, cols)
        ax = axes[r][c]
        im = ax.imshow(attn[:n, :n], aspect="auto", cmap="viridis", vmin=0)
        ax.set_title(f"Layer {layer_idx}", fontsize=8)

        # token labels only on edges
        if r == rows - 1:
            ax.set_xticks(range(n))
            ax.set_xticklabels(toks, rotation=90, fontsize=fs)
        else:
            ax.set_xticks([])
        if c == 0:
            ax.set_yticks(range(n))
            ax.set_yticklabels(toks, fontsize=fs)
        else:
            ax.set_yticks([])

    # hide unused subplots
    for ax_i in range(n_layers, rows * cols):
        r, c = divmod(ax_i, cols)
        axes[r][c].axis("off")

    fig.suptitle(f"{title_prefix}\n(avg over heads)", fontsize=10, y=1.01)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


def plot_last_token_row(
    layer_attns: dict[int, np.ndarray],
    tokens: list[str],
    title_prefix: str,
    out_path: Path,
    max_display: int = 64,
):
    """
    For each layer, show only the attention distribution of the LAST token
    (the answer position) over all prior tokens — a 1-D bar chart per layer.
    Useful for seeing which context tokens the model attends to when deciding yes/no.
    """
    n_layers = len(layer_attns)
    n = min(len(tokens), max_display)
    toks = tokens[:n]
    fs = max(4, 8 - n // 16)

    cols = min(4, n_layers)
    rows = (n_layers + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * max(5, n * 0.15), rows * 2.5),
                             squeeze=False)

    for ax_i, (layer_idx, attn) in enumerate(sorted(layer_attns.items())):
        r, c = divmod(ax_i, cols)
        ax = axes[r][c]
        last_row = attn[-1, :n]   # attention FROM last token TO all others
        ax.bar(range(n), last_row, color="steelblue", width=0.8)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_xticks(range(n))
        ax.set_xticklabels(toks, rotation=90, fontsize=fs)
        ax.set_title(f"Layer {layer_idx}", fontsize=8)
        ax.set_ylabel("Attn weight", fontsize=7)
        ax.yaxis.set_tick_params(labelsize=6)

    for ax_i in range(n_layers, rows * cols):
        r, c = divmod(ax_i, cols)
        axes[r][c].axis("off")

    fig.suptitle(f"{title_prefix}\nlast-token attention row (avg heads)", fontsize=10, y=1.01)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-query processing
# ─────────────────────────────────────────────────────────────────────────────

def process_query(
    query: dict,
    model,
    tokenizer,
    model_id: str,
    device: str,
    layer_indices: list[int] | None,
    head_indices: list[int] | None,
    avg_heads: bool,
    max_tokens: int,
    max_display: int,
    out_subdir: Path,
    n_layers_total: int,
    n_heads_total: int,
):
    qid       = query["question_id"]
    qtype     = query["query_type"]
    answer    = query["answer"]
    run_label = f"qid{qid}_{qtype}"

    print(f"\n── qid={qid}  type={qtype}  answer={answer}")
    print(f"   {query['text'][:100]} …")

    enc, tokens = tokenize_prompt(tokenizer, query, model_id, max_tokens)
    seq_len = enc["input_ids"].shape[1]
    print(f"   seq_len={seq_len}")

    all_attns = run_forward(model, enc, device)

    layers = pick_indices(layer_indices, n_layers_total, default_n=4, from_end=True)
    heads  = pick_indices(head_indices,  n_heads_total,  default_n=4, from_end=False)

    # avg-over-heads dict for summary plots (always computed)
    layer_avg = {
        li: all_attns[li][0].float().cpu().numpy().mean(axis=0)
        for li in layers
    }

    title_prefix = f"{run_label} | answer={answer}"

    # 1. Layer summary grid (avg heads)
    plot_layer_avg_grid(
        layer_avg, tokens,
        title_prefix=title_prefix,
        out_path=out_subdir / f"{run_label}_layers_avg.png",
        max_display=max_display,
    )

    # 2. Last-token attention bar chart (avg heads) — most informative for yes/no
    plot_last_token_row(
        layer_avg, tokens,
        title_prefix=title_prefix,
        out_path=out_subdir / f"{run_label}_last_token.png",
        max_display=max_display,
    )

    # 3. Per-head heatmaps (only if not avg_heads mode)
    if not avg_heads:
        for li in layers:
            a = all_attns[li][0].float().cpu().numpy()
            for hi in heads:
                plot_attention_heatmap(
                    a[hi], tokens,
                    title=f"{run_label} | L{li} H{hi} | answer={answer}",
                    out_path=out_subdir / f"{run_label}_L{li:02d}_H{hi:02d}.png",
                    max_display=max_display,
                )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       type=str, default=None)
    parser.add_argument("--model",        type=str, default=None)
    parser.add_argument("--adapter",      type=str, default=None,
                        help="PEFT LoRA adapter dir (optional).")
    parser.add_argument("--query_ids",    type=int, nargs="+", required=True,
                        help="One or more question_ids to visualize.")
    parser.add_argument("--queries_json", type=str, default="data/queries_easy.json")
    parser.add_argument("--out_dir",      type=str, default="finetune/attention_plots")
    parser.add_argument("--layers",       type=int, nargs="+", default=None)
    parser.add_argument("--heads",        type=int, nargs="+", default=None)
    parser.add_argument("--avg_heads",    action="store_true",
                        help="Skip per-head plots; only produce summary grids.")
    parser.add_argument("--max_tokens",   type=int, default=256)
    parser.add_argument("--max_display",  type=int, default=64)
    parser.add_argument("--dtype",        type=str, default="bf16", choices=list(DTYPE_MAP))
    args = parser.parse_args()

    # ── resolve model_id ──────────────────────────────────────────────────────
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text())
        model_id = cfg["model_id"]
        trust_remote = cfg.get("trust_remote_code", False)
    elif args.model:
        model_id = args.model
        trust_remote = False
    else:
        parser.error("Provide --config or --model.")

    dtype  = DTYPE_MAP[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[model] {model_id}  dtype={args.dtype}  device={device}")

    # ── output subdir ─────────────────────────────────────────────────────────
    model_slug = model_id.replace("/", "_")
    if args.adapter:
        model_slug += "_finetuned"
    out_subdir = Path(args.out_dir) / model_slug

    # ── load model ────────────────────────────────────────────────────────────
    print("[load] tokenizer …")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote)

    print("[load] model …")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=trust_remote,
        attn_implementation="eager",   # flash-attn doesn't return attention weights
    )

    if args.adapter:
        from peft import PeftModel
        print(f"[load] LoRA adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()

    model.eval()

    # infer dims from model config
    n_layers = model.config.num_hidden_layers
    n_heads  = model.config.num_attention_heads
    print(f"[model] {n_layers} layers × {n_heads} heads")

    # ── load queries ──────────────────────────────────────────────────────────
    queries = load_queries_by_id(args.queries_json, args.query_ids)
    print(f"[data] loaded {len(queries)} queries")

    # ── process each query ────────────────────────────────────────────────────
    for qid in args.query_ids:
        process_query(
            query=queries[qid],
            model=model,
            tokenizer=tok,
            model_id=model_id,
            device=device,
            layer_indices=args.layers,
            head_indices=args.heads,
            avg_heads=args.avg_heads,
            max_tokens=args.max_tokens,
            max_display=args.max_display,
            out_subdir=out_subdir,
            n_layers_total=n_layers,
            n_heads_total=n_heads,
        )

    print(f"\n[done] all plots → {out_subdir}/")


if __name__ == "__main__":
    main()

"""
plot_attention_highlight.py — Visualize attention as highlighted text.

For each query, shows the input text with words colored by how much the
last token (answer position) attends to them, averaged over the last N layers
and all heads. Subword tokens are merged back into words.

Outputs per query:
  - <out_dir>/<model_slug>/html/<run_label>.html   (easy to browse)
  - <out_dir>/<model_slug>/png/<run_label>.png     (for papers)

When --compare is set (base dir + finetuned dir given), produces a side-by-side
PNG comparing base vs fine-tuned attention on the same query.

Usage:
    # Single model (base only)
    python finetune/plot_attention_highlight.py \\
        --config finetune/configs/olmo3-7b-instruct.yaml \\
        --query_ids 23190 13239 17897 14947 37522 22573 7399 11827 329 11380

    # Base vs finetuned comparison
    python finetune/plot_attention_highlight.py \\
        --config  finetune/configs/olmo3-7b-instruct.yaml \\
        --adapter finetune/checkpoints/olmo3-7b-instruct-lora/final_adapter \\
        --compare \\
        --query_ids 23190 13239 17897 14947 37522 22573 7399 11827 329 11380
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from cladder_infer_yesno import build_chat_input, resolve_system_prompt

DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


# ─────────────────────────────────────────────────────────────────────────────
# Model / data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_queries_by_id(queries_json, ids):
    data = json.loads(Path(queries_json).read_text())
    if isinstance(data, dict) and "queries" in data:
        data = data["queries"]
    wanted = set(ids)
    result = {q["question_id"]: q for q in data if q["question_id"] in wanted}
    missing = wanted - set(result)
    if missing:
        raise ValueError(f"query_ids not found: {sorted(missing)}")
    return result


def load_model(model_id, dtype, device, trust_remote=False, adapter=None):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=trust_remote,
        attn_implementation="eager",
    )
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.eval()
    return tok, model


def _find_content_span(tokens_raw):
    """
    Return (start, end) indices of the actual question content tokens,
    stripping the chat-template wrapper (system prompt, role tags, etc.)

    Strategy: the user content starts after the last occurrence of a
    'user' role token followed by a newline, and ends before the final
    special token sequence that opens the assistant turn.
    Falls back to (0, len) if the pattern isn't found.
    """
    n = len(tokens_raw)

    # Find content start: position after the last 'user'+newline pair
    # Typical pattern: <|im_start|> user Ċ  <content>
    content_start = 0
    for i in range(n - 2):
        t = tokens_raw[i].lower().strip("▁Ġ ")
        if t in ("user", "human", "[inst]"):
            # skip one or two newline/space tokens then we're in content
            j = i + 1
            while j < n and tokens_raw[j] in ("Ċ", "ĊĊ", "\n", " "):
                j += 1
            content_start = j   # keep updating → take the *last* user turn

    # Find content end: last token before closing special token of user turn
    # Look for <|im_end|> / </s> / [/INST] after content_start
    content_end = n
    for i in range(content_start, n):
        t = tokens_raw[i]
        if t.startswith("<|im_end|") or t in ("</s>", "[/INST]", "<|eot_id|>"):
            content_end = i
            break

    if content_start >= content_end:
        return 0, n   # fallback: show everything
    return content_start, content_end


def _is_sink_token(tok_str):
    """True for tokens that are known attention sinks (special/structural)."""
    s = tok_str.strip()
    return (s.startswith("<") or s.startswith("[") or
            s in ("Ċ", "ĊĊ", "\n", "\n\n", "▁", "") or
            s.lower() in ("system", "user", "assistant", "human"))


def get_word_attention(model, tok, query, model_id, max_tokens, layers=None):
    """
    Returns (words, attn_weights) where:
      - words: list of display strings covering only the question content
      - attn_weights: np.ndarray shape (len(words),), normalised to [0,1]

    Key fix: we restrict to content tokens only (strip chat-template
    boilerplate) so attention-sink special tokens don't dominate the scale.
    """
    system_msg = resolve_system_prompt("AUTO", model_id)
    rendered = build_chat_input(tok, query["text"], system_msg=system_msg)
    enc = tok(rendered, return_tensors="pt", truncation=True, max_length=max_tokens)
    token_ids  = enc["input_ids"][0].tolist()
    tokens_raw = tok.convert_ids_to_tokens(token_ids)
    seq_len    = len(token_ids)

    with torch.no_grad():
        out = model(
            input_ids=enc["input_ids"].to(next(model.parameters()).device),
            attention_mask=enc.get("attention_mask"),
            output_attentions=True,
        )

    n_layers = len(out.attentions)
    # Determine which layers to average over
    if layers is None:
        # Default: penultimate layer only
        layer_indices = [n_layers - 2]
    else:
        layer_indices = [li for li in layers if 0 <= li < n_layers]
        if not layer_indices:
            layer_indices = [n_layers - 2]
    avg = np.zeros(seq_len, dtype=np.float32)
    for li in layer_indices:
        a = out.attentions[li][0].float().cpu().numpy()  # (heads, S, S)
        avg += a.mean(axis=0)[-1]
    avg /= len(layer_indices)

    # ── Restrict to content window (strips system prompt + role tags) ────────
    cs, ce = _find_content_span(tokens_raw)
    tokens_raw = tokens_raw[cs:ce]
    avg        = avg[cs:ce]

    # Zero out any remaining structural tokens within the content window
    for i, t in enumerate(tokens_raw):
        if _is_sink_token(t):
            avg[i] = 0.0

    # ── Merge subword tokens → words ─────────────────────────────────────────
    words, weights = [], []
    cur_word, cur_weight = "", 0.0
    for tok_str, w in zip(tokens_raw, avg):
        is_new = tok_str.startswith("▁") or tok_str.startswith("Ġ") or tok_str.startswith(" ")
        clean  = tok_str.lstrip("▁Ġ ")
        if not clean:
            continue
        if is_new and cur_word:
            words.append(cur_word); weights.append(cur_weight)
            cur_word, cur_weight = clean, float(w)
        else:
            cur_word += clean
            cur_weight = max(cur_weight, float(w))   # max-pool subwords
    if cur_word:
        words.append(cur_word); weights.append(cur_weight)

    weights = np.array(weights, dtype=np.float32)
    # Normalise to [0, 1] relative to this query
    wmax = weights.max()
    if wmax > 0:
        weights = weights / wmax
    return words, weights


# ─────────────────────────────────────────────────────────────────────────────
# HTML output
# ─────────────────────────────────────────────────────────────────────────────

def _rgba_str(r, g, b, alpha):
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha:.2f})"


def words_to_html_spans(words, weights, cmap):
    spans = []
    for w, wt in zip(words, weights):
        r, g, b, _ = cmap(float(wt))
        bg = _rgba_str(r, g, b, 0.85)
        txt_color = "#000" if wt < 0.6 else "#fff"
        spans.append(
            f'<span style="background:{bg};color:{txt_color};'
            f'border-radius:3px;padding:1px 3px;margin:1px;display:inline-block;">'
            f'{w}</span>'
        )
    return " ".join(spans)


def save_html(queries_data, model_label, out_path, cmap, layer_desc="penultimate layer"):
    """Write a single HTML file with all queries highlighted."""
    sections = []
    for qid, (words, weights, query) in sorted(queries_data.items()):
        spans = words_to_html_spans(words, weights, cmap)
        sections.append(f"""
        <div class="query">
          <div class="meta">
            <b>qid={qid}</b> &nbsp;|&nbsp;
            query_type=<b>{query['query_type']}</b> &nbsp;|&nbsp;
            answer=<b>{query['answer']}</b>
          </div>
          <div class="text">{spans}</div>
        </div>""")

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Attention Highlights — {model_label}</title>
<style>
  body {{ font-family: sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; }}
  h1   {{ font-size: 1.3em; }}
  .query {{ border: 1px solid #ddd; border-radius: 6px;
            padding: 14px 16px; margin-bottom: 18px; }}
  .meta  {{ font-size: 0.85em; color: #555; margin-bottom: 8px; }}
  .text  {{ font-size: 1.05em; line-height: 2.2em; }}
</style>
</head><body>
<h1>Attention Highlights — {model_label}</h1>
<p style="color:#666;font-size:.9em">
  Color intensity = how much the <em>answer token</em> attends to each word
  ({layer_desc}, avg all heads, normalised per query).
</p>
{''.join(sections)}
</body></html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"  saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib PNG output
# ─────────────────────────────────────────────────────────────────────────────

def render_highlighted_text_ax(ax, words, weights, cmap, title, fig_width_inches=14):
    """
    Render highlighted words as coloured boxes in a matplotlib Axes.
    Words flow naturally left-to-right, wrapping when the line is full
    (based on estimated character widths rather than a fixed word count).
    """
    ax.axis("off")
    ax.set_title(title, fontsize=10, pad=6, loc="left")

    fs = 9
    chars_per_line = int(fig_width_inches * 9.5)   # ~9.5 chars per inch at fs=9

    # Build lines by character-width accumulation
    lines, cur_line, cur_len = [], [], 0
    for word, wt in zip(words, weights):
        w_len = len(word) + 1   # +1 for space
        if cur_line and cur_len + w_len > chars_per_line:
            lines.append(cur_line)
            cur_line, cur_len = [(word, wt)], w_len
        else:
            cur_line.append((word, wt))
            cur_len += w_len
    if cur_line:
        lines.append(cur_line)

    n_lines = len(lines)
    # Each line gets equal vertical space; axes height is set by the caller
    y_start = 1.0 - 1.0 / (n_lines + 1) * 0.5

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    for li, line in enumerate(lines):
        y = 1.0 - (li + 0.7) / (n_lines + 0.5)
        x = 0.01
        for word, wt in line:
            r, g, b, _ = cmap(float(wt))
            txt_color = "black" if wt < 0.55 else "white"
            t = ax.text(
                x, y, word,
                ha="left", va="center", fontsize=fs,
                bbox=dict(boxstyle="round,pad=0.22", facecolor=(r, g, b),
                          edgecolor="none", alpha=0.88),
                color=txt_color, transform=ax.transAxes,
            )
            # Advance x by word width (approximate: chars * width_per_char)
            x += (len(word) + 1) / chars_per_line


def _n_lines(words, fig_width_inches=14):
    chars_per_line = int(fig_width_inches * 9.5)
    lines, cur = 1, 0
    for w in words:
        cur += len(w) + 1
        if cur > chars_per_line:
            lines += 1; cur = len(w) + 1
    return lines


def save_comparison_png(base_data, ft_data, query, out_path, cmap, model_label, layer_desc):
    """Base (top) vs fine-tuned (bottom) for one query, natural text flow."""
    W = 14
    words_b, weights_b, _ = base_data
    words_f, weights_f, _ = ft_data
    nl_b = _n_lines(words_b, W)
    nl_f = _n_lines(words_f, W)

    row_h = 0.38   # inches per text line
    pad   = 1.2    # title + spacing
    fig, axes = plt.subplots(
        2, 1,
        figsize=(W, nl_b * row_h + nl_f * row_h + pad * 2),
        gridspec_kw={"height_ratios": [nl_b, nl_f]},
    )
    fig.suptitle(
        f"{model_label}  ·  qid={query['question_id']}  "
        f"type={query['query_type']}  answer={query['answer']}\n"
        f"Color = answer-token attention ({layer_desc}, avg all heads, normalised)",
        fontsize=9, y=1.01,
    )
    render_highlighted_text_ax(axes[0], words_b, weights_b, cmap,
                                title="Zero-shot (base)", fig_width_inches=W)
    render_highlighted_text_ax(axes[1], words_f, weights_f, cmap,
                                title="Fine-tuned (LoRA)", fig_width_inches=W)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


def save_per_query_pngs(data_dict, out_dir, cmap, model_label, layer_desc):
    """One PNG per query — natural text flow, no tower."""
    W = 14
    out_dir.mkdir(parents=True, exist_ok=True)
    for qid, (words, weights, query) in sorted(data_dict.items()):
        nl = _n_lines(words, W)
        row_h = 0.38
        fig, ax = plt.subplots(figsize=(W, max(2.0, nl * row_h + 0.9)))
        fig.suptitle(
            f"{model_label}  ·  qid={qid}  type={query['query_type']}  answer={query['answer']}\n"
            f"Color = answer-token attention ({layer_desc}, avg all heads, normalised)",
            fontsize=9, y=1.0,
        )
        render_highlighted_text_ax(ax, words, weights, cmap,
                                   title="", fig_width_inches=W)
        plt.tight_layout()
        out_path = out_dir / f"qid{qid}_{query['query_type']}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",       type=str, default=None)
    ap.add_argument("--model",        type=str, default=None)
    ap.add_argument("--adapter",      type=str, default=None)
    ap.add_argument("--compare",      action="store_true",
                    help="Run base pass then finetuned pass and produce side-by-side PNGs.")
    ap.add_argument("--query_ids",    type=int, nargs="+", required=True)
    ap.add_argument("--queries_json", type=str, default="data/queries_easy.json")
    ap.add_argument("--out_dir",      type=str, default="finetune/attention_plots")
    ap.add_argument("--max_tokens",   type=int, default=512)
    ap.add_argument("--layers",       type=int, nargs="+", default=None,
                    help="Layer indices to use (default: penultimate layer only). "
                         "E.g. --layers 30 31 for last two layers.")
    ap.add_argument("--dtype",        type=str, default="bf16", choices=list(DTYPE_MAP))
    ap.add_argument("--cmap",         type=str, default="YlOrRd",
                    help="Matplotlib colormap name for highlights.")
    args = ap.parse_args()

    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text())
        model_id     = cfg["model_id"]
        trust_remote = cfg.get("trust_remote_code", False)
    elif args.model:
        model_id, trust_remote = args.model, False
    else:
        ap.error("Provide --config or --model.")

    dtype  = DTYPE_MAP[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cmap   = plt.get_cmap(args.cmap)

    model_slug = model_id.replace("/", "_")
    out_dir    = Path(args.out_dir)
    layers     = args.layers   # None → penultimate

    queries = load_queries_by_id(args.queries_json, args.query_ids)

    def run_pass(adapter_path, label, slug_suffix=""):
        slug = model_slug + slug_suffix
        print(f"\n[load] {label} ({'with adapter' if adapter_path else 'base'}) …")
        tok, model = load_model(model_id, dtype, device, trust_remote,
                                adapter=adapter_path)

        # Build layer description for plot titles
        n_layers = model.config.num_hidden_layers
        if layers is None:
            layer_desc = f"penultimate layer (L{n_layers-2})"
        elif len(layers) == 1:
            layer_desc = f"layer {layers[0]}"
        else:
            layer_desc = f"layers {layers}"

        data = {}
        for qid in args.query_ids:
            q = queries[qid]
            print(f"  qid={qid} {q['query_type']} …", end=" ", flush=True)
            words, weights = get_word_attention(
                model, tok, q, model_id, args.max_tokens, layers=layers)
            data[qid] = (words, weights, q)
            print("ok")

        # HTML (all queries in one browsable file)
        save_html(data, label, out_dir / slug / "highlights.html", cmap,
                  layer_desc=layer_desc)
        # Individual PNGs per query
        save_per_query_pngs(data, out_dir / slug / "per_query", cmap, label, layer_desc)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        return data, layer_desc

    base_data, layer_desc = run_pass(None, model_id, slug_suffix="")

    ft_data = None
    if args.compare and args.adapter:
        ft_data, _ = run_pass(args.adapter, f"{model_id} (fine-tuned)",
                              slug_suffix="_finetuned")

        # Side-by-side comparison PNG per query
        comp_dir = out_dir / f"{model_slug}_comparison"
        print(f"\n[compare] writing side-by-side PNGs → {comp_dir}/")
        for qid in args.query_ids:
            q = queries[qid]
            save_comparison_png(
                base_data[qid], ft_data[qid], q,
                out_path=comp_dir / f"qid{qid}_{q['query_type']}.png",
                cmap=cmap,
                model_label=model_id,
                layer_desc=layer_desc,
            )

    print(f"\n[done] plots in {out_dir}/")


if __name__ == "__main__":
    main()

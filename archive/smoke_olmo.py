import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="Say 'loaded' and then give me a 5-word fun fact about graphs.")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"[info] torch cuda available: {torch.cuda.is_available()}")
    if device == "cuda":
        print(f"[info] gpu: {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info()
        print(f"[info] cuda mem free/total (GB): {free/1e9:.2f}/{total/1e9:.2f}")

    print(f"[info] loading tokenizer: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"[info] loading model: {args.model} (dtype={dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    # Chat-template if available (important for instruct models)
    if hasattr(tok, "apply_chat_template"):
        messages = [{"role": "user", "content": args.prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(model.device)
    else:
        enc = tok(args.prompt, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=(args.temperature > 0),
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("[info] generating…")
    with torch.no_grad():
        out = model.generate(**enc, **gen_kwargs)

    decoded = tok.decode(out[0], skip_special_tokens=True)
    print("\n===== OUTPUT =====")
    print(decoded)
    print("==================\n")


if __name__ == "__main__":
    main()
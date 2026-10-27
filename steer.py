"""Run Qwen3.5-4B under each steering condition and log every reply.

lyra's recipe, as implemented in sarvkk/two-neurons: at every token, overwrite
the input to the MLP down projection for neuron n in layer L with

    s = dose * median(||residual leaving layer L||, tokens 1:) / ||W_down[:, n]||

Nothing in the weights or the prompt changes. Remove the hook and the model is
back to normal.

    uv run steer.py                      # full run, ~6 min on an M4 Max
    uv run steer.py --limit 1 --sums 2   # quick smoke test
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import CONDITIONS, MATH, PROMPTS, RANDOM_PAIR

MODEL = "Qwen/Qwen3.5-4B"


def pick_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/runs.jsonl")
    ap.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--limit", type=int, default=len(PROMPTS), help="number of prompts to run")
    ap.add_argument("--sums", type=int, default=len(MATH), help="number of addition sums to score")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    args = ap.parse_args()

    dev = pick_device(args.device)
    conditions = {c: CONDITIONS[c] for c in args.conditions}

    tok = AutoTokenizer.from_pretrained(MODEL)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
    print(f"[load] {MODEL} on {dev} in {time.time() - t0:.1f}s", flush=True)

    # Find each text layer and its MLP down projection (skip any vision tower).
    down, layers = {}, {}
    for name, mod in model.named_modules():
        m = re.search(r"layers\.(\d+)\.mlp\.down_proj$", name)
        if m and "visual" not in name:
            down[int(m.group(1))] = mod
            layers[int(m.group(1))] = model.get_submodule(name.rsplit(".mlp", 1)[0])

    colnorm = {(L, n): float(down[L].weight[:, n].float().norm()) for specs in conditions.values() for L, n, _ in specs}
    steered_layers = sorted({L for L, _ in colnorm})

    # The steering hook: pin one neuron's value going into the down projection.
    active = {}  # (layer, neuron) -> value, set per generation

    def pre_hook(L):
        def hook(module, hook_args):
            x = hook_args[0]
            for (LL, n), value in active.items():
                if LL == L:
                    x[..., n] = value
            return (x,)

        return hook

    for L, mod in down.items():
        mod.register_forward_pre_hook(pre_hook(L))

    def render(text):
        ids = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
            return_tensors="pt",
        )
        ids = ids if torch.is_tensor(ids) else ids["input_ids"]
        return ids.to(dev)

    def residual_medians(ids):
        """Median residual-stream norm leaving each steered layer, skipping token 0."""
        got, handles = {}, []
        for L in steered_layers:

            def cap(_m, _a, out, L=L):
                got[L] = out[0] if isinstance(out, tuple) else out

            handles.append(layers[L].register_forward_hook(cap))
        with torch.inference_mode():
            model(input_ids=ids, use_cache=False)
        for h in handles:
            h.remove()
        return {L: float(h[0, 1:, :].float().norm(dim=-1).median()) for L, h in got.items()}

    def generate(ids, cond, meds, max_new):
        active.clear()
        for L, n, dose in conditions[cond]:
            active[(L, n)] = dose * meds[L] / colnorm[(L, n)]
        with torch.inference_mode():
            out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new, do_sample=False)
        active.clear()
        return tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True).strip()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    score = {c: 0 for c in conditions}
    with out_path.open("w") as log:
        for cat, prompt in PROMPTS[: args.limit]:
            ids = render(prompt)
            meds = residual_medians(ids) if steered_layers else {}
            print(f"\n### [{cat}] {prompt}", flush=True)
            for cond in conditions:
                reply = generate(ids, cond, meds, args.max_new_tokens)
                log.write(
                    json.dumps(
                        {"kind": "prompt", "category": cat, "prompt": prompt, "condition": cond, "response": reply}
                    )
                    + "\n"
                )
                log.flush()
                print(f"--- {cond}: {reply[:200]!r}", flush=True)

        for prompt, answer in MATH[: args.sums]:
            ids = render(prompt)
            meds = residual_medians(ids) if steered_layers else {}
            for cond in conditions:
                reply = generate(ids, cond, meds, 12)
                ok = str(answer) in re.findall(r"\d+", reply)
                score[cond] += ok
                log.write(
                    json.dumps(
                        {
                            "kind": "math",
                            "prompt": prompt,
                            "answer": answer,
                            "condition": cond,
                            "response": reply,
                            "correct": ok,
                        }
                    )
                    + "\n"
                )

    print(f"\n[sums correct /{args.sums}] {score}")
    print(f"[random control] {RANDOM_PAIR}")
    print(f"[written] {out_path}")


if __name__ == "__main__":
    main()

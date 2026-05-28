"""Generate from a TurboQuant MoE model with experts streamed from disk.

Runs a model whose weights exceed available RAM by keeping only the resident
tensors in memory and streaming router-selected experts on demand.

Example (Qwen3.6-35B-A3B, ~16 GB on disk, runs in ~5 GB RAM):

    python -m turboquant_mlx.stream.stream_generate \\
        --model manjunathshiva/Qwen3.6-35B-A3B-tq3-g32 \\
        --prompt "Explain why the sky is blue." \\
        --max-tokens 256 --cache-budget-gb 3

    # save output to a markdown file
    python -m turboquant_mlx.stream.stream_generate \\
        --model manjunathshiva/Qwen3.6-35B-A3B-tq3-g32 \\
        --prompt "Explain why the sky is blue." \\
        --max-tokens 256 --cache-budget-gb 3 \\
        --output-dir ./risposte
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import mlx.core as mx

import turboquant_mlx.compat  # noqa: F401
from mlx_lm import generate as mlx_generate
from mlx_lm.sample_utils import make_sampler

from .loader import load_streaming


def _rss_gb() -> float:
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])
    return int(out) / 1024 / 1024


def _slug(text: str, max_words: int = 6) -> str:
    words = re.sub(r"[^\w\s]", "", text.lower()).split()
    return "_".join(words[:max_words])


def _split_think(text: str) -> tuple[str, str]:
    """Split <think>...</think> trace from the final answer."""
    m = re.match(r"\s*<think>(.*?)</think>\s*(.*)", text, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", text.strip()


def _save_markdown(
    output_dir: str,
    prompt: str,
    text: str,
    model: str,
    args: argparse.Namespace,
    stats: dict,
    gen_tok_s: float,
    e2e_tok_s: float,
    peak_rss_gb: float,
    peak_mlx_gb: float,
    n_tokens: int,
    gen_time_s: float,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _slug(prompt)
    filename = out / f"{ts}_{slug}.md"

    think, answer = _split_think(text)

    lines = [
        f"# {prompt}",
        "",
        f"> **Modello:** `{model}`  ",
        f"> **Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"> **Parametri:** cache `{args.cache_budget_gb} GB` · workers `{args.prefetch_workers}` · temp `{args.temp}` · max-tokens `{args.max_tokens}`{' · fast' if args.fast else ''}",
        "",
    ]

    if think:
        lines += [
            "<details>",
            "<summary>Chain of thought</summary>",
            "",
            think,
            "",
            "</details>",
            "",
        ]

    lines += [
        "## Risposta",
        "",
        answer if answer else text.strip(),
        "",
        "---",
        "",
        "## Stats",
        "",
        f"| Metrica | Valore |",
        f"|---|---|",
        f"| Decode speed | {gen_tok_s:.1f} tok/s |",
        f"| End-to-end speed | {e2e_tok_s:.1f} tok/s |",
        f"| Token generati | {n_tokens} in {gen_time_s:.1f}s |",
        f"| Peak RSS | {peak_rss_gb:.2f} GB |",
        f"| Peak MLX | {peak_mlx_gb:.2f} GB |",
        f"| Expert hit-rate | {stats['hit_rate']:.1%} |",
        f"| Cache residente | {stats['resident_gb']:.2f} GB |",
        f"| Disco letto | {stats['bytes_read_gb']:.1f} GB |",
    ]

    filename.write_text("\n".join(lines), encoding="utf-8")
    return filename


def main():
    p = argparse.ArgumentParser(
        description="Stream-generate from a TurboQuant MoE model (experts paged from disk)."
    )
    p.add_argument("--model", required=True, help="Local path or HF repo id of a TurboQuant model.")
    p.add_argument("--prompt", default="Why is the sky blue?")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--cache-budget-gb", type=float, default=3.0,
                   help="Max resident expert memory (LRU-evicted). Lower = less RAM, more disk reads.")
    p.add_argument("--prefetch-workers", type=int, default=8,
                   help="Threads for parallel per-layer expert reads. 1 = serial baseline.")
    p.add_argument("--fast", action="store_true", help="Disable QJL correction for faster decode.")
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--output-dir", default=None,
                   help="Directory where the response is saved as a markdown file.")
    args = p.parse_args()

    t0 = time.time()
    model, tok, cache = load_streaming(
        args.model, cache_budget_gb=args.cache_budget_gb, fast=args.fast,
        prefetch_workers=args.prefetch_workers,
    )
    print(f"[stream] loaded in {time.time() - t0:.1f}s | resident RSS={_rss_gb():.2f} GB")

    prompt = args.prompt
    if not args.no_chat_template and hasattr(tok, "apply_chat_template"):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}], add_generation_prompt=True
        )

    sampler = make_sampler(temp=args.temp)
    print("=" * 60)
    t = time.time()
    text = mlx_generate(model, tok, prompt=prompt, max_tokens=args.max_tokens,
                        sampler=sampler, verbose=True)
    dt = time.time() - t
    print("=" * 60)
    # Count tokens actually generated (the model may stop at EOS before
    # max_tokens) — dividing max_tokens by wall-time overstates the rate.
    n = len(tok.encode(text))
    peak_rss = _rss_gb()
    peak_mlx = mx.get_peak_memory() / 1e9
    gen_tok_s = n / dt
    e2e_tok_s = n / dt
    print(f"[stream] {n} generated tok in {dt:.1f}s = {gen_tok_s:.1f} tok/s (end-to-end) | "
          f"peak RSS={peak_rss:.2f} GB | mlx_peak={peak_mlx:.2f} GB")
    s = cache.stats()
    print(f"[stream] expert cache: hit_rate={s['hit_rate']:.1%} resident={s['resident_gb']:.2f} GB "
          f"disk_read={s['bytes_read_gb']:.1f} GB")

    if args.output_dir:
        saved = _save_markdown(
            output_dir=args.output_dir,
            prompt=args.prompt,
            text=text,
            model=args.model,
            args=args,
            stats=s,
            gen_tok_s=gen_tok_s,
            e2e_tok_s=e2e_tok_s,
            peak_rss_gb=peak_rss,
            peak_mlx_gb=peak_mlx,
            n_tokens=n,
            gen_time_s=dt,
        )
        print(f"[stream] risposta salvata in {saved}")


if __name__ == "__main__":
    main()

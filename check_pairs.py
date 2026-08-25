"""Preflight for a Binoculars scoring pair: tokenizers, memory, availability.

Run this BEFORE ``score_parquets.py --pair ...``. Everything it checks is
something that otherwise only fails after a multi-gigabyte download, or -- worse
-- silently produces numbers that cannot be compared:

1. **Tokenizer identity.** Binoculars encodes the text once, with the observer's
   tokenizer, and feeds those ids to both models. ``binoculars.utils`` therefore
   refuses a pair whose vocabularies differ. Some base/instruct pairs differ only
   in chat/tool control tokens appended at the end, which ordinary prose never
   encodes to; this script says which kind of difference a pair has, so the
   choice between "pick another pair" and ``--allow-token-mismatch`` is made on
   evidence. Only tokenizers are downloaded here -- a few megabytes.

2. **Memory.** Both models are resident on ONE device, and the cross-perplexity
   term materialises several ``batch x tokens x vocab`` tensors for each of
   them. Vocabulary width therefore matters as much as parameter count: the
   Falcon baseline's 65k vocabulary is unusually narrow, and a modern 128k-152k
   pair costs 2-2.5x as much per row at the same batch size. The table prints
   the estimate against the GPU actually present and suggests a batch size.

3. **Availability.** Gated repos (Llama, Gemma) need ``HF_TOKEN`` in the parent
   ``.env`` and an accepted licence; whether the weights are already in the Hub
   cache decides whether a run starts in seconds or in an hour.

Usage (from this directory, external/Binoculars):
    # every pair in the registry, tokenizers only
    uv run python check_pairs.py

    # just the shortlist, sized for the GPU you will actually use
    uv run python check_pairs.py --pairs qwen25-7b falcon3-7b llama31-8b --gpu 0

    # skip the network entirely: table and memory arithmetic only
    uv run python check_pairs.py --no-tokenizer-check

From the parent repo (episteme-ai):
    uv run --project external/Binoculars python external/Binoculars/check_pairs.py \
        --pairs qwen25-1_5b qwen25-7b falcon3-7b llama31-8b --gpu 0

Exit status: 0 if every checked pair is usable (identical or merely
extended tokenizers), 1 if any pair is unusable -- a re-mapped token, a repo
that cannot be reached, or a pair that does not fit the device at batch 1.
"""

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", nargs="*", default=None,
                    help="Pair keys to check (default: all of them)")
    ap.add_argument("--gpu", default=None,
                    help="value for CUDA_VISIBLE_DEVICES, so the memory column is "
                         "sized against the card the run will actually use")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Batch size to size the estimate for (default: %(default)s)")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="Sequence length to size the estimate for (default: the "
                         "detector's own truncation cap, 512)")
    ap.add_argument("--fp32-metrics", action="store_true",
                    help="Size the estimate for --fp32-metrics scoring, which keeps "
                         "an fp32 copy of the logits alongside the bf16 ones")
    ap.add_argument("--device-gib", type=float, default=None,
                    help="Assume a card of this size instead of asking torch. Use "
                         "when planning from a machine with no GPU (40 or 80).")
    ap.add_argument("--no-tokenizer-check", action="store_true",
                    help="Do not touch the network: print the table and the memory "
                         "arithmetic only")
    return ap.parse_args()


args = parse_args()

# Same reason as in score_parquets.py: CUDA_VISIBLE_DEVICES is read when the
# driver initialises, which happens inside torch.
if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from dotenv import load_dotenv  # noqa: E402

import model_pairs  # noqa: E402
from model_pairs import DEFAULT_MAX_TOKENS, PAIRS  # noqa: E402

load_dotenv()

MAX_TOKENS = args.max_tokens if args.max_tokens is not None else DEFAULT_MAX_TOKENS


def device_size_gib() -> tuple[float | None, str]:
    """(usable GiB, description) for the device this run would land on."""
    if args.device_gib is not None:
        return args.device_gib, f"assumed {args.device_gib:.0f} GiB (--device-gib)"
    try:
        import torch
    except ImportError:
        return None, "torch not importable"
    if not torch.cuda.is_available():
        return None, "no CUDA device visible"
    props = torch.cuda.get_device_properties(0)
    return props.total_memory / 1024 ** 3, f"{props.name} ({props.total_memory / 1024 ** 3:.0f} GiB)"


def hub_params(repo_id: str) -> float | None:
    """Parameter count in billions from the Hub's safetensors index, if reachable."""
    try:
        from huggingface_hub import model_info
        info = model_info(repo_id, files_metadata=False)
        total = getattr(getattr(info, "safetensors", None), "total", None)
        return total / 1e9 if total else None
    except Exception:
        return None


def cache_state(repo_id: str) -> str:
    """Whether the weights are already in the local Hub cache."""
    try:
        from huggingface_hub import scan_cache_dir
        for repo in scan_cache_dir().repos:
            if repo.repo_id == repo_id and repo.repo_type == "model":
                return f"cached {repo.size_on_disk / 1024 ** 3:.1f} GiB"
    except Exception:
        return "cache unknown"
    return "not cached"


def check_tokenizers(pair) -> tuple[str, str]:
    """(verdict, detail) for one pair's tokenizer compatibility."""
    from binoculars.utils import describe_tokenizer_diff, tokenizer_diff
    try:
        diff = tokenizer_diff(pair.observer, pair.performer)
    except Exception as exc:
        return "ERROR", f"{type(exc).__name__}: {exc}"

    if diff["identical"]:
        return "identical", f"{diff['size_1']} entries, byte-for-byte equal"
    if diff["compatible"]:
        extra = len(diff["only_in_1"]) + len(diff["only_in_2"])
        return "extended", (
            f"{extra} added token(s), no shared token re-mapped -- usable with "
            "--allow-token-mismatch\n"
            + describe_tokenizer_diff(pair.observer, pair.performer, diff)
        )
    return "INCOMPATIBLE", describe_tokenizer_diff(pair.observer, pair.performer, diff)


def main() -> int:
    keys = args.pairs or list(PAIRS)
    pairs = [model_pairs.resolve(k) for k in keys]

    device_gib, device_desc = device_size_gib()
    print()
    print("=" * 100)
    print("Binoculars scoring pairs")
    print(f"  device        : {device_desc}")
    print(f"  sized for     : batch {args.batch_size}, {MAX_TOKENS} tokens, "
          f"{'fp32' if args.fp32_metrics else 'bf16'} metrics")
    print(f"  HF_TOKEN      : {'set' if os.environ.get('HF_TOKEN') else 'NOT SET'}")
    print("=" * 100)
    header = (f"{'pair':<13}{'year':<6}{'params':>8}{'vocab':>9}{'weights':>9}"
              f"{'peak':>8}{'batch':>7}  {'column':<32}")
    print(header)
    print("-" * 100)

    problems: list[str] = []
    for pair in pairs:
        weights = pair.weights_gib
        peak = pair.total_gib(args.batch_size, MAX_TOKENS, args.fp32_metrics)
        if device_gib is None:
            batch = "?"
        else:
            best = pair.max_batch_size(device_gib, MAX_TOKENS, args.fp32_metrics)
            batch = str(best) if best else "NONE"
            if not best:
                problems.append(f"{pair.key}: does not fit {device_gib:.0f} GiB even at batch 1")
        print(f"{pair.key:<13}{pair.year:<6}{pair.params:>7.2f}B{pair.vocab:>9}"
              f"{weights:>8.1f}G{peak:>7.1f}G{batch:>7}  {pair.column:<32}")
    print("-" * 100)
    print("  weights = both models, bf16.  peak = weights + the logits/softmax working")
    print("  set at the batch size above.  batch = largest that fits with ~2 GiB slack.")
    print()

    for pair in pairs:
        flags = []
        if pair.calibrated:
            flags.append("thresholds calibrated")
        else:
            flags.append("thresholds NOT calibrated")
        if pair.gated:
            flags.append("gated repo")
        print(f"  {pair.key:<13} {pair.observer}  +  {pair.performer}")
        print(f"  {'':<13} {'; '.join(flags)}")
        if pair.note:
            print(f"  {'':<13} {pair.note}")
        print()

    if args.no_tokenizer_check:
        print("Tokenizer check skipped (--no-tokenizer-check).")
        return 1 if problems else 0

    print("=" * 100)
    print("Tokenizer consistency (downloads tokenizers only, no weights)")
    print("=" * 100)
    for pair in pairs:
        verdict, detail = check_tokenizers(pair)
        mark = {"identical": "✅", "extended": "⚠️ ", "INCOMPATIBLE": "❌", "ERROR": "❌"}[verdict]
        print(f"{mark} {pair.key:<13} {verdict}")
        for line in detail.splitlines():
            print(f"     {line}")
        print(f"     cache: {cache_state(pair.observer)} / {cache_state(pair.performer)}")
        hub = hub_params(pair.observer)
        if hub is not None and abs(hub - pair.params) > 0.15:
            print(f"     NOTE  the Hub reports {hub:.2f}B parameters, the registry "
                  f"assumes {pair.params:.2f}B -- memory estimate is off")
        if verdict in ("INCOMPATIBLE", "ERROR"):
            problems.append(f"{pair.key}: {verdict.lower()} tokenizers")
        print()

    if problems:
        print("Not usable as they stand:")
        for problem in problems:
            print(f"  ❌ {problem}")
        print()
        return 1

    print("✅ Every pair checked is usable.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

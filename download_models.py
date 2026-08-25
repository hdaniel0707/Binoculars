"""Pre-fetch the weights for one or more scoring pairs from the registry.

Scoring downloads what it needs on its own, but it does it *after* the
confirmation prompt, inside ``Binoculars(...)`` -- so a first run with a new pair
sits there for half an hour with no progress on the corpus and no obvious reason
why. Fetching first separates "the weights are not here yet" from "the scoring is
slow", and lets the download run while something else has the GPU.

Usage (from this directory, external/Binoculars):
    uv run python download_models.py --pairs qwen25-7b falcon3-7b
    uv run python download_models.py --pairs qwen25-1_5b      # ~6 GiB, the smoke test
    uv run python download_models.py --all                    # everything in the registry

Gated repos (Llama, Gemma) need HF_TOKEN in the parent .env and an accepted
licence on the Hub; ``check_pairs.py`` reports which pairs those are before
anything is fetched.
"""

import argparse

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

import model_pairs
from model_pairs import PAIRS


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", nargs="+", default=None,
                    help="Pair keys from model_pairs.py, e.g. qwen25-7b falcon3-7b")
    ap.add_argument("--all", action="store_true",
                    help="Fetch every pair in the registry (tens of gigabytes)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()  # HF_TOKEN, for the gated repos

    if args.all:
        keys = list(PAIRS)
    elif args.pairs:
        keys = args.pairs
    else:
        raise SystemExit("Nothing to do: pass --pairs KEY [KEY ...] or --all. "
                         f"Known pairs: {', '.join(PAIRS)}")

    pairs = [model_pairs.resolve(key) for key in keys]
    print(f"Fetching {len(pairs)} pair(s): {', '.join(p.key for p in pairs)}")
    for pair in pairs:
        for model_id in (pair.observer, pair.performer):
            print(f"Downloading {model_id} ...")
            snapshot_download(repo_id=model_id)
            print(f"Done: {model_id}")

    print("All models downloaded.")


if __name__ == "__main__":
    main()

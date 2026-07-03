"""Score the text column of one or more parquet files with Binoculars.

For each input parquet file, every row's text is scored with Binoculars and the
result is written to a new column. The scored dataframe is saved next to the
original file as "<original_name>_binox0.parquet".

Usage:
    uv run python score_parquets.py data/wp.parquet --observer gpt2 --performer gpt2 --limit 10
    uv run python score_parquets.py data/*.parquet
    uv run python score_parquets.py data/ --text-column text --batch-size 16
    ----
    uv run python score_parquets.py data/wp.parquet --cpu-threads 16 --gpu 0
    uv run python score_parquets.py data/essay.parquet --cpu-threads 16 --gpu 1
    uv run python score_parquets.py data/reuter.parquet --cpu-threads 16 --gpu 6
    uv run python score_parquets.py data/wp2.parquet --cpu-threads 16 --gpu 0
    uv run python score_parquets.py data/essay2.parquet --cpu-threads 16 --gpu 1
    uv run python score_parquets.py data/reuter2.parquet --cpu-threads 16 --gpu 6
"""

import argparse
import math
import os
from pathlib import Path

_arg_parser = argparse.ArgumentParser(description=__doc__)
_arg_parser.add_argument("inputs", nargs="+", help="Parquet file(s) and/or directories containing parquet files")
_arg_parser.add_argument("--text-column", default="text", help="Name of the column containing text to score")
_arg_parser.add_argument("--batch-size", type=int, default=1, help="Number of texts scored per batch")
_arg_parser.add_argument("--limit", type=int, default=0,
                          help="Only process the first N rows per file, for debugging (0 = no limit)")
_arg_parser.add_argument("--observer", default="tiiuae/falcon-7b", help="Observer model name or path")
_arg_parser.add_argument("--performer", default="tiiuae/falcon-7b-instruct", help="Performer model name or path")
_arg_parser.add_argument("--gpu", default=None,
                          help="value for CUDA_VISIBLE_DEVICES (which GPU(s) to use, e.g. '0' or '0,1'). "
                               "Default: leave unset so all visible GPUs are used")
_arg_parser.add_argument("--cpu-threads", type=int, default=None,
                          help="limit CPU threads used for inference (sets OMP_NUM_THREADS / MKL_NUM_THREADS). "
                               "Default: leave unset")
args = _arg_parser.parse_args()

# GPU/CPU constraints must be set before torch (and anything importing it) is loaded.
if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
if args.cpu_threads is not None:
    os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)

from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

from binoculars import Binoculars
from binoculars.cuda_util import check_cuda
from binoculars.env_utils import doublecheck_env, doublecheck_pkgs

load_dotenv()
doublecheck_env(".env")
doublecheck_pkgs(pyproject_path="pyproject.toml", verbose=True)
check_cuda()  # informational only; scoring still works on CPU, just slower


def collect_paths(inputs: list[str]) -> list[Path]:
    paths = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.parquet")))
        else:
            paths.append(p)
    # never re-score our own output files
    return [p for p in paths if not p.name.endswith("_binox0.parquet")]


def score_file(bino: Binoculars, path: Path, text_column: str, batch_size: int, limit: int) -> tuple[Path, int, int]:
    df = pd.read_parquet(path)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in {path}. Available columns: {list(df.columns)}")

    if limit > 0:
        df = df.head(limit)

    texts = df[text_column].fillna("").astype(str).tolist()

    # Binoculars/transformers can't handle a 0-token input, and there's nothing meaningful
    # to score in an empty/blank text anyway, so leave those rows as NaN.
    scoreable = [(i, t) for i, t in enumerate(texts) if t.strip()]
    skipped = len(texts) - len(scoreable)
    if skipped:
        print(f"Skipping {skipped} empty/blank row(s) in {path.name} (score set to NaN)")

    scores = [float("nan")] * len(texts)
    for i in tqdm(range(0, len(scoreable), batch_size), desc=path.name, unit="batch"):
        chunk = scoreable[i:i + batch_size]
        idxs, batch_texts = zip(*chunk)
        batch_scores = bino.compute_score(list(batch_texts))
        for idx, score in zip(idxs, batch_scores):
            scores[idx] = score

    df["binoculars_score"] = scores

    n_null = sum(1 for s in scores if math.isnan(s))
    n_good = len(scores) - n_null

    out_path = path.parent / f"{path.stem}_binox0.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} scored rows to {out_path} ({n_good} good, {n_null} null/skipped)")
    return out_path, n_good, n_null


def main():
    paths = collect_paths(args.inputs)
    if not paths:
        raise SystemExit("No parquet files found to score.")

    print(f"Found {len(paths)} file(s) to score:")
    for p in paths:
        print(f"  - {p}")

    bino = Binoculars(
        observer_name_or_path=args.observer,
        performer_name_or_path=args.performer,
    )

    total_good = 0
    total_null = 0
    for path in paths:
        print(f"\nScoring {path} ...")
        _, n_good, n_null = score_file(bino, path, args.text_column, args.batch_size, args.limit)
        total_good += n_good
        total_null += n_null

    total_rows = total_good + total_null
    print(
        f"\nSummary: {len(paths)} file(s), {total_rows} row(s) total "
        f"-> {total_good} good, {total_null} null/skipped"
    )


if __name__ == "__main__":
    main()

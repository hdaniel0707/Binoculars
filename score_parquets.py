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
    ----
    uv run python score_parquets.py data/essay_gpt54mini_binox0.parquet --cpu-threads 16 --gpu 0 --only-missing
    uv run python score_parquets.py data/reuter_gpt54mini_binox0.parquet --cpu-threads 16 --gpu 0 --only-missing
    uv run python score_parquets.py data/wp_gpt54mini_binox0.parquet --cpu-threads 16 --gpu 0 --only-missing
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
_arg_parser.add_argument("--only-missing", action="store_true",
                          help="Only compute scores for rows whose 'binoculars_score' is missing (NaN). "
                               "Requires the column to already exist in the parquet file(s). Reports how "
                               "many rows are missing, lists them, and asks for confirmation before scoring.")
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


def collect_paths(inputs: list[str], allow_output_files: bool = False) -> list[Path]:
    paths = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.parquet")))
        else:
            paths.append(p)
    if allow_output_files:
        return paths
    # never re-score our own output files as if they were fresh input
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

def find_missing_rows(path: Path, text_column: str) -> tuple[list[int], list[int]]:
    """Split rows in `path` with a missing (NaN) binoculars_score into two groups.

    Returns (scoreable_idx, blank_idx):
      - scoreable_idx: missing score, non-blank text -> can be scored
      - blank_idx: missing score, blank/empty text -> would be skipped anyway, stays NaN
    """
    df = pd.read_parquet(path)
    if "binoculars_score" not in df.columns:
        raise ValueError(
            f"'binoculars_score' column not found in {path}. "
            "Score the file without --only-missing first to create it."
        )
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in {path}. Available columns: {list(df.columns)}")

    missing_idx = df.index[df["binoculars_score"].isna()]
    texts = df[text_column].fillna("").astype(str)
    scoreable_idx = [i for i in missing_idx if texts[i].strip()]
    blank_idx = [i for i in missing_idx if not texts[i].strip()]
    return scoreable_idx, blank_idx


def score_missing_in_file(bino: Binoculars, path: Path, text_column: str, batch_size: int,
                           scoreable_idx: list[int]) -> int:
    """Score exactly the given rows in `path`, overwriting the file in place."""
    df = pd.read_parquet(path)
    texts = df[text_column].fillna("").astype(str)

    for i in tqdm(range(0, len(scoreable_idx), batch_size), desc=path.name, unit="batch"):
        chunk_idx = scoreable_idx[i:i + batch_size]
        batch_texts = [texts[j] for j in chunk_idx]
        batch_scores = bino.compute_score(batch_texts)
        for idx, score in zip(chunk_idx, batch_scores):
            df.loc[idx, "binoculars_score"] = score

    df.to_parquet(path, index=False)
    print(f"Updated {path} ({len(scoreable_idx)} scored)")
    return len(scoreable_idx)


def _preview(idxs: list[int], cap: int = 20) -> list:
    return idxs if len(idxs) <= cap else idxs[:cap] + [f"... (+{len(idxs) - cap} more)"]


def run_only_missing(paths: list[Path]):
    per_file: dict[Path, tuple[list[int], list[int]]] = {}
    total_scoreable = 0
    total_blank = 0
    for path in paths:
        scoreable_idx, blank_idx = find_missing_rows(path, args.text_column)
        if scoreable_idx or blank_idx:
            per_file[path] = (scoreable_idx, blank_idx)
            total_scoreable += len(scoreable_idx)
            total_blank += len(blank_idx)

    if total_scoreable == 0 and total_blank == 0:
        print("No missing 'binoculars_score' entries found. Nothing to do.")
        return

    print(
        f"\nFound {total_scoreable + total_blank} row(s) with a missing binoculars_score "
        f"across {len(per_file)} file(s):"
    )
    for path, (scoreable_idx, blank_idx) in per_file.items():
        print(f"  - {path}: {len(scoreable_idx)} scoreable, {len(blank_idx)} with blank/empty text (stays null)")
        if scoreable_idx:
            print(f"      scoreable indices: {_preview(scoreable_idx)}")
        if blank_idx:
            print(f"      blank-text indices (will be skipped): {_preview(blank_idx)}")

    if total_scoreable == 0:
        print("\nAll missing rows have blank/empty text and can't be scored. Nothing to do.")
        return

    answer = input(
        f"\nProceed with scoring {total_scoreable} row(s) with missing scores? "
        f"({total_blank} row(s) with blank text will be left null) [y/N]: "
    ).strip().lower()
    if answer not in ("y", "yes"):
        print("Aborted.")
        return

    bino = Binoculars(
        observer_name_or_path=args.observer,
        performer_name_or_path=args.performer,
    )

    total_scored = 0
    for path, (scoreable_idx, _) in per_file.items():
        if not scoreable_idx:
            continue
        total_scored += score_missing_in_file(bino, path, args.text_column, args.batch_size, scoreable_idx)

    print(f"\nSummary: scored {total_scored} row(s). {total_blank} row(s) left null (blank text).")


def debug_scoreable_stat(path: Path, text_column: str,limit: int):
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
        print(f"❌ Skipping {skipped} empty/blank row(s) in {path.name} (score set to NaN)")
    else:
        print("✅ Data is good!")

def main():
    paths = collect_paths(args.inputs, allow_output_files=args.only_missing)
    if not paths:
        raise SystemExit("No parquet files found to score.")

    print(f"Found {len(paths)} file(s) to score:")
    for p in paths:
        print(f"  - {p}")

    if args.only_missing:
        run_only_missing(paths)
        return

    for path in paths:
        debug_scoreable_stat(path, args.text_column,args.limit)

    assert False

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

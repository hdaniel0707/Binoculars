"""Score the text column of one or more parquet files with Binoculars.

For each input parquet file, every row's text is scored with Binoculars and the
result is written to a new column. The scored dataframe is saved next to the
original file as "<original_name>_binox0.parquet".

Usage:
    uv run python score_parquets.py data/wp.parquet --observer gpt2 --performer gpt2 --limit 10
    uv run python score_parquets.py data/*.parquet
    uv run python score_parquets.py data/ --text-column text --batch-size 16
    uv run python score_parquets.py data/wp.parquet
    uv run python score_parquets.py data/essay.parquet
    uv run python score_parquets.py data/reuter.parquet
    uv run python score_parquets.py data/wp2.parquet
    uv run python score_parquets.py data/essay2.parquet
    uv run python score_parquets.py data/reuter2.parquet
"""

import argparse
from pathlib import Path

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


def score_file(bino: Binoculars, path: Path, text_column: str, batch_size: int, limit: int) -> Path:
    df = pd.read_parquet(path)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in {path}. Available columns: {list(df.columns)}")

    if limit > 0:
        df = df.head(limit)

    texts = df[text_column].fillna("").astype(str).tolist()

    scores = []
    for i in tqdm(range(0, len(texts), batch_size), desc=path.name, unit="batch"):
        batch = texts[i:i + batch_size]
        scores.extend(bino.compute_score(batch))

    df["binoculars_score"] = scores

    out_path = path.parent / f"{path.stem}_binox0.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} scored rows to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Score parquet files with Binoculars.")
    parser.add_argument("inputs", nargs="+", help="Parquet file(s) and/or directories containing parquet files")
    parser.add_argument("--text-column", default="text", help="Name of the column containing text to score")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of texts scored per batch")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only process the first N rows per file, for debugging (0 = no limit)")
    parser.add_argument("--observer", default="tiiuae/falcon-7b", help="Observer model name or path")
    parser.add_argument("--performer", default="tiiuae/falcon-7b-instruct", help="Performer model name or path")
    args = parser.parse_args()

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

    for path in paths:
        print(f"\nScoring {path} ...")
        score_file(bino, path, args.text_column, args.batch_size, args.limit)


if __name__ == "__main__":
    main()

"""Score the text column of one or more parquet files with Binoculars.

Reads each Parquet, works out which rows already hold a usable
``binoculars_score``, reports the findings, asks whether the rows that already
have one should be recomputed, and then fills in whatever is left.

"Already present" is decided per row, not per file: a column that exists but is
half Null counts as *partial*, and the missing rows are filled even when the
user declines to recompute. Rows whose text is empty (None/NaN/blank) are never
sent to the models and keep a Null score -- Binoculars cannot score a 0-token
input, and there is nothing meaningful to score there anyway.

Files are written **in place** unless ``-o`` is given, which is only accepted
when the command resolves to a single input file. The Parquet is saved after
every file finishes, so an interrupt half-way through a multi-file run does not
throw away the files that already completed. A file that fails mid-run is
reported and skipped -- the remaining files still run.

**Exit status**, because this script is mostly run by a pipeline rather than by
hand, and a caller has nothing else to go on:

    0   every file that had work to do was scored and written, or there was
        nothing to do because everything already holds a score
    1   at least one file failed, or the confirmation prompt was declined

Skipping a bad file is what keeps a multi-file run going; it is *not* success,
and returning 0 for it once let an unscored corpus through a pipeline that then
embedded it and reported the whole run finished. So a run with anything in
"skipped / failed" ends non-zero, however many other files went through.

Where a score is recomputed on top of an existing one, the summary reports the
change in three buckets -- identical / drifted / changed -- since re-running the
models on the same text rarely reproduces bit-identical floats. It also counts
the rows that crossed the decision threshold, which is the part that actually
changes a downstream verdict. The summary is printed and saved next to each
output as ``<stem>_rescore_summary.json``.

``--limit N`` scores only the first N rows of a file, for debugging or timing a
run before committing to the whole corpus. Rows past the limit keep whatever
they already have; nothing is dropped from the output, so the file written under
a limit is a *complete* copy in which only the first N rows were touched. Every
count in the findings table is reported within that window -- the ``window``
column shows how many rows it covers -- while the whole-file totals stay in the
saved JSON.

``--limit`` together with ``--recompute`` is the intended smoke test for a
rescoring run: on an already-scored file every row in the window has an old
value, so a 20-row run exercises the whole diff path (buckets, deltas, verdict
flips, the summary JSON) in a couple of minutes instead of hours. Point it at a
scratch ``-o`` copy so the real file is never touched. What it does *not* tell
you is how big the change is corpus-wide -- the first 20 rows are one arbitrary
slice, often all of one class -- so read it as "the mechanism works", not as an
estimate of the eventual numbers.

``--pair`` chooses the two models that do the scoring, from the registry in
``model_pairs.py``, and with them the column their scores go in. This matters
more than it looks: a Binoculars score is a ratio of two *particular* models'
perplexities, so scores from two pairs are on different scales and are not
comparable -- they must never share a column, and a rescore with a new pair is
not a rescore at all, it is a new feature. The default stays ``falcon-7b``
writing ``binoculars_score``, which is what every score already on disk is, so
existing commands keep their meaning. ``--score-column`` overrides the name;
``--observer``/``--performer`` still take arbitrary models but then *require* it.
The per-file summary JSON is likewise named after the column, so one pair's
record no longer overwrites another's.

``--max-tokens``, ``--fp32-metrics`` and ``--allow-token-mismatch`` exist for the
same experiments: the truncation cap, fp32 metrics for the wide-vocabulary pairs
(a soft-target cross-entropy over 152k bf16 terms is a good deal noisier than
over Falcon's 65k), and the escape hatch for a base/instruct pair whose
tokenizers differ only in added chat tokens. Run ``check_pairs.py`` before any of
them -- it settles tokenizer compatibility and memory from tokenizers alone,
before 30 GB of weights is downloaded.

``--gpu`` and ``--cpu-threads`` pin the run to one GPU and cap the CPU pools, so
several corpora can be scored side by side on a multi-GPU box without the runs
fighting each other for threads. Both are applied before torch is imported,
which is why the command line is parsed at module level.

Usage (from this directory, external/Binoculars):
    # fill in only the rows that have no score yet (the default)
    uv run python score_parquets.py data/wp.parquet --cpu-threads 16 --gpu 0
    uv run python score_parquets.py data/ --text-column text --batch-size 16

    # rescore every row, keeping a record of what moved
    uv run python score_parquets.py data/wp_binox0.parquet --recompute --cpu-threads 16 --gpu 0
    uv run python score_parquets.py data/essay_binox0.parquet --recompute --cpu-threads 16 --gpu 1
    uv run python score_parquets.py data/reuter_binox0.parquet --recompute --cpu-threads 16 --gpu 6

Debugging a rescore before committing to the full corpus:
    # 1. 20 rows, into a scratch copy, rescoring what is already there.
    #    Keep the prompts (no --yes) the first time so the findings table can be
    #    read before anything is written.
    uv run python score_parquets.py data/wp_binox0.parquet \
        --recompute --limit 20 -o data/wp_debug.parquet --cpu-threads 16 --gpu 0

    # 2. Check the findings table before confirming:
    #      window == 20, scored == 20, missing == 0, blank == 0
    #    "scored 20" is what makes this a real test of --recompute: every row in
    #    the window has an old score, so all 20 land in the "compared" bucket.
    #    A window showing missing == 20 means the rows were never scored and the
    #    run would only be filling gaps -- it would not exercise the diff at all.

    # 3. Check the printed summary and data/wp_debug_rescore_summary.json:
    #      change.compared == 20, change.new == 0, change.lost == 0
    #      identical + drifted + changed == 20
    #    Then diff the scratch copy against the original to confirm only the
    #    first 20 rows moved; wp_debug.parquet is a full copy of the file.

    # 4. Happy with it -> drop --limit and -o and rescore in place.
    uv run python score_parquets.py data/wp_binox0.parquet --recompute --cpu-threads 16 --gpu 0

Running from the parent repo (episteme-ai), which vendors this one as a submodule:
    # --project points uv at THIS submodule's venv, which is separate from the
    # parent's on purpose (incompatible transformers pins) and is the only one
    # holding torch + the binoculars package. Without it the parent's venv is
    # used and the doublecheck_pkgs table at startup comes up all "Missing".
    uv run --project external/Binoculars python external/Binoculars/score_parquets.py \
        data/parquet/wp_binox0.parquet --recompute --limit 20 \
        -o data/parquet/wp_debug.parquet --cpu-threads 16 --gpu 0

    # Paths are resolved against the cwd, so from the parent root they are the
    # parent's own data/parquet/*.parquet -- which is where the corpora actually
    # live. HF_TOKEN comes from the parent's .env either way: load_dotenv() walks
    # up from this file and finds it, since this submodule has no .env of its own
    # (only .env_sample). The doublecheck_env(".env") banner is cwd-relative, so
    # it prints the parent's keys from here and "Did not find file .env." from
    # inside the submodule -- informational only, it gates nothing.

My example:
    uv run --project external/Binoculars python external/Binoculars/score_parquets.py \
        data/parquet/x_ghostbuster_gpt54mini_0807A.parquet --recompute --limit 20 \
        -o data/parquet/x_ghostbuster_gpt54mini_0807A_binox0.parquet --cpu-threads 16 --gpu 5

--------------------------------------------------------------------------------
COMPARING SCORING PAIRS -- the run book
--------------------------------------------------------------------------------
Why: with the Falcon pair, gpt-5.6-luna text is separated on the Ghostbuster
domains (ROC-AUC 0.74-0.84) and *inverted* on the science domains (0.19-0.46,
AI scoring above human). Same generator, same standardisation, opposite result,
so the question is whether a newer, more science-competent pair recovers the
science domains without losing the Ghostbuster ones. Every command below is run
from the parent repo root (episteme-ai). Commands are given for GPU 0; change
--gpu per run to spread them over the box.

⚠️  ONE PAIR AT A TIME PER FILE. Adding a column is a read-modify-write of the
whole Parquet: the frame is read at startup and written back when the file
finishes. Two pairs scoring the SAME file concurrently therefore both start from
the version without either column, and whichever finishes last silently drops
the other's -- with a normal exit status and a summary claiming success.
Parallelise across *files* (science on one GPU, ghostbuster on another), never
across pairs on one file. ``epai.utils.parquet_utils.column_fill(path, column)``
reads one column's pages and is the cheap way to confirm afterwards that both
columns are really there.

0. Preflight -- no weights are downloaded, so this costs nothing:
    uv run --project external/Binoculars python external/Binoculars/check_pairs.py \
        --pairs falcon-7b qwen25-1_5b qwen25-7b falcon3-7b llama31-8b --gpu 0
   Read: tokenizer verdict per pair (identical / extended / INCOMPATIBLE), and
   the suggested batch size for the card the run will use. A pair reported
   "extended" needs --allow-token-mismatch below; "INCOMPATIBLE" is out.

1. Plumbing check, 20 rows into a scratch copy, with the pair already on disk:
    uv run --project external/Binoculars python external/Binoculars/score_parquets.py \
        data/parquet/science_v3_gpt56luna_0811A.parquet --pair falcon3-1b --limit 20 \
        -o /tmp/sci_pairtest.parquet --cpu-threads 16 --gpu 0
   Expect: state "absent" for the new column (nothing to recompute -- a new pair
   writes a new column), change.new == 20, and a summary written to
   /tmp/sci_pairtest__binoculars_score_falcon3_1b_rescore_summary.json.

2. Cheap signal check before committing to 7B weights:
    uv run --project external/Binoculars python external/Binoculars/score_parquets.py \
        data/parquet/science_v3_gpt56luna_0811A.parquet --pair qwen25-1_5b \
        --cpu-threads 16 --gpu 0 --yes
   ~6 GiB of weights, minutes for the corpus. The only question asked of it is
   whether the science distributions stop being inverted. If a 1.5B modern pair
   does not move them at all, a 7B one of the same family probably will not
   either, and the cause is not the age of the pair.

3. The pairs to report, on both corpora -- the science corpus is the one that
   fails, the Ghostbuster corpus is the control that must not break:
    for PAIR in qwen25-7b falcon3-7b llama31-8b; do
      uv run --project external/Binoculars python external/Binoculars/score_parquets.py \
          data/parquet/science_v3_gpt56luna_0811A.parquet --pair $PAIR \
          --fp32-metrics --cpu-threads 16 --gpu 0 --yes
      uv run --project external/Binoculars python external/Binoculars/score_parquets.py \
          data/parquet/ghostbuster_gpt56luna.parquet --pair $PAIR \
          --fp32-metrics --cpu-threads 16 --gpu 0 --yes
    done
   Each file is ~6000 rows; the Falcon pair does that in 17-30 min at batch 1,
   and these pairs are the same size. Add --batch-size N with the number
   check_pairs.py suggested. --fp32-metrics is on because these vocabularies are
   2-2.5x wider than Falcon's; drop it if memory is tight.

4. Read the result, one run per column -- the analysis script averages every
   binoculars_* column it finds unless it is told which one:
    uv run python -m epai.ai_detection.analyse.analyse_score_human_vs_ai \
        data/parquet/science_v3_gpt56luna_0811A.parquet:gpt56luna:0811A \
        --score-cols binoculars_score_qwen25_7b
   ROC-AUC per domain is the number that matters: > 0.5 means the pair at least
   points the right way, and the Falcon column in the same file is the baseline
   to beat. Note that a row with a null in ANY selected column is dropped, so a
   partially-scored column (a --limit run written into the real file) quietly
   shrinks the comparison -- which is why step 1 writes to /tmp.

4b. Optional, and only worth the GPU time once step 4 says a pair works: the
   same pair over the generator series, everything else held fixed, which is the
   figure "detector AUROC vs generator generation" is made from --
    ghostbuster_gpt35ts.parquet:gpt35t:0000A       (2023 generator)
    ghostbuster_gpt54mini.parquet:gpt54mini:0701A  (2025)
    ghostbuster_gpt56luna.parquet:gpt56luna:0701A  (2026)
   These three are comparable to each other only because all three were
   standardised; ghostbuster_gpt35t.parquet (no trailing "s") is the
   un-standardised original and does not belong in the series.

5. Only once a pair is chosen: refit its threshold on a held-out split and pass
   it as --threshold, otherwise the flip counts in the summaries are measured
   against Falcon's constants and mean nothing.
"""

import argparse
import os
from pathlib import Path

# Torch-free on purpose, so the pair can be resolved (and a bad --pair rejected)
# before the imports below pull torch in. See the CUDA_VISIBLE_DEVICES note.
import model_pairs
from model_pairs import COLUMN_PREFIX, DEFAULT_MAX_TOKENS, DEFAULT_PAIR, PAIRS


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="+",
                    help="Parquet file(s) and/or directories containing parquet files")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="Output path (default: overwrite each input in place). Only "
                         "accepted when the inputs resolve to a single file.")
    ap.add_argument("--text-column", default="text",
                    help="Name of the column containing text to score")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Number of texts scored per batch")
    ap.add_argument("--limit", type=int, default=None,
                    help="Debug option: only score the first N rows of each file. Later "
                         "rows keep whatever they already have, they are not dropped.")
    ap.add_argument("--pair", default=DEFAULT_PAIR, choices=sorted(PAIRS),
                    help="Scoring pair from the registry in model_pairs.py. Sets the "
                         "observer, the performer AND the score column together, so two "
                         "pairs never overwrite each other's numbers. Default: "
                         "%(default)s, the paper's pair and this repo's baseline. Run "
                         "check_pairs.py first -- it verifies tokenizers and memory "
                         "without downloading any weights.")
    ap.add_argument("--observer", default=None,
                    help="Observer model name or path, overriding --pair. Requires "
                         "--score-column, since a pair off the registry has no column "
                         "of its own.")
    ap.add_argument("--performer", default=None,
                    help="Performer model name or path, overriding --pair. Requires "
                         "--score-column.")
    ap.add_argument("--score-column", default=None,
                    help="Column the scores are written to. Default: the column the "
                         "chosen --pair owns. Scores from different pairs are NOT "
                         "comparable, so they never share a column.")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                    help="Truncation cap in tokens (default: %(default)s, the "
                         "detector's own). Text past it is not scored, so a corpus "
                         "whose classes differ in length is partly being compared on "
                         "different amounts of text.")
    ap.add_argument("--fp32-metrics", action="store_true",
                    help="Compute perplexity and cross-perplexity in fp32 while the "
                         "models still run in bf16. Worth it for the 128k-152k "
                         "vocabulary pairs, where the soft-target cross-entropy sums "
                         "over four times as many bf16 terms as Falcon's 65k.")
    ap.add_argument("--allow-token-mismatch", action="store_true",
                    help="Accept a pair whose tokenizers differ only in added control "
                         "tokens (no shared token re-mapped). Report what "
                         "check_pairs.py says before reaching for this.")
    ap.add_argument("--gpu", default=None,
                    help="value for CUDA_VISIBLE_DEVICES (which GPU(s) to use, e.g. '0' or "
                         "'0,1'). Default: leave unset so all visible GPUs are used")
    ap.add_argument("--cpu-threads", type=int, default=None,
                    help="limit CPU threads used for inference (sets OMP_NUM_THREADS / "
                         "MKL_NUM_THREADS). Default: leave unset")
    ap.add_argument("--recompute", action="store_true",
                    help="Rescore rows that already have a score, without asking.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Decision threshold used to count rows whose verdict flipped "
                         "(score < threshold => AI-generated). Default: the Binoculars "
                         "accuracy-optimised threshold.")
    ap.add_argument("--yes", action="store_true",
                    help="Skip the confirmation prompts (implies keeping existing scores "
                         "unless --recompute is also given).")
    ap.add_argument("--no-summary-file", action="store_true",
                    help="Print the summary but do not write the *_rescore_summary.json "
                         "files next to the outputs.")
    return ap.parse_args()


args = parse_args()


def resolve_pair(args: argparse.Namespace) -> tuple[str, str, str, model_pairs.Pair | None]:
    """(observer, performer, score column, registry entry) for this run.

    The registry is the normal path: ``--pair`` names two models *and* the column
    they own. Explicit ``--observer``/``--performer`` still work, but then the
    column has to be named too -- writing an unregistered pair's scores into
    ``binoculars_score`` would overwrite the Falcon baseline with numbers on a
    different scale, and nothing downstream could tell afterwards.
    """
    pair = model_pairs.resolve(args.pair)
    explicit = args.observer is not None or args.performer is not None
    if not explicit:
        return pair.observer, pair.performer, args.score_column or pair.column, pair

    observer = args.observer or pair.observer
    performer = args.performer or pair.performer
    known = model_pairs.pair_for_models(observer, performer)
    if known is not None:
        return observer, performer, args.score_column or known.column, known
    if args.score_column is None:
        raise SystemExit(
            f"--observer/--performer name a pair that is not in the registry "
            f"({observer} + {performer}), so there is no column it owns. Pass "
            f"--score-column NAME (starting with {COLUMN_PREFIX!r}), or add the pair "
            f"to model_pairs.py."
        )
    return observer, performer, args.score_column, None


OBSERVER, PERFORMER, SCORE_COL, PAIR = resolve_pair(args)

if not SCORE_COL.startswith(COLUMN_PREFIX):
    # Not fatal -- the column is written either way -- but the analysis script
    # (epai/ai_detection/analyse/analyse_score_human_vs_ai.py --metric binox)
    # finds score columns by this prefix, so a column named anything else is
    # invisible to it unless every later run passes --score-cols by hand.
    print(f"⚠️  Score column {SCORE_COL!r} does not start with {COLUMN_PREFIX!r}; "
          "the human-vs-AI analysis will not pick it up on its own.")

# CUDA_VISIBLE_DEVICES is read when the CUDA driver initialises, and the thread
# limits when the OpenMP runtime loads -- both of which happen inside torch.
# Setting them after torch is imported is too late, so they go here, before the
# imports below pull torch in transitively through binoculars.
if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
if args.cpu_threads is not None:
    os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)

import json  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from tqdm import tqdm  # noqa: E402

from binoculars import Binoculars  # noqa: E402
from binoculars.cuda_util import check_cuda  # noqa: E402
from binoculars.detector import BINOCULARS_ACCURACY_THRESHOLD  # noqa: E402
from binoculars.env_utils import doublecheck_env, doublecheck_pkgs  # noqa: E402

# How far a recomputed score may move and still count as the same score. Binoculars
# is a ratio of two model perplexities in the ~0.7-1.1 range, and re-running the
# models on the same text rarely reproduces bit-identical floats, so tiny drift is
# separated from real change instead of testing for equality.
IDENTICAL_ABS = 1e-6
DRIFT_ABS = 1e-3

load_dotenv()
# cwd-relative on purpose: the .env belongs to the checkout being worked in, so
# running this from a parent repo that vendors this one picks up *its* .env.
doublecheck_env(".env")
# Relative to this file, NOT to cwd: the check compares requirements against what
# is installed in the running interpreter, which is always this project's venv, so
# the requirements have to be this project's too. Identical to "pyproject.toml"
# when run from the project root; correct as well when run from anywhere else,
# such as from the root of a repo that has this one as a submodule.
doublecheck_pkgs(pyproject_path=Path(__file__).resolve().parent / "pyproject.toml",
                 verbose=True)
check_cuda()  # informational only; scoring still works on CPU, just slower

FLIP_THRESHOLD = args.threshold if args.threshold is not None else BINOCULARS_ACCURACY_THRESHOLD

# 0.9015 / 0.8536 were selected on Falcon-7B + Falcon-7B-Instruct outputs and mean
# nothing for any other pair -- the score is a ratio of two *particular* models'
# perplexities, and a new pair puts it on a different scale. The flip counts are
# the only thing in this script that reads a threshold, and the classifier
# downstream consumes the raw score, so an uncalibrated pair is not a reason to
# stop; it is a reason not to read "flip_to_ai" as a verdict. A per-pair threshold
# has to be refit on a held-out split before any accuracy/FPR table is written.
THRESHOLD_IS_CALIBRATED = bool(args.threshold is not None or (PAIR and PAIR.calibrated))
if not THRESHOLD_IS_CALIBRATED:
    print(f"\n⚠️  The decision threshold {FLIP_THRESHOLD:.6f} was calibrated for "
          f"{model_pairs.PAIRS['falcon-7b'].observer} + "
          f"{model_pairs.PAIRS['falcon-7b'].performer}, not for this pair.\n"
          "    Scores are still written normally; only the verdict-flip counts in the "
          "summary are meaningless.\n"
          "    Pass --threshold to count flips against a threshold refit for this pair.")


@dataclass
class FileStatus:
    """What one Parquet currently looks like, before anything is scored."""

    # Every count below except n_rows, n_filled_total and n_blank_scored describes
    # the --limit window, so the findings table matches the work that will be done.
    # Within the window: n_window == n_blank + n_filled + n_missing.
    path: Path
    n_rows: int  # rows in the file, ignoring --limit
    n_window: int  # rows inside the --limit window (== n_rows without a limit)
    exists: bool  # the score column is present at all
    n_filled: int  # non-blank rows holding a usable score
    n_missing: int  # non-blank rows with no score
    n_blank: int  # rows whose text is empty -> never scored, stay Null
    n_filled_total: int  # rows holding a usable score in the whole file
    n_blank_scored: int  # whole file: blank text yet carrying a score, left untouched
    scoreable: np.ndarray  # bool mask over the whole file, True where a row will be scored

    @property
    def state(self) -> str:
        if not self.exists:
            return "absent"
        if self.n_filled == 0:
            return "empty"
        if self.n_missing > 0:
            return "partial"
        return "complete"


def _is_empty_text(value) -> bool:
    """True for None/NaN or a string that is empty or only whitespace."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == ""


def _as_score(value) -> float | None:
    """Coerce a stored score cell to a float, or None if it holds no usable score."""
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(score) else score


def collect_paths(inputs: list[str]) -> list[Path]:
    """Expand the command line into a de-duplicated list of Parquet files."""
    paths: list[Path] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.parquet")))
        else:
            paths.append(p)

    seen: set[Path] = set()
    unique: list[Path] = []
    for p in paths:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(p)
    return unique


def check_parquet(df: pd.DataFrame, text_col: str, path: Path) -> np.ndarray:
    """Validate one Parquet and return the boolean mask of scoreable rows.

    Raises if the file is unusable: no rows, or no text column. A file where
    every text is blank is allowed through -- it simply has nothing to do, which
    the findings table reports rather than aborting a multi-file run over it.
    """
    if len(df) == 0:
        raise ValueError(f"{path} has no rows.")
    if text_col not in df.columns:
        raise ValueError(
            f"Column {text_col!r} not found in {path}. Available columns: {list(df.columns)}"
        )
    return np.array([not _is_empty_text(v) for v in df[text_col]])


def window_mask(n_rows: int, limit: int | None) -> np.ndarray:
    """Boolean mask of the rows ``--limit`` lets the run consider.

    Rows past the limit are left untouched, they are not dropped -- the output
    keeps every row, only fewer of them get a score. Kept separate from the
    blank-text mask so the findings never report an out-of-window row as blank.
    """
    window = np.ones(n_rows, dtype=bool)
    if limit is None:
        return window
    if limit < 1:
        raise ValueError(f"--limit must be >= 1, got {limit}")
    window[limit:] = False
    return window


def inspect(path: Path, text_col: str, limit: int | None) -> tuple[pd.DataFrame, FileStatus]:
    """Read one Parquet and report which rows already hold a usable score."""
    df = pd.read_parquet(path)
    nonblank = check_parquet(df, text_col, path)
    window = window_mask(len(df), limit)
    scoreable = nonblank & window

    if SCORE_COL not in df.columns:
        filled = np.zeros(len(df), dtype=bool)
        exists = False
    else:
        # The column can exist and still be Null per row, so look at the values.
        filled = np.array([_as_score(v) is not None for v in df[SCORE_COL]])
        exists = True

    return df, FileStatus(
        path=path,
        n_rows=len(df),
        n_window=int(window.sum()),
        exists=exists,
        n_filled=int((window & nonblank & filled).sum()),
        n_missing=int((window & nonblank & ~filled).sum()),
        n_blank=int((window & ~nonblank).sum()),
        n_filled_total=int(filled.sum()),
        # Independent of the window: a score sitting on blank text is stale
        # whatever the limit is, and this run will not touch it either way.
        n_blank_scored=int((~nonblank & filled).sum()),
        scoreable=scoreable,
    )


def positions_to_score(df: pd.DataFrame, st: FileStatus, recompute_existing: bool) -> np.ndarray:
    """Integer row positions to send to the models for one file."""
    if recompute_existing or not st.exists:
        return np.flatnonzero(st.scoreable)
    filled = np.array([_as_score(v) is not None for v in df[SCORE_COL]])
    return np.flatnonzero(st.scoreable & ~filled)


def describe_device() -> str:
    """One line naming what the models will actually run on."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    return f"CUDA_VISIBLE_DEVICES={visible}" if visible else "all visible devices (unset)"


def print_findings(statuses: list[FileStatus], text_col: str) -> None:
    print()
    print("=" * 86)
    print("Parquets")
    print(f"  files         : {len(statuses)}")
    print(f"  text column   : {text_col!r}")
    print(f"  score column  : {SCORE_COL!r}")
    print(f"  pair          : {PAIR.key if PAIR else 'unregistered'}"
          f"{'' if THRESHOLD_IS_CALIBRATED else '  (thresholds not calibrated for it)'}")
    print(f"  observer      : {OBSERVER}")
    print(f"  performer     : {PERFORMER}")
    print(f"  max tokens    : {args.max_tokens}")
    print(f"  metrics dtype : {'fp32' if args.fp32_metrics else 'bf16 (model dtype)'}")
    print(f"  gpu           : {describe_device()}")
    print(f"  cpu threads   : {args.cpu_threads if args.cpu_threads is not None else 'unset'}")
    print("=" * 94)
    print(f"{'file':<38}{'state':<10}{'rows':>7}{'window':>8}{'scored':>8}"
          f"{'missing':>9}{'blank':>7}  note")
    print("-" * 94)
    for st in statuses:
        note = ""
        if st.n_blank_scored:
            note = f"{st.n_blank_scored} blank row(s) carry a stale score"
        elif st.n_blank:
            note = "blank text stays Null"
        name = st.path.name
        if len(name) > 36:
            name = "..." + name[-33:]
        print(
            f"{name:<38}{st.state:<10}{st.n_rows:>7}{st.n_window:>8}{st.n_filled:>8}"
            f"{st.n_missing:>9}{st.n_blank:>7}  {note}"
        )
    print("-" * 94)

    total_blank = sum(st.n_blank for st in statuses)
    if total_blank:
        print(f"❌ {total_blank} empty/blank row(s) across all files (score left Null)")
    else:
        print("✅ Every row has a non-empty text.")
    total_stale = sum(st.n_blank_scored for st in statuses)
    if total_stale:
        print(f"⚠️  {total_stale} row(s) with blank text already carry a score. They are "
              "left exactly as they are — this run neither rescores nor clears them.")


def ask_yes_no(question: str, default: bool = False) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    answer = input(question + suffix).strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes")


def plan(statuses: list[FileStatus], recompute_existing: bool) -> list[FileStatus]:
    """The files that actually need work under the chosen policy."""
    if recompute_existing:
        return [st for st in statuses if st.n_filled > 0 or st.n_missing > 0]
    return [st for st in statuses if st.n_missing > 0]


def score_positions(bino: Binoculars, df: pd.DataFrame, text_col: str,
                    positions: np.ndarray, batch_size: int) -> tuple[pd.DataFrame, list, list]:
    """Score the given row positions and write them into the score column.

    Returns (df, old_values, new_values), the two value lists aligned to
    ``positions`` so the caller can diff what the rescoring changed.
    """
    values: list = list(df[SCORE_COL]) if SCORE_COL in df.columns else [float("nan")] * len(df)
    old_values = [_as_score(values[pos]) for pos in positions]

    texts = df[text_col].astype(str).to_numpy()
    new_values: list = []
    for i in tqdm(range(0, len(positions), batch_size), desc=df.attrs.get("name", "scoring"),
                  unit="batch", leave=False):
        chunk = positions[i:i + batch_size]
        batch_scores = bino.compute_score([texts[pos] for pos in chunk])
        new_values.extend(batch_scores)
        for pos, score in zip(chunk, batch_scores):
            values[pos] = score

    df[SCORE_COL] = pd.Series(values, index=df.index, dtype="float64")
    return df, old_values, new_values


def delta_summary(old_values: list, new_values: list) -> dict[str, object]:
    """Classify each rescored row by how far the new score moved."""
    stats: dict[str, object] = {
        "compared": 0,  # rows with a usable old and new score
        "new": 0,  # old value was Null (nothing to compare against)
        "lost": 0,  # had a score, the new one is Null (scoring produced nothing)
        "identical": 0,  # |delta| <= IDENTICAL_ABS
        "drifted": 0,  # IDENTICAL_ABS < |delta| <= DRIFT_ABS
        "changed": 0,  # |delta| >  DRIFT_ABS
        "flip_to_ai": 0,  # crossed the threshold downwards: human -> AI verdict
        "flip_to_human": 0,  # crossed the threshold upwards: AI -> human verdict
    }
    deltas: list[float] = []
    before: list[float] = []
    after: list[float] = []

    for old_raw, new_raw in zip(old_values, new_values):
        old = _as_score(old_raw)
        new = _as_score(new_raw)
        if new is None:
            if old is not None:
                stats["lost"] += 1
            continue
        if old is None:
            stats["new"] += 1
            continue

        delta = new - old
        deltas.append(delta)
        before.append(old)
        after.append(new)
        stats["compared"] += 1

        magnitude = abs(delta)
        if magnitude <= IDENTICAL_ABS:
            stats["identical"] += 1
        elif magnitude <= DRIFT_ABS:
            stats["drifted"] += 1
        else:
            stats["changed"] += 1

        # The buckets above measure movement; this measures whether the movement
        # matters, i.e. whether a downstream verdict at the threshold changed.
        was_ai = old < FLIP_THRESHOLD
        is_ai = new < FLIP_THRESHOLD
        if was_ai and not is_ai:
            stats["flip_to_human"] += 1
        elif is_ai and not was_ai:
            stats["flip_to_ai"] += 1

    stats["deltas"] = deltas
    stats["before"] = before
    stats["after"] = after
    return stats


def _distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
    }


def summary_payload(st: FileStatus, out_path: Path, n_scored: int, seconds: float,
                    stats: dict[str, object], recompute_existing: bool) -> dict[str, object]:
    """The machine-readable record of one file's run, saved beside the output."""
    deltas = stats["deltas"]
    signed = _distribution(deltas)
    absolute = _distribution([abs(d) for d in deltas])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "input": str(st.path),
        "output": str(out_path),
        "pair": PAIR.key if PAIR else None,
        "observer": OBSERVER,
        "performer": PERFORMER,
        "text_column": args.text_column,
        "score_column": SCORE_COL,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "fp32_metrics": args.fp32_metrics,
        "limit": args.limit,
        "recompute": recompute_existing,
        "flip_threshold": FLIP_THRESHOLD,
        # False => the flip counts below were measured against a threshold
        # belonging to a different pair, and mean nothing.
        "flip_threshold_calibrated": THRESHOLD_IS_CALIBRATED,
        "bucket_thresholds": {"identical_abs": IDENTICAL_ABS, "drift_abs": DRIFT_ABS},
        # Counts describe the --limit window, except the *_total / whole-file ones.
        "rows": {
            "total": st.n_rows,
            "in_limit_window": st.n_window,
            "scoreable": int(st.scoreable.sum()),
            "blank": st.n_blank,
            "already_scored_before_run": st.n_filled,
            "missing_before_run": st.n_missing,
            "state_before_run": st.state,
            "scored_in_whole_file_before_run": st.n_filled_total,
            "blank_with_stale_score_whole_file": st.n_blank_scored,
        },
        "run": {
            "scored": n_scored,
            "seconds": round(seconds, 3),
            "rows_per_second": round(n_scored / seconds, 3) if seconds else None,
        },
        "change": {
            "compared": stats["compared"],
            "new": stats["new"],
            "lost": stats["lost"],
            "identical": stats["identical"],
            "drifted": stats["drifted"],
            "changed": stats["changed"],
            "flip_to_ai": stats["flip_to_ai"],
            "flip_to_human": stats["flip_to_human"],
            "delta_signed": signed,
            "delta_abs": absolute,
            "score_before": _distribution(stats["before"]),
            "score_after": _distribution(stats["after"]),
        },
    }


def summary_path_for(out_path: Path) -> Path:
    """Where one file's summary goes -- one per (output, score column).

    Scoring the same corpus with a second pair writes a second column, so it must
    also write a second summary: keeping the old name would have each pair
    silently overwrite the record of the one before it, which is exactly the
    comparison the runs exist to make. The baseline column keeps the original
    name so the summaries already on disk stay where they are.
    """
    suffix = "" if SCORE_COL == PAIRS[DEFAULT_PAIR].column else f"__{SCORE_COL}"
    return out_path.parent / f"{out_path.stem}{suffix}_rescore_summary.json"


def save_summary(payload: dict[str, object], out_path: Path) -> Path:
    path = summary_path_for(out_path)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def print_file_stats(payload: dict[str, object], indent: str = "      ") -> None:
    """The human-readable half of one file's change statistics."""
    change = payload["change"]
    if change["new"]:
        print(f"{indent}new (no previous score)          : {change['new']}")
    if change["lost"]:
        print(f"{indent}lost (score could not be made)   : {change['lost']}")

    # Only meaningful where a previous score existed to compare against.
    if not change["compared"]:
        return
    print(f"{indent}compared with the old score      : {change['compared']}")
    print(f"{indent}  identical (|d| <= {IDENTICAL_ABS:g})       : {change['identical']}")
    print(f"{indent}  drifted   (|d| <= {DRIFT_ABS:g})       : {change['drifted']}")
    print(f"{indent}  changed   (|d| >  {DRIFT_ABS:g})       : {change['changed']}")

    absolute = change["delta_abs"]
    signed = change["delta_signed"]
    print(f"{indent}|delta| : max {absolute['max']:.6f}  median {absolute['median']:.6f}  "
          f"mean {absolute['mean']:.6f}")
    print(f"{indent} delta  : mean {signed['mean']:+.6f}  "
          f"(min {signed['min']:+.6f}, max {signed['max']:+.6f})")

    before, after = change["score_before"], change["score_after"]
    print(f"{indent}score   : mean {before['mean']:.6f} -> {after['mean']:.6f}   "
          f"median {before['median']:.6f} -> {after['median']:.6f}")
    flips = change["flip_to_ai"] + change["flip_to_human"]
    print(f"{indent}verdict flips @ {FLIP_THRESHOLD:.6f}   : {flips}"
          f"  ({change['flip_to_ai']} -> AI, {change['flip_to_human']} -> human)")


def print_summary(done: list[tuple[FileStatus, Path, dict]], failed: list[tuple[Path, str]],
                  written_summaries: list[Path]) -> None:
    print()
    print("=" * 86)
    print("Binoculars scores written")
    print("=" * 86)
    for st, out_path, payload in done:
        run = payload["run"]
        rate = run["rows_per_second"]
        rate_label = f"{rate:.2f} rows/s" if rate else "instant"
        print(f"  {out_path}")
        print(f"      scored {run['scored']} row(s) in {run['seconds']:.1f}s  ({rate_label})")
        print_file_stats(payload)
        print()

    if failed:
        print("  skipped / failed:")
        for path, message in failed:
            print(f"    {path}  {message}")
        print()

    if done:
        totals = {key: sum(p["change"][key] for _, _, p in done)
                  for key in ("compared", "new", "lost", "identical", "drifted", "changed",
                              "flip_to_ai", "flip_to_human")}
        total_scored = sum(p["run"]["scored"] for _, _, p in done)
        print("-" * 86)
        print(f"  Totals: {len(done)} file(s), {total_scored} row(s) scored")
        print(f"    new {totals['new']}  |  lost {totals['lost']}  |  "
              f"compared {totals['compared']}")
        print(f"    identical {totals['identical']}  |  drifted {totals['drifted']}  |  "
              f"changed {totals['changed']}")
        print(f"    verdict flips: {totals['flip_to_ai']} -> AI, "
              f"{totals['flip_to_human']} -> human")
        print("-" * 86)

    if written_summaries:
        print()
        print("  summary files:")
        for path in written_summaries:
            print(f"    {path}")

    if not done and not failed:
        print("  nothing computed.")
    print()


def validate_out_path(paths: list[Path]) -> None:
    """Reject -o when it cannot mean one unambiguous thing."""
    if args.out is not None and len(paths) > 1:
        raise SystemExit(
            f"-o/--out takes a single output path, but the inputs resolve to "
            f"{len(paths)} files. Run them one at a time, or drop -o to write in place."
        )


def main() -> None:
    # args is parsed at import time, before torch, so --gpu/--cpu-threads apply.
    paths = collect_paths(args.inputs)
    if not paths:
        raise SystemExit("No parquet files found to score.")
    validate_out_path(paths)

    frames: dict[Path, pd.DataFrame] = {}
    statuses: list[FileStatus] = []
    for path in paths:
        df, st = inspect(path, args.text_column, args.limit)
        df.attrs["name"] = path.name
        frames[path] = df
        statuses.append(st)

    print_findings(statuses, args.text_column)
    if args.limit is not None:
        print(f"NOTE  --limit {args.limit}: only the first {args.limit} row(s) of each file "
              "are considered and the counts above are within that window. Rows past "
              "the limit keep their current values and are still written to the output.")

    # Decide what to do with the rows that already hold a score.
    n_present = sum(1 for st in statuses if st.n_filled > 0)
    recompute_existing = args.recompute
    if n_present and not recompute_existing and not args.yes:
        recompute_existing = ask_yes_no(
            f"{n_present} file(s) already have scores. Rescore the rows that have one?",
            default=False,
        )

    todo = plan(statuses, recompute_existing)
    if not todo:
        print("\nEverything is already scored — nothing to do.")
        return

    print()
    print(f"Will score {len(todo)} file(s):")
    total_positions = 0
    for st in todo:
        positions = positions_to_score(frames[st.path], st, recompute_existing)
        total_positions += len(positions)
        out_path = args.out or st.path
        print(f"  {st.path.name:<38} {len(positions):>7} rows  ({st.state})  -> {out_path}")
    print(f"  {'total':<38} {total_positions:>7} rows")

    if not args.no_summary_file:
        print()
        print("Change statistics will be saved to:")
        for st in todo:
            print(f"  {summary_path_for(args.out or st.path)}")

    in_place = [st.path for st in todo if args.out is None]
    if in_place:
        print()
        print(f"NOTE  {len(in_place)} file(s) will be overwritten IN PLACE. "
              "Pass -o to write elsewhere (single file only).")

    if not args.yes:
        answer = input("Proceed? Type 'yes': ").strip().lower()
        if answer != "yes":
            # Non-zero: declining is a decision not to do the work, and a
            # caller that reads only the exit code must not take it for a run
            # that scored everything.
            raise SystemExit("Aborted — no changes made.")

    bino = Binoculars(
        observer_name_or_path=OBSERVER,
        performer_name_or_path=PERFORMER,
        max_token_observed=args.max_tokens,
        strict_tokenizer_check=not args.allow_token_mismatch,
        fp32_metrics=args.fp32_metrics,
    )

    done: list[tuple[FileStatus, Path, dict]] = []
    failed: list[tuple[Path, str]] = []
    written_summaries: list[Path] = []

    for st in tqdm(todo, desc="Files", unit="file"):
        positions = positions_to_score(frames[st.path], st, recompute_existing)
        if len(positions) == 0:
            continue

        out_path = args.out or st.path
        tqdm.write(f"→ {st.path.name}  ({len(positions)} rows, batch {args.batch_size})")
        started = time.perf_counter()
        try:
            df, old_values, new_values = score_positions(
                bino, frames[st.path], args.text_column, positions, args.batch_size
            )
        except Exception as exc:  # keep going: one bad file must not lose the rest
            failed.append((st.path, f"{type(exc).__name__}: {exc}"))
            tqdm.write(f"  failed: {type(exc).__name__}: {exc}")
            continue

        elapsed = time.perf_counter() - started
        payload = summary_payload(
            st, out_path, len(positions), elapsed, delta_summary(old_values, new_values),
            recompute_existing,
        )

        # Save after every file so an interrupt does not lose finished work.
        df.to_parquet(out_path, index=False)
        done.append((st, out_path, payload))
        if not args.no_summary_file:
            written_summaries.append(save_summary(payload, out_path))

    print_summary(done, failed, written_summaries)

    # After the summary, not instead of it: the caller gets a non-zero status
    # and the operator still gets the whole table, including the files that did
    # go through. The per-file `except` above is what keeps one bad file from
    # losing the rest -- it is not a reason to call the run a success.
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

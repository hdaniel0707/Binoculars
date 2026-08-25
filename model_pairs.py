"""The scoring pairs Binoculars can be run with, and what each one costs.

Binoculars scores text with *two* models -- an observer and a performer -- and
the paper's numbers are all for one particular pair, ``tiiuae/falcon-7b`` +
``tiiuae/falcon-7b-instruct`` (2023). That pair is only a detector for as long
as it remains a reasonable proxy for whatever wrote the text: when the generator
is far outside its competence the ratio stops separating the classes, and it can
invert outright -- machine text scoring *above* human text, which reads as a
below-chance detector rather than a useless one.

This module is the list of alternative pairs plus the arithmetic needed to
choose one, and it is deliberately free of torch so that
:mod:`score_parquets` can read it before torch is imported (which is what lets
``--gpu`` still take effect) and :mod:`check_pairs` can read it on a machine
with no GPU at all.

Every pair is base + instruct of one family, which is not a stylistic choice:

* the two models must share a tokenizer, because the text is encoded once and
  both models are fed the same ids (see ``binoculars.utils``);
* the paper's ablation found the method works best when the two models are
  *close* in capability -- it is not contrastive decoding, where a strong model
  is paired with a deliberately weak one;
* the signal comes from the instruction tuning: performance rises nearly
  monotonically with how instruction-tuned the performer is.

``params`` and ``vocab`` are the published figures, used only to estimate
memory before anything is downloaded; ``check_pairs.py`` refines them from the
Hub when it can reach it.
"""

from dataclasses import dataclass

# What Binoculars truncates to, and therefore the sequence length every memory
# estimate below is written for. Overridable per run with --max-tokens.
DEFAULT_MAX_TOKENS = 512

# Peak transient tensors, counted in full logits-sized copies, summed over BOTH
# models: the two logits tensors themselves, the observer's softmax, and the
# working copies inside the soft-target cross-entropy. A rough upper bound --
# the point is to land on a safe batch size, not to predict the allocator.
_LOGIT_COPIES = 6

_GIB = 1024 ** 3


@dataclass(frozen=True)
class Pair:
    """One observer/performer pair and the column its scores belong in."""

    key: str  # --pair value
    observer: str
    performer: str
    column: str  # score column, so two pairs never overwrite each other
    params: float  # parameters per model, in billions (both models are ~equal)
    vocab: int  # logit width -- as important as params for peak memory
    year: str  # release of the pair, i.e. how modern a proxy it is
    calibrated: bool = False  # are the hardcoded thresholds valid for it?
    gated: bool = False  # needs HF_TOKEN with accepted licence
    note: str = ""

    @property
    def weights_gib(self) -> float:
        """Both models resident, bfloat16."""
        return 2 * self.params * 1e9 * 2 / _GIB

    def transient_gib(self, batch_size: int, max_tokens: int = DEFAULT_MAX_TOKENS,
                      fp32_metrics: bool = False) -> float:
        """Peak logits/softmax working set for one batch, both models."""
        # With --fp32-metrics the bf16 logits stay live while the fp32 copies
        # are made, so the per-copy cost is 2 + 4 bytes, not 4.
        bytes_per_element = 6 if fp32_metrics else 2
        return (batch_size * max_tokens * self.vocab
                * bytes_per_element * _LOGIT_COPIES) / _GIB

    def total_gib(self, batch_size: int, max_tokens: int = DEFAULT_MAX_TOKENS,
                  fp32_metrics: bool = False) -> float:
        return self.weights_gib + self.transient_gib(batch_size, max_tokens, fp32_metrics)

    def fits(self, device_gib: float, batch_size: int,
             max_tokens: int = DEFAULT_MAX_TOKENS, fp32_metrics: bool = False,
             slack_gib: float = 2.0) -> bool:
        return self.total_gib(batch_size, max_tokens, fp32_metrics) + slack_gib <= device_gib

    def max_batch_size(self, device_gib: float, max_tokens: int = DEFAULT_MAX_TOKENS,
                       fp32_metrics: bool = False, slack_gib: float = 2.0,
                       cap: int = 32) -> int:
        """Largest batch size that still fits, or 0 if even one row does not."""
        for batch_size in range(cap, 0, -1):
            if self.fits(device_gib, batch_size, max_tokens, fp32_metrics, slack_gib):
                return batch_size
        return 0


# Ordered by intended use: the baseline first, then the two 7-8B replacements
# worth reporting, then the cheap pairs a smoke test should run before anything
# large is downloaded.
PAIRS: dict[str, Pair] = {
    "falcon-7b": Pair(
        key="falcon-7b",
        observer="tiiuae/falcon-7b",
        performer="tiiuae/falcon-7b-instruct",
        column="binoculars_score",  # the legacy name: every existing score is this pair
        params=7.22, vocab=65024, year="2023",
        calibrated=True,
        note="The paper's pair and this repo's baseline. Thresholds are only "
             "valid here. Keep it as the control, do not delete its column.",
    ),
    "qwen25-7b": Pair(
        key="qwen25-7b",
        observer="Qwen/Qwen2.5-7B",
        performer="Qwen/Qwen2.5-7B-Instruct",
        column="binoculars_score_qwen25_7b",
        params=7.62, vocab=152064, year="2024",
        note="Primary replacement: 18T-token pretrain heavy on maths, code and "
             "scientific text, which is exactly where the Falcon pair fails.",
    ),
    "falcon3-7b": Pair(
        key="falcon3-7b",
        observer="tiiuae/Falcon3-7B-Base",
        performer="tiiuae/Falcon3-7B-Instruct",
        column="binoculars_score_falcon3_7b",
        params=7.46, vocab=131072, year="2024",
        note="Same family as the baseline, one generation newer -- the "
             "controlled comparison in which only the age of the pair changed.",
    ),
    "llama31-8b": Pair(
        key="llama31-8b",
        observer="meta-llama/Llama-3.1-8B",
        performer="meta-llama/Llama-3.1-8B-Instruct",
        column="binoculars_score_llama31_8b",
        params=8.03, vocab=128256, year="2024",
        gated=True,
        note="Third family, different tokenizer and data mix. Gated on the Hub: "
             "needs HF_TOKEN and an accepted licence.",
    ),
    "qwen3-8b": Pair(
        key="qwen3-8b",
        observer="Qwen/Qwen3-8B-Base",
        performer="Qwen/Qwen3-8B",
        column="binoculars_score_qwen3_8b",
        params=8.19, vocab=151936, year="2025",
        note="Newest available. The performer is a hybrid reasoning model; it "
             "is only scoring likelihoods here, but its chat template and "
             "thinking mode are irrelevant to that, not helpful.",
    ),
    "qwen25-3b": Pair(
        key="qwen25-3b",
        observer="Qwen/Qwen2.5-3B",
        performer="Qwen/Qwen2.5-3B-Instruct",
        column="binoculars_score_qwen25_3b",
        params=3.09, vocab=151936, year="2024",
        note="Cheap first pass with the same data mix as qwen25-7b.",
    ),
    "qwen25-1_5b": Pair(
        key="qwen25-1_5b",
        observer="Qwen/Qwen2.5-1.5B",
        performer="Qwen/Qwen2.5-1.5B-Instruct",
        column="binoculars_score_qwen25_1_5b",
        params=1.54, vocab=151936, year="2024",
        note="Smallest sensible smoke test: ~6 GiB of weights, minutes per corpus.",
    ),
    "falcon3-1b": Pair(
        key="falcon3-1b",
        observer="tiiuae/Falcon3-1B-Base",
        performer="tiiuae/Falcon3-1B-Instruct",
        column="binoculars_score_falcon3_1b",
        params=1.67, vocab=131072, year="2024",
        note="Already fetched by download_models.py -- the zero-download way to "
             "prove the plumbing (new column, new summary file) works.",
    ),
    "mistral-v03": Pair(
        key="mistral-v03",
        observer="mistralai/Mistral-7B-v0.3",
        performer="mistralai/Mistral-7B-Instruct-v0.3",
        column="binoculars_score_mistral_v03",
        params=7.25, vocab=32768, year="2024",
        note="Memory-pressure fallback: a 32k vocabulary makes it by far the "
             "cheapest 7B pair. Only marginally newer than the baseline, so a "
             "weak proxy for a 2026 generator.",
    ),
    "gemma2-9b": Pair(
        key="gemma2-9b",
        observer="google/gemma-2-9b",
        performer="google/gemma-2-9b-it",
        column="binoculars_score_gemma2_9b",
        params=9.24, vocab=256000, year="2024",
        gated=True,
        note="80 GB card only: 34 GiB of weights plus a 256k-wide logits "
             "tensor. Gated on the Hub.",
    ),
}

DEFAULT_PAIR = "falcon-7b"

# Every score column starts with this, which is what makes
# epai/ai_detection/analyse/analyse_score_human_vs_ai.py --metric binox find
# them. That script averages every matching column unless it is given
# --score-cols, so one run per pair, naming its column, is the correct usage.
COLUMN_PREFIX = "binoculars_"


def resolve(key: str) -> Pair:
    try:
        return PAIRS[key]
    except KeyError:
        raise SystemExit(
            f"Unknown --pair {key!r}. Known pairs: {', '.join(PAIRS)}"
        ) from None


def pair_for_models(observer: str, performer: str) -> Pair | None:
    """The registry entry for an explicitly given pair of model ids, if any."""
    for pair in PAIRS.values():
        if pair.observer == observer and pair.performer == performer:
            return pair
    return None

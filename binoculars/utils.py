"""Tokenizer checks shared by the detector and the preflight script.

Binoculars encodes the text **once**, with the observer's tokenizer, and feeds
those same ``input_ids`` to both models. That only means anything if an id
denotes the same token for both, which is why the pair has to share a
tokenizer. The original check was an equality test on the two vocabularies; it
is kept, but split so a caller can see *how* they differ, because the two ways
of differing are not equally fatal:

* a token that both know but at **different ids** -- the performer would read
  ids that mean something else. Always fatal.
* a token **only one side declares** -- in practice a chat/tool control token an
  instruct variant adds at the end of the vocabulary. Ordinary prose never
  encodes to it, so the ids the models actually see are unaffected. This is what
  makes an otherwise usable base/instruct pair fail the equality test, and it is
  the only case ``strict=False`` lets through.
"""

from transformers import AutoTokenizer

# How many differing tokens are named in a message before it is truncated.
_DIFF_SHOWN = 12


def _vocab(model_id: str) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    vocab = getattr(tokenizer, "vocab", None)
    return dict(vocab) if vocab is not None else dict(tokenizer.get_vocab())


def tokenizer_diff(model_id_1: str, model_id_2: str) -> dict:
    """Compare two tokenizers' vocabularies without loading any weights.

    Returns a dict with ``identical``, the two sizes, the tokens declared by
    only one side, and ``remapped`` -- tokens both know but at different ids,
    which is the disagreement that actually corrupts the scores.
    """
    v1, v2 = _vocab(model_id_1), _vocab(model_id_2)
    shared = v1.keys() & v2.keys()
    remapped = sorted(tok for tok in shared if v1[tok] != v2[tok])
    return {
        "identical": v1 == v2,
        "size_1": len(v1),
        "size_2": len(v2),
        "only_in_1": sorted(v1.keys() - v2.keys()),
        "only_in_2": sorted(v2.keys() - v1.keys()),
        "remapped": remapped,
        "compatible": not remapped,
    }


def describe_tokenizer_diff(model_id_1: str, model_id_2: str, diff: dict) -> str:
    """One human-readable paragraph explaining a non-identical pair."""
    def head(tokens: list[str]) -> str:
        shown = ", ".join(repr(t) for t in tokens[:_DIFF_SHOWN])
        extra = len(tokens) - _DIFF_SHOWN
        return shown + (f", ... (+{extra} more)" if extra > 0 else "")

    lines = [
        f"Tokenizers differ for {model_id_1} and {model_id_2} "
        f"({diff['size_1']} vs {diff['size_2']} entries)."
    ]
    if diff["remapped"]:
        lines.append(f"  {len(diff['remapped'])} token(s) share a name but not an id: "
                     f"{head(diff['remapped'])}")
    if diff["only_in_1"]:
        lines.append(f"  only in {model_id_1}: {head(diff['only_in_1'])}")
    if diff["only_in_2"]:
        lines.append(f"  only in {model_id_2}: {head(diff['only_in_2'])}")
    return "\n".join(lines)


def assert_tokenizer_consistency(model_id_1: str, model_id_2: str, strict: bool = True) -> dict:
    """Raise unless the two models can share one encoding of the input.

    ``strict`` (the default, and the original behaviour) demands identical
    vocabularies. ``strict=False`` accepts a pair that differs only in added
    tokens, printing what it accepted; a re-mapped shared token still raises,
    because no flag makes that safe.
    """
    diff = tokenizer_diff(model_id_1, model_id_2)
    if diff["identical"]:
        return diff

    message = describe_tokenizer_diff(model_id_1, model_id_2, diff)
    if strict or not diff["compatible"]:
        raise ValueError(
            message + "\n"
            + ("Re-mapped tokens cannot be worked around: the two models would read "
               "different tokens from the same ids.\n" if not diff["compatible"] else
               "Pass strict=False (score_parquets.py: --allow-token-mismatch) if the "
               "extra tokens are control tokens ordinary text never encodes to.\n")
        )

    print("⚠️  " + message)
    print("    Accepted: no shared token is re-mapped, so the ids both models see are "
          "the same for ordinary text. Only added control tokens differ.")
    return diff

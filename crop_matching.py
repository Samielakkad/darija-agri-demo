"""Dependency-free Arabic normalization and crop alias matching."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence


ARABIC_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
ARABIC_LETTER_REPLACEMENTS = {
    "آ": "ا",
    "أ": "ا",
    "إ": "ا",
    "ٱ": "ا",
    "ى": "ي",
    "ة": "ه",
}
ALIAS_SEPARATORS = re.compile(r"[/,،()]+")


def normalize_arabic(text: str) -> list[str]:
    """Normalize text into comparable Arabic/Latin alphanumeric tokens."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = unicodedata.normalize("NFKC", text)
    text = ARABIC_DIACRITICS.sub("", text).replace("ـ", "")
    for source, replacement in ARABIC_LETTER_REPLACEMENTS.items():
        text = text.replace(source, replacement)

    normalized = []
    for character in text.casefold():
        category = unicodedata.category(character)
        if category[0] in {"L", "N"}:
            normalized.append(character)
        elif category in {"Cf", "Mn", "Me"}:
            # Formatting controls and residual combining marks should not split words.
            continue
        else:
            normalized.append(" ")
    return "".join(normalized).split()


def split_crop_guess(value: str) -> list[str]:
    """Split an alias entry that contains alternative spellings."""
    return [part.strip() for part in ALIAS_SEPARATORS.split(value) if part.strip()]


def _contains_alias(prediction: Sequence[str], alias: Sequence[str]) -> bool:
    if not alias:
        return False

    alias_length = len(alias)
    if any(
        list(prediction[start : start + alias_length]) == list(alias)
        for start in range(len(prediction) - alias_length + 1)
    ):
        return True

    # Models sometimes omit spaces inside Arabic crop names. Equality with one
    # complete token preserves that useful case without unsafe substring matches.
    compact_alias = "".join(alias)
    return compact_alias in prediction


def crop_match(
    prediction: str, aliases: Mapping[str, Sequence[str]]
) -> str | None:
    """Return the class with the longest boundary-safe alias in ``prediction``."""
    prediction_tokens = normalize_arabic(prediction)
    best_class = None
    best_length = 0

    for class_name, class_aliases in aliases.items():
        for raw_alias in class_aliases:
            for alias_part in split_crop_guess(raw_alias):
                alias_tokens = normalize_arabic(alias_part)
                compact_length = len("".join(alias_tokens))
                if (
                    compact_length > best_length
                    and _contains_alias(prediction_tokens, alias_tokens)
                ):
                    best_class = class_name
                    best_length = compact_length

    return best_class

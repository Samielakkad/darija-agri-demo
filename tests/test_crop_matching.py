import json
import unittest
from pathlib import Path

from crop_matching import crop_match, normalize_arabic, split_crop_guess


ROOT = Path(__file__).resolve().parents[1]


class NormalizeArabicTests(unittest.TestCase):
    def test_normalizes_diacritics_tatweel_and_letter_variants(self):
        self.assertEqual(normalize_arabic("اَلْقَـمْحُ"), ["القمح"])
        self.assertEqual(normalize_arabic("آ إ أ ٱ ى ة"), ["ا", "ا", "ا", "ا", "ي", "ه"])

    def test_treats_arabic_and_latin_punctuation_as_boundaries(self):
        self.assertEqual(
            normalize_arabic("القمح، والطماطم... CROP-ID"),
            ["القمح", "والطماطم", "crop", "id"],
        )

    def test_preserves_digits_and_removes_format_controls(self):
        self.assertEqual(normalize_arabic("ق‍مح 2026"), ["قمح", "2026"])

    def test_rejects_non_string_input(self):
        with self.assertRaisesRegex(TypeError, "text must be a string"):
            normalize_arabic(None)


class CropMatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "crop_aliases.json").open(encoding="utf-8") as alias_file:
            cls.aliases = json.load(alias_file)

    def test_every_shipped_alias_resolves_to_its_class(self):
        for class_name, aliases in self.aliases.items():
            for alias in aliases:
                with self.subTest(class_name=class_name, alias=alias):
                    prediction = f"هاد المحصول هو {alias}."
                    self.assertEqual(crop_match(prediction, self.aliases), class_name)

    def test_matches_diacritics_and_arabic_punctuation(self):
        self.assertEqual(crop_match("هَذَا اَلْقَـمْحُ، واضح", self.aliases), "wheat")

    def test_longest_alias_wins_for_overlapping_crop_names(self):
        self.assertEqual(crop_match("هذه الذرة الرفيعة", self.aliases), "jowar")

    def test_matches_compact_multiword_arabic_alias(self):
        self.assertEqual(crop_match("الصورة فيها جوزالهند", self.aliases), "coconut")

    def test_does_not_match_inside_a_longer_word(self):
        self.assertIsNone(crop_match("هذا البناء جديد", self.aliases))
        self.assertIsNone(crop_match("الحمصة صغيرة", self.aliases))
        self.assertIsNone(crop_match("teaching", {"tea": ["tea"]}))

    def test_requires_multiword_aliases_in_order_and_together(self):
        aliases = {"coconut": ["جوز الهند"]}
        self.assertIsNone(crop_match("الهند فيها جوز كثير", aliases))
        self.assertIsNone(crop_match("جوز طازج من جنوب الهند", aliases))

    def test_handles_empty_prediction_and_empty_aliases(self):
        self.assertIsNone(crop_match("", self.aliases))
        self.assertIsNone(crop_match("القمح", {"wheat": ["", "()"]}))

    def test_split_crop_guess_supports_alias_alternatives(self):
        self.assertEqual(
            split_crop_guess("القمح / الحنطة، wheat (grain)"),
            ["القمح", "الحنطة", "wheat", "grain"],
        )


if __name__ == "__main__":
    unittest.main()

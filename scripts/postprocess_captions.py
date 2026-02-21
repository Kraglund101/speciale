"""Post-process captions_nano.json: ASCII normalization + annotation leak removal.

Saves cleaned version to captions_nano_clean.json. Original is NOT modified.
"""
import json
import re
import os

SRC = "anomverse_extension/datasets/realiad_1024/captions_nano.json"
DST = "anomverse_extension/datasets/realiad_1024/captions_nano_clean.json"

# ── 1. ASCII normalization table ──────────────────────────────────────────────
ASCII_MAP = {
    "\u2019": "'",   # right single quote → ASCII apostrophe
    "\u2018": "'",   # left single quote → ASCII apostrophe
    "\u201c": '"',   # left double quote → ASCII double quote
    "\u201d": '"',   # right double quote → ASCII double quote
    "\u2014": "--",  # em dash → double hyphen
    "\u2013": "-",   # en dash → hyphen
    "\u2011": "-",   # non-breaking hyphen → hyphen
    "\u00ad": "",    # soft hyphen → remove (invisible break hint)
    "\u0435": "e",   # Cyrillic е → Latin e (visually identical)
}

# Korean word that slipped in (1 entry)
KOREAN_PATTERN = re.compile(r"\s*주변\s*")

# Combining acute accent normalization: é (e + U+0301) → e
COMBINING_ACUTE = re.compile(r"e\u0301")


def ascii_normalize(text: str) -> str:
    """Replace non-ASCII typography with ASCII equivalents."""
    for char, replacement in ASCII_MAP.items():
        text = text.replace(char, replacement)
    text = KOREAN_PATTERN.sub(" ", text)
    text = COMBINING_ACUTE.sub("e", text)
    # Clean up any double spaces introduced
    text = re.sub(r"  +", " ", text).strip()
    return text


# ── 2. Annotation leak patterns ──────────────────────────────────────────────
# These are the TRUE leak patterns — prepositional phrases referencing the
# red bounding box annotations we drew on the input images.
#
# Strategy: remove the annotation phrase, preserving surrounding grammar.
# These are always redundant because the real spatial location is already
# stated elsewhere in the sentence.

ANNOTATION_PATTERNS = [
    # "within/in the highlighted area/region/zone"
    # Optional preceding comma or ", located" etc.
    re.compile(
        r",?\s*(?:located\s+)?(?:within|in|inside)\s+the\s+(?:red[- ])?highlighted\s+"
        r"(?:area|region|zone|section)",
        re.IGNORECASE,
    ),
    # "within/in the marked area/region"
    re.compile(
        r",?\s*(?:located\s+)?(?:within|in|inside)\s+the\s+(?:red[- ])?marked\s+"
        r"(?:area|region|zone|section)",
        re.IGNORECASE,
    ),
    # "inside/within the red box (in the full image)"
    re.compile(
        r",?\s*(?:located\s+)?(?:within|in|inside)\s+the\s+red\s+box"
        r"(?:\s+in\s+the\s+full\s+image)?",
        re.IGNORECASE,
    ),
    # "in the full view and is highlighted in the close-up" → just "in the full view"
    re.compile(
        r"\s+and\s+is\s+highlighted\s+in\s+the\s+close-up",
        re.IGNORECASE,
    ),
    # "shown as" when referencing annotations (very rare)
    re.compile(
        r",?\s+shown\s+in\s+the\s+red[- ](?:box|highlighted|marked)\s+(?:area|region)",
        re.IGNORECASE,
    ),
]

# Sentence-initial annotation references need special handling:
# "In the red-highlighted area, ..." → drop the phrase, capitalize what follows
SENTENCE_INITIAL_PATTERNS = [
    re.compile(
        r"^(?:In|Within|Inside)\s+the\s+(?:red[- ])?(?:highlighted|marked|boxed)\s+"
        r"(?:area|region|zone|section),?\s*",
        re.IGNORECASE,
    ),
]


def fix_annotation_leaks(text: str) -> tuple[str, list[str]]:
    """Remove annotation reference phrases. Returns (cleaned_text, list_of_removals)."""
    removals = []

    # Handle mid-sentence patterns
    for pat in ANNOTATION_PATTERNS:
        match = pat.search(text)
        if match:
            removals.append(match.group().strip())
            text = pat.sub("", text)

    # Handle sentence-initial patterns
    for pat in SENTENCE_INITIAL_PATTERNS:
        match = pat.match(text)
        if match:
            removals.append(match.group().strip())
            text = pat.sub("", text)
            # Capitalize first letter of remaining text
            if text:
                text = text[0].upper() + text[1:]

    # Clean up artifacts: double spaces, leading/trailing commas, double periods
    text = re.sub(r"  +", " ", text)
    text = re.sub(r"\s*,\s*,", ",", text)    # double commas
    text = re.sub(r"\.\s*\.", ".", text)      # double periods
    text = re.sub(r",\s*\.", ".", text)       # comma before period
    text = re.sub(r"^\s*,\s*", "", text)      # leading comma
    text = text.strip()

    return text, removals


# ── 3. Run ────────────────────────────────────────────────────────────────────
def main():
    with open(SRC, encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} entries from {SRC}")

    ascii_changed = 0
    annotation_fixed = 0
    all_annotation_fixes = []

    for i, entry in enumerate(data):
        # --- ASCII normalization ---
        orig_short = entry["caption_short"]
        orig_long = entry["caption_long"]

        entry["caption_short"] = ascii_normalize(entry["caption_short"])
        entry["caption_long"] = ascii_normalize(entry["caption_long"])

        if entry["caption_short"] != orig_short or entry["caption_long"] != orig_long:
            ascii_changed += 1

        # --- Annotation leak removal ---
        short_clean, short_removals = fix_annotation_leaks(entry["caption_short"])
        long_clean, long_removals = fix_annotation_leaks(entry["caption_long"])

        if short_removals or long_removals:
            annotation_fixed += 1
            all_annotation_fixes.append({
                "index": i,
                "file": entry["filename"],
                "product": entry["product"],
                "defect": entry["defect_name"],
                "short_removals": short_removals,
                "long_removals": long_removals,
                "short_before": entry["caption_short"],
                "short_after": short_clean,
                "long_before": entry["caption_long"],
                "long_after": long_clean,
            })

        entry["caption_short"] = short_clean
        entry["caption_long"] = long_clean

    # Save cleaned version
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)

    print(f"\nSaved cleaned captions to {DST}")
    print(f"\n=== SUMMARY ===")
    print(f"  ASCII normalized: {ascii_changed} entries")
    print(f"  Annotation leaks fixed: {annotation_fixed} entries")
    print(f"  Total entries: {len(data)}")

    # Print all annotation fixes for review
    if all_annotation_fixes:
        print(f"\n=== ANNOTATION LEAK FIXES ({len(all_annotation_fixes)}) ===")
        for fix in all_annotation_fixes:
            print(f"\n  [{fix['index']}] {fix['product']}/{fix['defect']} - {fix['file']}")
            for r in fix["short_removals"]:
                print(f"    SHORT removed: \"{r}\"")
            for r in fix["long_removals"]:
                print(f"    LONG removed: \"{r}\"")
            if fix["short_removals"]:
                print(f"    SHORT before: {fix['short_before'][:120]}...")
                print(f"    SHORT after:  {fix['short_after'][:120]}...")
            if fix["long_removals"]:
                print(f"    LONG before:  {fix['long_before'][:150]}...")
                print(f"    LONG after:   {fix['long_after'][:150]}...")


if __name__ == "__main__":
    main()

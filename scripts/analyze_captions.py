"""
Comprehensive analysis of captions_nano.json for post-processing cleanup.

Checks for:
1. Non-ASCII characters (curly quotes, em dashes, non-breaking hyphens, etc.)
2. Bbox annotation leaks (red box, bounding box, etc.)
3. Chinese/CJK characters
4. Measurement leaks (mm, cm, inch)
5. Empty or suspiciously short captions
6. Missing SHORT:/LONG: format in raw_response
7. Duplicate captions
8. Very long captions (over word budget)
9. Self-referential language (Image 1, crop, close-up, etc.)
10. Prompt leakage
"""

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_captions(path: str) -> list[dict[str, Any]]:
    """Load the captions JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries")
    return data


def analyze_non_ascii(data: list[dict]) -> dict:
    """Find all non-ASCII characters across caption fields."""
    results = {
        "char_counts": Counter(),
        "affected_entries": [],
        "per_char_examples": defaultdict(list),
    }

    fields = ["caption_short", "caption_long", "raw_response"]

    for i, entry in enumerate(data):
        entry_non_ascii = {}
        for field in fields:
            text = entry.get(field, "")
            if not text:
                continue
            non_ascii = [(ch, ord(ch), unicodedata.name(ch, f"U+{ord(ch):04X}"))
                         for ch in text if ord(ch) > 127]
            if non_ascii:
                entry_non_ascii[field] = non_ascii
                for ch, code, name in non_ascii:
                    results["char_counts"][(ch, code, name)] += 1
                    if len(results["per_char_examples"][(ch, code, name)]) < 5:
                        # Find context around the character
                        idx = text.index(ch)
                        start = max(0, idx - 30)
                        end = min(len(text), idx + 30)
                        context = text[start:end]
                        results["per_char_examples"][(ch, code, name)].append({
                            "entry_idx": i,
                            "product": entry.get("product", ""),
                            "defect_name": entry.get("defect_name", ""),
                            "field": field,
                            "context": context,
                        })
        if entry_non_ascii:
            results["affected_entries"].append({
                "entry_idx": i,
                "product": entry.get("product", ""),
                "defect_name": entry.get("defect_name", ""),
                "filename": entry.get("filename", ""),
                "chars": entry_non_ascii,
            })

    return results


def analyze_bbox_leaks(data: list[dict]) -> dict:
    """Find captions that describe bounding box annotations."""
    patterns = [
        (r"\bred\s+box\b", "red box"),
        (r"\bred\s+rectangle\b", "red rectangle"),
        (r"\bbounding\s+box\b", "bounding box"),
        (r"\bred\s+outline\b", "red outline"),
        (r"\bhighlighted\b", "highlighted"),
        (r"\bmarked\s+area\b", "marked area"),
        (r"\bannotated\b", "annotated"),
        (r"\bred\s+border\b", "red border"),
        (r"\bindicated\s+by\b", "indicated by"),
        (r"\benclosed\b", "enclosed"),
        (r"\boutlined\s+in\s+red\b", "outlined in red"),
        (r"\bred\s+square\b", "red square"),
        (r"\bred\s+marking\b", "red marking"),
        (r"\bred\s+highlight\b", "red highlight"),
        (r"\bred\s+frame\b", "red frame"),
        (r"\bred\s+circle\b", "red circle"),
        (r"\bred\s+annotation\b", "red annotation"),
        (r"\bbounded\s+by\b", "bounded by"),
        (r"\bred\s+overlay\b", "red overlay"),
        (r"\bred\s+boundary\b", "red boundary"),
        (r"\bred\s+line\b", "red line"),
        (r"\bred\s+contour\b", "red contour"),
    ]

    results = {"affected_entries": [], "pattern_counts": Counter()}
    fields = ["caption_short", "caption_long"]

    for i, entry in enumerate(data):
        entry_matches = {}
        for field in fields:
            text = entry.get(field, "")
            if not text:
                continue
            field_matches = []
            for pattern, label in patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                if matches:
                    for m in matches:
                        start = max(0, m.start() - 40)
                        end = min(len(text), m.end() + 40)
                        context = text[start:end]
                        field_matches.append({
                            "pattern": label,
                            "match": m.group(),
                            "context": context,
                        })
                        results["pattern_counts"][label] += 1
            if field_matches:
                entry_matches[field] = field_matches
        if entry_matches:
            results["affected_entries"].append({
                "entry_idx": i,
                "product": entry.get("product", ""),
                "defect_name": entry.get("defect_name", ""),
                "filename": entry.get("filename", ""),
                "caption_short": entry.get("caption_short", ""),
                "caption_long": entry.get("caption_long", ""),
                "matches": entry_matches,
            })

    return results


def analyze_cjk(data: list[dict]) -> dict:
    """Find Chinese/CJK characters."""
    cjk_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\u2e80-\u2eff\u3000-\u303f\uff00-\uffef]')
    results = {"affected_entries": []}
    fields = ["caption_short", "caption_long", "raw_response"]

    for i, entry in enumerate(data):
        entry_matches = {}
        for field in fields:
            text = entry.get(field, "")
            if not text:
                continue
            matches = cjk_pattern.findall(text)
            if matches:
                entry_matches[field] = matches
        if entry_matches:
            results["affected_entries"].append({
                "entry_idx": i,
                "product": entry.get("product", ""),
                "defect_name": entry.get("defect_name", ""),
                "filename": entry.get("filename", ""),
                "chars": entry_matches,
            })

    return results


def analyze_measurements(data: list[dict]) -> dict:
    """Find measurement leaks (mm, cm, inch, etc.)."""
    patterns = [
        (r'\b\d+\.?\d*\s*mm\b', "mm"),
        (r'\b\d+\.?\d*\s*cm\b', "cm"),
        (r'\b\d+\.?\d*\s*inch(?:es)?\b', "inch"),
        (r'\b\d+\.?\d*\s*μm\b', "μm"),
        (r'\b\d+\.?\d*\s*micron\b', "micron"),
        (r'\b\d+\.?\d*\s*meter\b', "meter"),
        (r'\b\d+\.?\d*\s*pixel\b', "pixel"),
        (r'\b\d+\s*×\s*\d+\b', "dimension"),
        (r'\b\d+x\d+\b', "dimension_x"),
    ]
    results = {"affected_entries": [], "pattern_counts": Counter()}
    fields = ["caption_short", "caption_long"]

    for i, entry in enumerate(data):
        entry_matches = {}
        for field in fields:
            text = entry.get(field, "")
            if not text:
                continue
            field_matches = []
            for pattern, label in patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                if matches:
                    for m in matches:
                        start = max(0, m.start() - 30)
                        end = min(len(text), m.end() + 30)
                        context = text[start:end]
                        field_matches.append({
                            "pattern": label,
                            "match": m.group(),
                            "context": context,
                        })
                        results["pattern_counts"][label] += 1
            if field_matches:
                entry_matches[field] = field_matches
        if entry_matches:
            results["affected_entries"].append({
                "entry_idx": i,
                "product": entry.get("product", ""),
                "defect_name": entry.get("defect_name", ""),
                "filename": entry.get("filename", ""),
                "caption_short": entry.get("caption_short", ""),
                "caption_long": entry.get("caption_long", ""),
                "matches": entry_matches,
            })

    return results


def analyze_short_captions(data: list[dict]) -> dict:
    """Find empty or suspiciously short captions."""
    results = {"empty": [], "short": []}
    fields = ["caption_short", "caption_long"]

    for i, entry in enumerate(data):
        for field in fields:
            text = entry.get(field, "")
            if not text or text.strip() == "":
                results["empty"].append({
                    "entry_idx": i,
                    "product": entry.get("product", ""),
                    "defect_name": entry.get("defect_name", ""),
                    "filename": entry.get("filename", ""),
                    "field": field,
                    "text": repr(text),
                })
            elif len(text.strip()) < 10:
                results["short"].append({
                    "entry_idx": i,
                    "product": entry.get("product", ""),
                    "defect_name": entry.get("defect_name", ""),
                    "filename": entry.get("filename", ""),
                    "field": field,
                    "text": text,
                    "length": len(text.strip()),
                })

    return results


def analyze_format(data: list[dict]) -> dict:
    """Check raw_response for proper SHORT:/LONG: format."""
    results = {"missing_short": [], "missing_long": [], "malformed": []}

    for i, entry in enumerate(data):
        raw = entry.get("raw_response", "")
        if not raw:
            results["malformed"].append({
                "entry_idx": i,
                "product": entry.get("product", ""),
                "defect_name": entry.get("defect_name", ""),
                "filename": entry.get("filename", ""),
                "issue": "empty raw_response",
                "raw_response": repr(raw),
            })
            continue

        has_short = bool(re.search(r'\bSHORT\s*:', raw))
        has_long = bool(re.search(r'\bLONG\s*:', raw))

        if not has_short:
            results["missing_short"].append({
                "entry_idx": i,
                "product": entry.get("product", ""),
                "defect_name": entry.get("defect_name", ""),
                "filename": entry.get("filename", ""),
                "raw_response_start": raw[:200],
            })
        if not has_long:
            results["missing_long"].append({
                "entry_idx": i,
                "product": entry.get("product", ""),
                "defect_name": entry.get("defect_name", ""),
                "filename": entry.get("filename", ""),
                "raw_response_start": raw[:200],
            })

    return results


def analyze_duplicates(data: list[dict]) -> dict:
    """Find exact duplicate captions."""
    results = {"caption_short_dupes": defaultdict(list), "caption_long_dupes": defaultdict(list)}

    for i, entry in enumerate(data):
        cs = entry.get("caption_short", "")
        cl = entry.get("caption_long", "")
        if cs:
            results["caption_short_dupes"][cs].append({
                "entry_idx": i,
                "product": entry.get("product", ""),
                "defect_name": entry.get("defect_name", ""),
                "filename": entry.get("filename", ""),
            })
        if cl:
            results["caption_long_dupes"][cl].append({
                "entry_idx": i,
                "product": entry.get("product", ""),
                "defect_name": entry.get("defect_name", ""),
                "filename": entry.get("filename", ""),
            })

    # Filter to only actual duplicates (>1 occurrence)
    results["caption_short_dupes"] = {k: v for k, v in results["caption_short_dupes"].items() if len(v) > 1}
    results["caption_long_dupes"] = {k: v for k, v in results["caption_long_dupes"].items() if len(v) > 1}

    return results


def analyze_long_captions(data: list[dict]) -> dict:
    """Find captions that exceed word budgets."""
    results = {"short_over_100": [], "long_over_250": []}

    for i, entry in enumerate(data):
        cs = entry.get("caption_short", "")
        cl = entry.get("caption_long", "")

        cs_words = len(cs.split()) if cs else 0
        cl_words = len(cl.split()) if cl else 0

        if cs_words > 100:
            results["short_over_100"].append({
                "entry_idx": i,
                "product": entry.get("product", ""),
                "defect_name": entry.get("defect_name", ""),
                "filename": entry.get("filename", ""),
                "word_count": cs_words,
                "caption": cs,
            })
        if cl_words > 250:
            results["long_over_250"].append({
                "entry_idx": i,
                "product": entry.get("product", ""),
                "defect_name": entry.get("defect_name", ""),
                "filename": entry.get("filename", ""),
                "word_count": cl_words,
                "caption": cl,
            })

    return results


def analyze_self_referential(data: list[dict]) -> dict:
    """Find captions with self-referential language."""
    patterns = [
        (r'\bimage\s+1\b', "image 1"),
        (r'\bimage\s+2\b', "image 2"),
        (r'\bimage\s+3\b', "image 3"),
        (r'\bfirst\s+image\b', "first image"),
        (r'\bsecond\s+image\b', "second image"),
        (r'\bthird\s+image\b', "third image"),
        (r'\b(?:the|this)\s+crop\b', "the/this crop"),
        (r'\bclose[\s-]up\b', "close-up"),
        (r'\bzoomed[\s-]in\b', "zoomed in"),
        (r'\bmagnified\b', "magnified"),
        (r'\b(?:the|this)\s+photo\b', "the/this photo"),
        (r'\b(?:the|this)\s+photograph\b', "the/this photograph"),
        (r'\b(?:the|this)\s+picture\b', "the/this picture"),
        (r'\bin\s+the\s+image\b', "in the image"),
        (r'\bin\s+this\s+image\b', "in this image"),
        (r'\bthe\s+image\s+shows\b', "the image shows"),
        (r'\bas\s+shown\b', "as shown"),
        (r'\bvisible\s+in\s+the\b', "visible in the"),
        (r'\bleft\s+image\b', "left image"),
        (r'\bright\s+image\b', "right image"),
        (r'\b(?:the|this)\s+view\b', "the/this view"),
        (r'\bfull[\s-]resolution\b', "full-resolution"),
        (r'\breference\s+image\b', "reference image"),
    ]

    results = {"affected_entries": [], "pattern_counts": Counter()}
    fields = ["caption_short", "caption_long"]

    for i, entry in enumerate(data):
        entry_matches = {}
        for field in fields:
            text = entry.get(field, "")
            if not text:
                continue
            field_matches = []
            for pattern, label in patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                if matches:
                    for m in matches:
                        start = max(0, m.start() - 40)
                        end = min(len(text), m.end() + 40)
                        context = text[start:end]
                        field_matches.append({
                            "pattern": label,
                            "match": m.group(),
                            "context": context,
                        })
                        results["pattern_counts"][label] += 1
            if field_matches:
                entry_matches[field] = field_matches
        if entry_matches:
            results["affected_entries"].append({
                "entry_idx": i,
                "product": entry.get("product", ""),
                "defect_name": entry.get("defect_name", ""),
                "filename": entry.get("filename", ""),
                "caption_short": entry.get("caption_short", ""),
                "caption_long": entry.get("caption_long", ""),
                "matches": entry_matches,
            })

    return results


def analyze_prompt_leakage(data: list[dict]) -> dict:
    """Find captions that contain fragments of the system prompt."""
    prompt_fragments = [
        r'\bshort\s*:\s*\d+',
        r'\blong\s*:\s*\d+',
        r'\bword\s+budget\b',
        r'\btoken\s+budget\b',
        r'\bsystem\s+prompt\b',
        r'\byou\s+are\s+a\b',
        r'\byour\s+task\s+is\b',
        r'\bdescribe\s+the\s+defect\b',
        r'\bprovide\s+a\s+caption\b',
        r'\bgenerate\s+a\s+caption\b',
        r'\bSHORT\s*:',
        r'\bLONG\s*:',
        r'\binstruction\b',
        r'\buser\s*:\b',
        r'\bassistant\s*:\b',
        r'\b<\|',  # special tokens
        r'\b\[INST\]',
        r'\b\[/INST\]',
    ]

    results = {"affected_entries": [], "pattern_counts": Counter()}
    fields = ["caption_short", "caption_long"]

    for i, entry in enumerate(data):
        entry_matches = {}
        for field in fields:
            text = entry.get(field, "")
            if not text:
                continue
            field_matches = []
            for pattern in prompt_fragments:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                if matches:
                    for m in matches:
                        start = max(0, m.start() - 30)
                        end = min(len(text), m.end() + 30)
                        context = text[start:end]
                        field_matches.append({
                            "pattern": pattern,
                            "match": m.group(),
                            "context": context,
                        })
                        results["pattern_counts"][pattern] += 1
            if field_matches:
                entry_matches[field] = field_matches
        if entry_matches:
            results["affected_entries"].append({
                "entry_idx": i,
                "product": entry.get("product", ""),
                "defect_name": entry.get("defect_name", ""),
                "filename": entry.get("filename", ""),
                "caption_short": entry.get("caption_short", ""),
                "caption_long": entry.get("caption_long", ""),
                "matches": entry_matches,
            })

    return results


def write_report(
    output_path: str,
    non_ascii: dict,
    bbox: dict,
    cjk: dict,
    measurements: dict,
    short_caps: dict,
    format_issues: dict,
    duplicates: dict,
    long_caps: dict,
    self_ref: dict,
    prompt_leak: dict,
    total: int,
) -> None:
    """Write the full detailed report to file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("CAPTION POST-PROCESSING ANALYSIS REPORT\n")
        f.write(f"Total entries analyzed: {total}\n")
        f.write("=" * 80 + "\n\n")

        # 1. Non-ASCII
        f.write("-" * 80 + "\n")
        f.write("1. NON-ASCII CHARACTERS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total entries with non-ASCII: {len(non_ascii['affected_entries'])}\n\n")

        f.write("Character frequency table:\n")
        for (ch, code, name), count in sorted(non_ascii["char_counts"].items(), key=lambda x: -x[1]):
            f.write(f"  U+{code:04X} ({name}): {count} occurrences\n")
            examples = non_ascii["per_char_examples"][(ch, code, name)]
            for ex in examples[:3]:
                f.write(f"    Example [{ex['product']}/{ex['defect_name']}] in {ex['field']}: ...{ex['context']}...\n")
        f.write("\n")

        f.write("ALL affected entries:\n")
        for entry in non_ascii["affected_entries"]:
            f.write(f"  [{entry['entry_idx']}] {entry['product']}/{entry['defect_name']} - {entry['filename']}\n")
            for field, chars in entry["chars"].items():
                unique_chars = set((ch, code, name) for ch, code, name in chars)
                for ch, code, name in unique_chars:
                    count_in_field = sum(1 for c, _, _ in chars if c == ch)
                    f.write(f"    {field}: U+{code:04X} ({name}) x{count_in_field}\n")
        f.write("\n")

        # 2. Bbox leaks
        f.write("-" * 80 + "\n")
        f.write("2. BOUNDING BOX / ANNOTATION LEAKS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total entries with bbox leaks: {len(bbox['affected_entries'])}\n\n")

        f.write("Pattern frequency:\n")
        for pattern, count in bbox["pattern_counts"].most_common():
            f.write(f"  '{pattern}': {count}\n")
        f.write("\n")

        f.write("ALL affected entries:\n")
        for entry in bbox["affected_entries"]:
            f.write(f"  [{entry['entry_idx']}] {entry['product']}/{entry['defect_name']} - {entry['filename']}\n")
            f.write(f"    SHORT: {entry['caption_short']}\n")
            f.write(f"    LONG:  {entry['caption_long']}\n")
            for field, matches in entry["matches"].items():
                for m in matches:
                    f.write(f"    >> {field}: matched '{m['pattern']}' -> \"{m['match']}\" in: ...{m['context']}...\n")
            f.write("\n")

        # 3. CJK
        f.write("-" * 80 + "\n")
        f.write("3. CHINESE/CJK CHARACTERS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total entries with CJK: {len(cjk['affected_entries'])}\n\n")
        for entry in cjk["affected_entries"]:
            f.write(f"  [{entry['entry_idx']}] {entry['product']}/{entry['defect_name']} - {entry['filename']}\n")
            for field, chars in entry["chars"].items():
                f.write(f"    {field}: {''.join(chars)}\n")
        f.write("\n")

        # 4. Measurements
        f.write("-" * 80 + "\n")
        f.write("4. MEASUREMENT LEAKS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total entries with measurements: {len(measurements['affected_entries'])}\n\n")

        f.write("Pattern frequency:\n")
        for pattern, count in measurements["pattern_counts"].most_common():
            f.write(f"  '{pattern}': {count}\n")
        f.write("\n")

        f.write("ALL affected entries:\n")
        for entry in measurements["affected_entries"]:
            f.write(f"  [{entry['entry_idx']}] {entry['product']}/{entry['defect_name']} - {entry['filename']}\n")
            for field, matches in entry["matches"].items():
                for m in matches:
                    f.write(f"    >> {field}: '{m['pattern']}' -> \"{m['match']}\" in: ...{m['context']}...\n")
            f.write("\n")

        # 5. Short captions
        f.write("-" * 80 + "\n")
        f.write("5. EMPTY OR SUSPICIOUSLY SHORT CAPTIONS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Empty: {len(short_caps['empty'])}, Short (<10 chars): {len(short_caps['short'])}\n\n")

        if short_caps["empty"]:
            f.write("Empty captions:\n")
            for entry in short_caps["empty"]:
                f.write(f"  [{entry['entry_idx']}] {entry['product']}/{entry['defect_name']} - {entry['filename']} - {entry['field']}: {entry['text']}\n")
            f.write("\n")

        if short_caps["short"]:
            f.write("Short captions (<10 chars):\n")
            for entry in short_caps["short"]:
                f.write(f"  [{entry['entry_idx']}] {entry['product']}/{entry['defect_name']} - {entry['filename']} - {entry['field']}: \"{entry['text']}\" ({entry['length']} chars)\n")
            f.write("\n")

        # 6. Format issues
        f.write("-" * 80 + "\n")
        f.write("6. MISSING SHORT:/LONG: FORMAT IN RAW_RESPONSE\n")
        f.write("-" * 80 + "\n")
        f.write(f"Missing SHORT: {len(format_issues['missing_short'])}\n")
        f.write(f"Missing LONG: {len(format_issues['missing_long'])}\n")
        f.write(f"Malformed (empty): {len(format_issues['malformed'])}\n\n")

        if format_issues["malformed"]:
            f.write("Malformed entries:\n")
            for entry in format_issues["malformed"]:
                f.write(f"  [{entry['entry_idx']}] {entry['product']}/{entry['defect_name']} - {entry['filename']}: {entry['issue']}\n")
            f.write("\n")

        if format_issues["missing_short"]:
            f.write("Missing SHORT: prefix:\n")
            for entry in format_issues["missing_short"]:
                f.write(f"  [{entry['entry_idx']}] {entry['product']}/{entry['defect_name']} - {entry['filename']}\n")
                f.write(f"    raw_response start: {entry['raw_response_start']}\n")
            f.write("\n")

        if format_issues["missing_long"]:
            f.write("Missing LONG: prefix:\n")
            for entry in format_issues["missing_long"]:
                f.write(f"  [{entry['entry_idx']}] {entry['product']}/{entry['defect_name']} - {entry['filename']}\n")
                f.write(f"    raw_response start: {entry['raw_response_start']}\n")
            f.write("\n")

        # 7. Duplicates
        f.write("-" * 80 + "\n")
        f.write("7. DUPLICATE CAPTIONS\n")
        f.write("-" * 80 + "\n")
        n_short_dupes = sum(len(v) for v in duplicates["caption_short_dupes"].values())
        n_long_dupes = sum(len(v) for v in duplicates["caption_long_dupes"].values())
        f.write(f"Duplicate caption_short groups: {len(duplicates['caption_short_dupes'])} (affecting {n_short_dupes} entries)\n")
        f.write(f"Duplicate caption_long groups: {len(duplicates['caption_long_dupes'])} (affecting {n_long_dupes} entries)\n\n")

        if duplicates["caption_short_dupes"]:
            f.write("Duplicate caption_short:\n")
            for caption, entries in sorted(duplicates["caption_short_dupes"].items(), key=lambda x: -len(x[1])):
                f.write(f"  [{len(entries)}x] \"{caption[:120]}...\"\n" if len(caption) > 120 else f"  [{len(entries)}x] \"{caption}\"\n")
                for e in entries:
                    f.write(f"    - [{e['entry_idx']}] {e['product']}/{e['defect_name']} - {e['filename']}\n")
            f.write("\n")

        if duplicates["caption_long_dupes"]:
            f.write("Duplicate caption_long:\n")
            for caption, entries in sorted(duplicates["caption_long_dupes"].items(), key=lambda x: -len(x[1])):
                f.write(f"  [{len(entries)}x] \"{caption[:120]}...\"\n" if len(caption) > 120 else f"  [{len(entries)}x] \"{caption}\"\n")
                for e in entries:
                    f.write(f"    - [{e['entry_idx']}] {e['product']}/{e['defect_name']} - {e['filename']}\n")
            f.write("\n")

        # 8. Long captions
        f.write("-" * 80 + "\n")
        f.write("8. VERY LONG CAPTIONS\n")
        f.write("-" * 80 + "\n")
        f.write(f"caption_short > 100 words: {len(long_caps['short_over_100'])}\n")
        f.write(f"caption_long > 250 words: {len(long_caps['long_over_250'])}\n\n")

        if long_caps["short_over_100"]:
            f.write("caption_short over 100 words:\n")
            for entry in long_caps["short_over_100"]:
                f.write(f"  [{entry['entry_idx']}] {entry['product']}/{entry['defect_name']} ({entry['word_count']} words)\n")
                f.write(f"    {entry['caption']}\n\n")

        if long_caps["long_over_250"]:
            f.write("caption_long over 250 words:\n")
            for entry in long_caps["long_over_250"]:
                f.write(f"  [{entry['entry_idx']}] {entry['product']}/{entry['defect_name']} ({entry['word_count']} words)\n")
                f.write(f"    {entry['caption']}\n\n")

        # 9. Self-referential
        f.write("-" * 80 + "\n")
        f.write("9. SELF-REFERENTIAL LANGUAGE\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total entries with self-referential language: {len(self_ref['affected_entries'])}\n\n")

        f.write("Pattern frequency:\n")
        for pattern, count in self_ref["pattern_counts"].most_common():
            f.write(f"  '{pattern}': {count}\n")
        f.write("\n")

        f.write("ALL affected entries:\n")
        for entry in self_ref["affected_entries"]:
            f.write(f"  [{entry['entry_idx']}] {entry['product']}/{entry['defect_name']} - {entry['filename']}\n")
            for field, matches in entry["matches"].items():
                for m in matches:
                    f.write(f"    >> {field}: '{m['pattern']}' -> \"{m['match']}\" in: ...{m['context']}...\n")
            f.write("\n")

        # 10. Prompt leakage
        f.write("-" * 80 + "\n")
        f.write("10. PROMPT LEAKAGE\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total entries with prompt leakage: {len(prompt_leak['affected_entries'])}\n\n")

        f.write("Pattern frequency:\n")
        for pattern, count in prompt_leak["pattern_counts"].most_common():
            f.write(f"  '{pattern}': {count}\n")
        f.write("\n")

        f.write("ALL affected entries:\n")
        for entry in prompt_leak["affected_entries"]:
            f.write(f"  [{entry['entry_idx']}] {entry['product']}/{entry['defect_name']} - {entry['filename']}\n")
            for field, matches in entry["matches"].items():
                for m in matches:
                    f.write(f"    >> {field}: '{m['pattern']}' -> \"{m['match']}\" in: ...{m['context']}...\n")
            f.write("\n")

    print(f"\nFull report saved to: {output_path}")


def print_summary(
    non_ascii: dict,
    bbox: dict,
    cjk: dict,
    measurements: dict,
    short_caps: dict,
    format_issues: dict,
    duplicates: dict,
    long_caps: dict,
    self_ref: dict,
    prompt_leak: dict,
    total: int,
) -> None:
    """Print summary table to stdout."""
    print("\n" + "=" * 90)
    print("CAPTION POST-PROCESSING ANALYSIS - SUMMARY")
    print(f"Total entries: {total}")
    print("=" * 90)
    print(f"{'#':<4} {'Category':<45} {'Affected Entries':<18} {'Severity':<10}")
    print("-" * 90)

    n_short_dupes = len(duplicates["caption_short_dupes"])
    n_long_dupes = len(duplicates["caption_long_dupes"])

    rows = [
        ("1", "Non-ASCII characters", len(non_ascii["affected_entries"]), "MEDIUM"),
        ("2", "Bbox/annotation leaks", len(bbox["affected_entries"]), "HIGH"),
        ("3", "Chinese/CJK characters", len(cjk["affected_entries"]), "HIGH"),
        ("4", "Measurement leaks", len(measurements["affected_entries"]), "LOW"),
        ("5a", "Empty captions", len(short_caps["empty"]), "HIGH"),
        ("5b", "Very short captions (<10 chars)", len(short_caps["short"]), "HIGH"),
        ("6a", "Missing SHORT: in raw_response", len(format_issues["missing_short"]), "MEDIUM"),
        ("6b", "Missing LONG: in raw_response", len(format_issues["missing_long"]), "MEDIUM"),
        ("6c", "Malformed raw_response", len(format_issues["malformed"]), "HIGH"),
        ("7a", f"Duplicate caption_short groups", n_short_dupes, "MEDIUM"),
        ("7b", f"Duplicate caption_long groups", n_long_dupes, "MEDIUM"),
        ("8a", "caption_short > 100 words", len(long_caps["short_over_100"]), "LOW"),
        ("8b", "caption_long > 250 words", len(long_caps["long_over_250"]), "LOW"),
        ("9", "Self-referential language", len(self_ref["affected_entries"]), "MEDIUM"),
        ("10", "Prompt leakage", len(prompt_leak["affected_entries"]), "HIGH"),
    ]

    for num, cat, count, severity in rows:
        print(f"{num:<4} {cat:<45} {count:<18} {severity:<10}")

    print("=" * 90)

    # Non-ASCII detail breakdown
    print("\n--- Non-ASCII Character Breakdown ---")
    for (ch, code, name), count in sorted(non_ascii["char_counts"].items(), key=lambda x: -x[1]):
        # Count is across all fields including raw_response; show it
        print(f"  U+{code:04X}  {name:<40} {count:>6} occurrences")

    # Bbox detail
    if bbox["pattern_counts"]:
        print("\n--- Bbox Leak Pattern Breakdown ---")
        for pattern, count in bbox["pattern_counts"].most_common():
            print(f"  '{pattern}':{' ' * (25 - len(pattern))} {count:>6}")

    # Self-ref detail
    if self_ref["pattern_counts"]:
        print("\n--- Self-Referential Pattern Breakdown ---")
        for pattern, count in self_ref["pattern_counts"].most_common():
            print(f"  '{pattern}':{' ' * (25 - len(pattern))} {count:>6}")

    # Duplicate detail
    if n_short_dupes > 0:
        print(f"\n--- Top Duplicate caption_short (showing top 5) ---")
        for caption, entries in sorted(duplicates["caption_short_dupes"].items(), key=lambda x: -len(x[1]))[:5]:
            trunc = caption[:80] + "..." if len(caption) > 80 else caption
            products = set(e["product"] for e in entries)
            print(f"  [{len(entries)}x] products={products}")
            print(f"        \"{trunc}\"")

    if n_long_dupes > 0:
        print(f"\n--- Top Duplicate caption_long (showing top 5) ---")
        for caption, entries in sorted(duplicates["caption_long_dupes"].items(), key=lambda x: -len(x[1]))[:5]:
            trunc = caption[:80] + "..." if len(caption) > 80 else caption
            products = set(e["product"] for e in entries)
            print(f"  [{len(entries)}x] products={products}")
            print(f"        \"{trunc}\"")

    # Examples for key categories
    print("\n--- Representative Examples ---")

    if bbox["affected_entries"]:
        print("\n  BBOX LEAKS (first 5):")
        for entry in bbox["affected_entries"][:5]:
            print(f"    [{entry['entry_idx']}] {entry['product']}/{entry['defect_name']}")
            for field, matches in entry["matches"].items():
                for m in matches:
                    print(f"      {field}: \"{m['match']}\" in: ...{m['context']}...")

    if prompt_leak["affected_entries"]:
        print("\n  PROMPT LEAKAGE (first 5):")
        for entry in prompt_leak["affected_entries"][:5]:
            print(f"    [{entry['entry_idx']}] {entry['product']}/{entry['defect_name']}")
            for field, matches in entry["matches"].items():
                for m in matches:
                    print(f"      {field}: \"{m['match']}\" in: ...{m['context']}...")

    if measurements["affected_entries"]:
        print("\n  MEASUREMENT LEAKS (first 5):")
        for entry in measurements["affected_entries"][:5]:
            print(f"    [{entry['entry_idx']}] {entry['product']}/{entry['defect_name']}")
            for field, matches in entry["matches"].items():
                for m in matches:
                    print(f"      {field}: \"{m['match']}\" in: ...{m['context']}...")

    # Word count distribution
    print("\n--- Word Count Distribution ---")
    short_words = []
    long_words = []
    for entry in data:
        cs = entry.get("caption_short", "")
        cl = entry.get("caption_long", "")
        if cs:
            short_words.append(len(cs.split()))
        if cl:
            long_words.append(len(cl.split()))

    if short_words:
        import statistics
        print(f"  caption_short: min={min(short_words)}, max={max(short_words)}, "
              f"mean={statistics.mean(short_words):.1f}, median={statistics.median(short_words):.0f}, "
              f"p95={sorted(short_words)[int(len(short_words)*0.95)]}")
    if long_words:
        print(f"  caption_long:  min={min(long_words)}, max={max(long_words)}, "
              f"mean={statistics.mean(long_words):.1f}, median={statistics.median(long_words):.0f}, "
              f"p95={sorted(long_words)[int(len(long_words)*0.95)]}")


if __name__ == "__main__":
    import sys
    import time

    captions_path = r"C:\Users\frede\Desktop\kandidat\speciale\anomverse_extension\datasets\realiad_1024\captions_nano.json"
    output_path = r"C:\Users\frede\Desktop\kandidat\speciale\results\caption_postprocess_analysis.txt"

    print(f"Loading captions from: {captions_path}")
    data = load_captions(captions_path)
    total = len(data)

    t0 = time.time()

    print("Analyzing non-ASCII characters...")
    non_ascii = analyze_non_ascii(data)

    print("Analyzing bbox/annotation leaks...")
    bbox = analyze_bbox_leaks(data)

    print("Analyzing CJK characters...")
    cjk = analyze_cjk(data)

    print("Analyzing measurement leaks...")
    measurements = analyze_measurements(data)

    print("Analyzing short/empty captions...")
    short_caps = analyze_short_captions(data)

    print("Analyzing raw_response format...")
    format_issues = analyze_format(data)

    print("Analyzing duplicate captions...")
    duplicates = analyze_duplicates(data)

    print("Analyzing long captions...")
    long_caps = analyze_long_captions(data)

    print("Analyzing self-referential language...")
    self_ref = analyze_self_referential(data)

    print("Analyzing prompt leakage...")
    prompt_leak = analyze_prompt_leakage(data)

    elapsed = time.time() - t0
    print(f"\nAnalysis complete in {elapsed:.1f}s")

    # Write full report
    write_report(
        output_path, non_ascii, bbox, cjk, measurements,
        short_caps, format_issues, duplicates, long_caps,
        self_ref, prompt_leak, total,
    )

    # Print summary
    print_summary(
        non_ascii, bbox, cjk, measurements,
        short_caps, format_issues, duplicates, long_caps,
        self_ref, prompt_leak, total,
    )

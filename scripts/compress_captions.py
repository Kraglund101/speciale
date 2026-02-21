"""Compress captions that exceed 77 CLIP tokens by removing filler phrases.

The captions follow a consistent template from the same model, so rule-based
compression works well. Preserves all defect details, locations, and descriptions
while cutting formulaic filler.
"""
import json
import re
from transformers import CLIPTokenizer


def count_tokens(text: str, tokenizer: CLIPTokenizer) -> int:
    return len(tokenizer.encode(text))


def compress_caption(text: str, tokenizer: CLIPTokenizer, target: int = 77) -> str:
    """Progressively compress a caption until it fits within target CLIP tokens.

    Applies increasingly aggressive compression rules, checking token count
    after each pass. Preserves all defect information and descriptions.
    """
    if count_tokens(text, tokenizer) <= target:
        return text

    # Pass 1: Remove heavy filler phrases (least information loss)
    rules_pass1 = [
        # "The image depicts a X, with" -> "X with"
        (r"^The image depicts (?:a |an |the )?(.+?),?\s*with ", r"\1 with "),
        # "The image also shows" -> "Also,"
        (r"The image also shows (?:a |an |the )?", "Also, "),
        # "The defect is characterized by" -> filler
        (r"The defect is characterized by ", "It shows "),
        # "It exhibits a/an" -> "showing"
        (r"(?:,\s*)?(?:and )?[Ii]t exhibits (?:a |an )?", ", showing "),
        # "The notable features include" -> cut
        (r"\.\s*The notable features? (?:include|are|is) .+$", "."),
        # "indicating a disruption/potential/possible" trailing clauses
        (r",\s*indicating (?:a )?(?:potential |possible )?(?:disruption|loss|break|flaw|impairment).+?\.", "."),
        # "likely to affect the functionality" trailing
        (r"\.\s*The (?:bent|damaged|defective).+?(?:likely|may|could).+?(?:functionality|performance|operation).+?\.", "."),
        # Remove "observed"
        (r" defect observed ", " defect "),
        (r" observed ", " "),
    ]

    result = text.strip()
    for pattern, replacement in rules_pass1:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    result = result.strip()
    if result and not result.endswith("."):
        # Try to end at last period
        last_period = result.rfind(".")
        if last_period > len(result) * 0.4:
            result = result[:last_period + 1]

    if count_tokens(result, tokenizer) <= target:
        return result

    # Pass 2: More aggressive compression
    rules_pass2 = [
        # "It shows" -> remove if we can
        (r"\.\s*It shows ", ". Shows "),
        # "compared to the surrounding" trailing
        (r",?\s*compared to the (?:surrounding|adjacent|neighboring).+?\.", "."),
        # "making it stand out" trailing
        (r",?\s*making it (?:stand out|visible|noticeable).+?\.", "."),
        # "which appears to be" -> ""
        (r",?\s*which appears to be ", ", appearing "),
        # "as well as" trailing clauses
        (r",?\s*as well as .+?\.", "."),
        # Double spaces
        (r"  +", " "),
    ]

    for pattern, replacement in rules_pass2:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    result = result.strip()

    if count_tokens(result, tokenizer) <= target:
        return result

    # Pass 3: Sentence truncation from the end, preserving all defect mentions
    sentences = re.split(r"(?<=\.)\s+", result)
    while len(sentences) > 1 and count_tokens(" ".join(sentences), tokenizer) > target:
        sentences.pop()
    result = " ".join(sentences)

    if count_tokens(result, tokenizer) <= target:
        return result

    # Pass 4: If single long sentence, truncate at last clause boundary
    # Try comma boundaries
    parts = result.split(", ")
    while len(parts) > 1 and count_tokens(", ".join(parts), tokenizer) > target:
        parts.pop()
    result = ", ".join(parts)
    if not result.endswith("."):
        result = result.rstrip(",").strip() + "."

    if count_tokens(result, tokenizer) <= target:
        return result

    # Pass 5: Hard token truncation as last resort
    tokens = tokenizer.encode(result)
    truncated = tokenizer.decode(tokens[:76], skip_special_tokens=True).strip()
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated.rstrip(",.;:") + "."


def main() -> None:
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    json_path = "C:/Users/frede/Desktop/kandidat/speciale/anomverse_extension/datasets/combined_datasets.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rewritten = 0
    still_over = 0
    token_savings = 0
    pass_stats = {"already_ok": 0, "pass1": 0, "pass2": 0, "pass3": 0, "pass4": 0, "pass5": 0}

    for sample in data["samples"]:
        caption = sample.get("caption", "")
        if not caption:
            sample["caption_77"] = ""
            continue

        orig_tokens = count_tokens(caption, tokenizer)
        if orig_tokens <= 77:
            sample["caption_77"] = caption
            pass_stats["already_ok"] += 1
            continue

        compressed = compress_caption(caption, tokenizer, target=77)
        new_tokens = count_tokens(compressed, tokenizer)
        sample["caption_77"] = compressed
        rewritten += 1
        token_savings += orig_tokens - new_tokens

        if new_tokens > 77:
            still_over += 1

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Rewritten: {rewritten}")
    print(f"Still over 77: {still_over}")
    print(f"Total CLIP tokens saved: {token_savings}")
    print(f"Avg tokens saved per rewrite: {token_savings / rewritten:.1f}")

    # Show examples
    print("\n=== EXAMPLES ===")
    count = 0
    for sample in data["samples"]:
        cap = sample.get("caption", "")
        cap77 = sample.get("caption_77", "")
        if cap and cap77 and cap != cap77:
            orig_n = count_tokens(cap, tokenizer)
            new_n = count_tokens(cap77, tokenizer)
            if orig_n > 100:  # Show a heavy example
                print(f"\n[{sample['dataset']}/{sample['product']}/{sample['defect_type']}]")
                print(f"ORIGINAL ({orig_n} tokens):")
                print(f"  {cap}")
                print(f"COMPRESSED ({new_n} tokens):")
                print(f"  {cap77}")
                count += 1
                if count >= 5:
                    break


if __name__ == "__main__":
    main()

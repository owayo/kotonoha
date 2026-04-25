"""Convert Kanjium accents.txt to kotonoha-format CSV.

Input format: lemma\\treading_hiragana\\taccent_types (comma-separated for multiple)
Output format: lemma,reading_katakana,accent_type
"""

from __future__ import annotations

import csv
from pathlib import Path

# Hiragana to Katakana mapping
def _hira_to_kata(s: str) -> str:
    out = []
    for ch in s:
        c = ord(ch)
        # Hiragana range 0x3041-0x3096
        if 0x3041 <= c <= 0x3096:
            out.append(chr(c + 0x60))  # Shift to katakana
        elif ch == "ゔ":
            out.append("ヴ")
        else:
            out.append(ch)
    return "".join(out)


def main() -> None:
    src = Path("/mnt/c/GitHub/kotonoha-training-data/raw/kanjium_accents.txt")
    dst = Path("/mnt/c/GitHub/kotonoha-training-data/train/accent_dict_kanjium.csv")
    dst.parent.mkdir(parents=True, exist_ok=True)

    seen: dict[tuple[str, str], str] = {}
    skipped = 0
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                skipped += 1
                continue
            lemma, reading_hira, accent_str = parts
            reading_kata = _hira_to_kata(reading_hira.strip())
            # Take first accent if multiple (prefer canonical)
            first = accent_str.split(",")[0].strip()
            if not first.isdigit():
                skipped += 1
                continue
            accent = int(first)
            if accent < 0 or accent > 20:
                skipped += 1
                continue
            seen[(lemma, reading_kata)] = str(accent)

    with open(dst, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lemma", "reading", "accent_type"])
        for (lemma, reading), accent in seen.items():
            writer.writerow([lemma, reading, accent])

    print(f"Wrote {len(seen)} entries (skipped {skipped})")
    print(f"Saved to {dst}")


if __name__ == "__main__":
    main()

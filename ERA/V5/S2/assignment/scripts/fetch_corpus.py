"""Fetch the India Wikipedia article's plain-text extract in English, Hindi,
Telugu, and Marathi, saving each to data/corpus/{lang}.txt.

Localized article titles are discovered via English Wikipedia's langlinks
rather than hardcoded, since article titles differ per language and langlinks
are the authoritative cross-language mapping.
"""

import json
import pathlib
import sys

import requests

LANGS = ["en", "hi", "te", "mr"]
EN_TITLE = "India"
OUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "corpus"
USER_AGENT = "S2-assignment-bpe-tokenizer/1.0 (educational use)"


def api_get(lang: str, params: dict) -> dict:
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {**params, "format": "json"}
    resp = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_localized_titles() -> dict[str, str]:
    """Return {lang: title} for hi/te/mr, discovered via English langlinks."""
    data = api_get(
        "en",
        {
            "action": "query",
            "titles": EN_TITLE,
            "prop": "langlinks",
            "lllimit": 500,
            "redirects": 1,
        },
    )
    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    langlinks = page.get("langlinks", [])
    titles = {"en": EN_TITLE}
    wanted = {l for l in LANGS if l != "en"}
    for link in langlinks:
        if link["lang"] in wanted:
            titles[link["lang"]] = link["*"]
    missing = wanted - titles.keys()
    if missing:
        sys.exit(f"Could not find langlinks for: {missing}")
    return titles


def fetch_extract(lang: str, title: str) -> str:
    data = api_get(
        lang,
        {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "explaintext": 1,
            "redirects": 1,
        },
    )
    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    if "missing" in page:
        sys.exit(f"[{lang}] page '{title}' not found")
    extract = page.get("extract", "")
    if not extract.strip():
        sys.exit(f"[{lang}] page '{title}' had an empty extract")
    return extract


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    titles = get_localized_titles()
    print("Resolved titles:", json.dumps(titles, ensure_ascii=False))

    for lang in LANGS:
        title = titles[lang]
        text = fetch_extract(lang, title)
        out_path = OUT_DIR / f"{lang}.txt"
        out_path.write_text(text, encoding="utf-8")
        word_count = len(text.split())
        print(f"[{lang}] '{title}' -> {out_path} ({len(text)} chars, ~{word_count} words)")


if __name__ == "__main__":
    main()

"""Fetch the India Wikipedia article's *faithful* Markdown rendering in
English, Hindi, Telugu, Marathi, and Bengali, saving each to
data/corpus/{lang}.txt (+ .meta.json).

"Faithful" means the HTML-to-Markdown conversion keeps visible article
content that a plain `explaintext` extract throws away: links (with their
URLs), references, tables, headers, image captions, navboxes. This matters
because the tokenizer's vocabulary needs to actually cover the punctuation
classes (`/`, `#`, `:`, `-`, `|`, `[`, `]`) that markdown/URL-shaped text is
built from -- a plain-prose corpus never teaches the tokenizer those
characters, which is how the original submission's tokenizer ended up
unable to round-trip a URL faithfully.

Localized article titles for hi/te/mr/bn are discovered via English
Wikipedia's langlinks rather than hardcoded, since titles differ per
language and langlinks are the authoritative cross-language mapping.
"""

import json
import re
import sys
import time
import pathlib
from urllib.parse import quote, urljoin

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import regex

LANGS = ["en", "hi", "te", "mr", "bn"]
EN_TITLE = "India"
OUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "corpus"
USER_AGENT = "S2-assignment-bpe-tokenizer/2.0 (educational use)"

WORD_PATTERN = regex.compile(r"[\p{L}\p{M}\p{N}]+")


def api_get(lang: str, params: dict) -> dict:
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {**params, "format": "json"}
    resp = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_localized_titles() -> dict[str, str]:
    """Return {lang: title} for hi/te/mr/bn, discovered via English langlinks."""
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


def absolutize_links(soup: BeautifulSoup, lang: str) -> None:
    base = f"https://{lang}.wikipedia.org/wiki/"
    for tag in soup.find_all(["a", "img", "source"]):
        attr = "href" if tag.name == "a" else "src"
        value = tag.get(attr)
        if not value:
            continue
        if value.startswith("//"):
            tag[attr] = "https:" + value
        elif value.startswith("./"):
            tag[attr] = urljoin(base, value[2:])
        elif value.startswith("/"):
            tag[attr] = urljoin(f"https://{lang}.wikipedia.org", value)


def strip_only_technical_noise(node: BeautifulSoup, soup: BeautifulSoup) -> None:
    for tag in node(["script", "style", "meta"]):
        tag.decompose()
    for tag in node.find_all("link"):
        rel = " ".join(tag.get("rel") or [])
        href = tag.get("href") or ""
        if "mw:PageProp/Category" in rel and href:
            tag.replace_with(soup.new_string(f"\nCategory: {href}\n"))
        else:
            tag.decompose()


def normalize_markdown(markdown: str) -> str:
    markdown = markdown.replace("\xa0", " ")
    markdown = re.sub(r"\n{4,}", "\n\n\n", markdown)
    markdown = re.sub(r"[ \t]+\n", "\n", markdown)
    return markdown.strip() + "\n"


def wordish_units(text: str) -> int:
    return len(WORD_PATTERN.findall(text))


def fetch_faithful_markdown(lang: str, title: str) -> str:
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/html/{quote(title)}"
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=(8, 30))
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")
    body = soup.find("body") or soup
    strip_only_technical_noise(body, soup)
    absolutize_links(body, lang)
    markdown = normalize_markdown(
        md(str(body), heading_style="ATX", bullets="-", strip=["span"])
    )
    return markdown


def build_one(lang: str, title: str) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    txt_path = OUT_DIR / f"{lang}.txt"
    meta_path = OUT_DIR / f"{lang}.meta.json"

    markdown = fetch_faithful_markdown(lang, title)
    txt_path.write_text(markdown, encoding="utf-8")

    meta = {
        "lang": lang,
        "title": title,
        "variant": "wiki_faithful_markdown",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "chars": len(markdown),
        "wordish_units": wordish_units(markdown),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    titles = get_localized_titles()
    print("Resolved titles:", json.dumps(titles, ensure_ascii=False))

    for lang in LANGS:
        meta = build_one(lang, titles[lang])
        print(f"[{lang}] '{titles[lang]}' -> {meta['chars']} chars, {meta['wordish_units']} word-ish units")


if __name__ == "__main__":
    main()

"""
Niche Dataset Scraper
======================
Scrapes text datasets from the internet based on a specific niche or a list of URLs.
Cleans and formats text data for causal language model training.
"""

import os
import re
import time
import argparse
import urllib.request
import urllib.parse
import json
from bs4 import BeautifulSoup


# ═══════════════════════════════════════════════════════════════════════
# Text Cleaning Utilities
# ═══════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """Clean raw scraped text to make it suitable for language modeling.

    - Removes HTML tags (if BeautifulSoup missed any).
    - Removes Wikipedia citation brackets like [1], [2], [citation needed].
    - Normalizes white spaces and newlines.
    - Strips lines that are too short or look like navigation.
    """
    if not text:
        return ""

    # Remove inline citations [1], [12], [citation needed]
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[[a-zA-Z\s]+\]", "", text)
    text = re.sub(r"\[edit\]", "", text)

    # Normalize unicode spaces
    text = re.sub(r"\xa0", " ", text)
    text = re.sub(r"\u200b", "", text)

    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        # Filter out navigation elements or very short fragments
        if len(line) < 30:
            continue
        # Filter out lines that look like copyright/legal strings
        if "terms of use" in line.lower() or "privacy policy" in line.lower():
            continue
        cleaned_lines.append(line)

    # Reconnect paragraphs with single newlines
    cleaned_text = "\n".join(cleaned_lines)
    # Remove multiple consecutive blank lines or spaces
    cleaned_text = re.sub(r"\n{2,}", "\n\n", cleaned_text)
    cleaned_text = re.sub(r" {2,}", " ", cleaned_text)

    return cleaned_text.strip()


# ═══════════════════════════════════════════════════════════════════════
# Wikipedia Search and Scraping
# ═══════════════════════════════════════════════════════════════════════

def search_wikipedia(query: str, limit: int = 10) -> list[str]:
    """Search Wikipedia for articles matching a niche/query.

    Returns:
        List of article titles.
    """
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "utf8": 1,
    }
    url = f"https://en.wikipedia.org/w/api.php?{urllib.parse.urlencode(params)}"
    
    print(f"[Scraper] Querying Wikipedia API for niche: '{query}'...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "UltraGPT-Scraper/1.0"})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode("utf-8"))
            search_results = data.get("query", {}).get("search", [])
            titles = [result["title"] for result in search_results]
            print(f"[Scraper] Found {len(titles)} matching articles.")
            return titles
    except Exception as e:
        print(f"[Scraper] Error searching Wikipedia: {e}")
        return []


def fetch_wikipedia_article(title: str) -> str:
    """Fetch raw text content of a specific Wikipedia article.

    Returns:
        Cleaned plain text of the article.
    """
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": 1,  # Request plain text, not HTML
        "exlimit": 1,
    }
    url = f"https://en.wikipedia.org/w/api.php?{urllib.parse.urlencode(params)}"
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "UltraGPT-Scraper/1.0"})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for page_id, page in pages.items():
                if "extract" in page:
                    return page["extract"]
    except Exception as e:
        print(f"[Scraper] Error fetching Wikipedia article '{title}': {e}")
    return ""


def scrape_wikipedia_niche(query: str, limit: int = 10, delay: float = 1.0) -> str:
    """Search and extract text from Wikipedia for a specific niche query."""
    titles = search_wikipedia(query, limit=limit)
    corpus = []
    
    for idx, title in enumerate(titles):
        print(f"[Scraper] [{idx+1}/{len(titles)}] Extracting: '{title}'...")
        article_text = fetch_wikipedia_article(title)
        if article_text:
            cleaned = clean_text(article_text)
            if cleaned:
                corpus.append(f"--- Article: {title} ---\n{cleaned}")
        time.sleep(delay)  # Be polite
        
    return "\n\n".join(corpus)


# ═══════════════════════════════════════════════════════════════════════
# General Web Page Scraping
# ═══════════════════════════════════════════════════════════════════════

def scrape_url(url: str) -> str:
    """Fetch a web page and extract clean text from it.

    Strips scripts, styling, headers, footers, and pulls the core paragraphs.
    """
    print(f"[Scraper] Scraping custom URL: {url} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read()
            soup = BeautifulSoup(html, "html.parser")
            
            # Kill script, style, head, footer elements
            for element in soup(["script", "style", "head", "title", "meta", "[document]", "footer", "nav"]):
                element.decompose()
                
            # Grab all paragraphs or block level elements
            text_blocks = []
            for p in soup.find_all(["p", "article", "section"]):
                text = p.get_text()
                if len(text.strip()) > 30:
                    text_blocks.append(text)
                    
            raw_text = "\n".join(text_blocks)
            return clean_text(raw_text)
    except Exception as e:
        print(f"[Scraper] Failed to scrape {url}: {e}")
    return ""


# ═══════════════════════════════════════════════════════════════════════
# CLI Interface
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Niche Dataset Web Scraper")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--niche", type=str, help="Search query/niche to scrape Wikipedia articles for")
    group.add_argument("--urls", type=str, nargs="+", help="Specific website URLs to scrape")
    
    parser.add_argument("--limit", type=int, default=10, help="Maximum Wikipedia articles to scrape (default: 10)")
    parser.add_argument("--output", type=str, default="data_pipeline/dataset.txt", help="Output path (default: data_pipeline/dataset.txt)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay in seconds between requests (default: 1.0)")
    parser.add_argument("--append", action="store_true", help="Append text if output file already exists")
    
    args = parser.parse_args()

    scraped_content = ""

    if args.niche:
        scraped_content = scrape_wikipedia_niche(args.niche, limit=args.limit, delay=args.delay)
    elif args.urls:
        scraped_texts = []
        for idx, url in enumerate(args.urls):
            text = scrape_url(url)
            if text:
                scraped_texts.append(f"--- Source: {url} ---\n{text}")
            if idx < len(args.urls) - 1:
                time.sleep(args.delay)
        scraped_content = "\n\n".join(scraped_texts)

    if not scraped_content.strip():
        print("[Scraper] No content retrieved. Output file unchanged.")
        return

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    mode = "a" if args.append else "w"
    with open(args.output, mode, encoding="utf-8") as f:
        f.write(scraped_content)
        f.write("\n\n")

    action = "Appended" if args.append else "Wrote"
    print(f"[Scraper] {action} {len(scraped_content):,} characters of clean text to {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Dependency-free web search for Pip-Compact (standard library only, no API keys).

Scrapes DuckDuckGo's HTML endpoint with urllib and parses results with html.parser.
Use fetch_context(query) to get a plain-text block suitable for prepending to a prompt.

Usage (CLI check):
    python web_search.py "your query"
"""
import sys
import urllib.parse
import urllib.request
from html.parser import HTMLParser

DDG_URL = "https://html.duckduckgo.com/html/"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) PipCompact/1.0 (+https://github.com/coffee-hour/Pip-Compact)"
MAX_SNIPPET_CHARS = 300
MAX_CONTEXT_CHARS = 1200


def _unwrap_url(href):
    """DuckDuckGo wraps targets as //duckduckgo.com/l/?uddg=<encoded url>. Return the real URL."""
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urllib.parse.urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = urllib.parse.parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return target
    return href


class _DDGParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results = []
        self._field = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        classes = (a.get("class") or "").split()
        if tag == "a" and "result__a" in classes:
            self.results.append({"title": "", "url": _unwrap_url(a.get("href", "")), "snippet": ""})
            self._field = "title"
        elif "result__snippet" in classes:
            self._field = "snippet"

    def handle_endtag(self, tag):
        if tag in ("a", "div") and self._field:
            self._field = None

    def handle_data(self, data):
        if self._field and self.results:
            self.results[-1][self._field] += data


def search(query, max_results=3, timeout=10):
    """Return a list of {"title", "url", "snippet"} dicts. Raises on network errors."""
    body = urllib.parse.urlencode({"q": query}).encode("utf-8")
    req = urllib.request.Request(
        DDG_URL,
        data=body,
        headers={"User-Agent": USER_AGENT, "Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        page = resp.read().decode("utf-8", errors="replace")
    parser = _DDGParser()
    parser.feed(page)
    out = []
    for r in parser.results:
        title = " ".join(r["title"].split())
        snippet = " ".join(r["snippet"].split())[:MAX_SNIPPET_CHARS]
        if not (title or snippet) or not r["url"]:
            continue
        out.append({"title": title, "url": r["url"], "snippet": snippet})
        if len(out) >= max_results:
            break
    return out


def format_context(results):
    """Turn search results into a compact text block for the prompt."""
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['title']} ({r['url']})\n    {r['snippet']}")
    text = "\n".join(lines)
    return text[:MAX_CONTEXT_CHARS]


def fetch_context(query, max_results=3, timeout=10):
    """Search and format. Returns "" on any failure so chat keeps working offline."""
    try:
        results = search(query, max_results=max_results, timeout=timeout)
    except Exception as exc:  # network down, DDG blocked, etc.
        print(f"(web search unavailable: {exc})", file=sys.stderr)
        return ""
    return format_context(results)


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "python standard library"
    res = search(q)
    print(format_context(res) or "(no results)")

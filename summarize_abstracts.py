#!/usr/bin/env python3
"""
Summarize PubMed abstracts with a local Llama 3.3 70B served by vLLM
(OpenAI-compatible API).

Pipeline:  load messy JSON -> clean/normalize -> build payload -> POST with
timeouts + retries -> write results (one status per record, never crashes).

Usage:
    python summarize_abstracts.py --input data/raw_abstracts.json --output results.json
    python summarize_abstracts.py --dry-run          # no network, prints payloads
Env overrides: LLM_URL, LLM_MODEL, LLM_READ_TIMEOUT, LLM_MAX_RETRIES
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import requests

log = logging.getLogger("summarizer")

# ----------------------------------------------------------------- config ---
DEFAULT_URL = "http://192.168.1.100:8000/v1/chat/completions"
DEFAULT_MODEL = "llama-3.3-70b"      # must match --served-model-name in vLLM
PROMPT = "Summarize the clinical outcome of this study in one sentence."
SYSTEM_PROMPT = (
    "You are a careful biomedical assistant. Use only the information in the "
    "abstract. Do not add facts. Reply with exactly one sentence."
)
CONNECT_TIMEOUT_S = 5.0     # LAN: if we can't connect in 5s the server is down
READ_TIMEOUT_S = 120.0      # 70B model can legitimately take 45s+ under load
MAX_RETRIES = 3
BACKOFF_BASE_S = 2.0
MAX_ABSTRACT_CHARS = 8000   # ~2k tokens; protects the server's max-model-len
# Retry only on transient failures. 4xx (bad request / wrong model) are not retried.
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}

# Key aliases seen in the legacy database ("messy" schema).
TITLE_KEYS = ("ArticleTitle", "title", "Title", "article_title")
ABSTRACT_KEYS = ("abstract_text", "Abstract", "abstract", "AbstractText")
AUTHOR_KEYS = ("authorList", "authors", "Authors", "author_list")
ID_KEYS = ("id", "ID", "pmid", "PMID")
JUNK_VALUES = {"", "null", "none", "n/a", "na", "nan", "unknown", "undefined", "-"}


# ----------------------------------------------------------------- models ---
@dataclass
class Record:
    pmid: str
    title: Optional[str]
    abstract: str
    authors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class Result:
    pmid: str
    status: str                       # ok | skipped | failed
    summary: Optional[str] = None
    title: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 0
    latency_s: Optional[float] = None
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------- cleaning ---
def _clean_text(value: Any) -> Optional[str]:
    """Return a trimmed string or None for null/empty/placeholder values."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (list, tuple)):          # abstract split in sections
        value = " ".join(str(v) for v in value if v is not None)
    text = re.sub(r"\s+", " ", str(value)).strip()
    return None if text.lower() in JUNK_VALUES else text


def _first(raw: dict, keys: tuple[str, ...]) -> Any:
    """First key that exists AND holds a usable value (not just first key)."""
    for k in keys:
        if k in raw and _clean_text(raw[k]) is not None:
            return raw[k]
    return None


def _clean_authors(value: Any) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else re.split(r"[;|]", str(value))
    out = [_clean_text(i) for i in items]
    return [a for a in out if a]


def _clean_id(value: Any) -> Optional[str]:
    """'PMID:349281' / 349281 / '349281' -> '349281'. Garbage -> None."""
    text = _clean_text(value)
    if text is None:
        return None
    m = re.fullmatch(r"(?:PMID[:\s]*)?(\d+)", text, flags=re.I)
    return m.group(1) if m else None


def clean_record(raw: Any, index: int) -> tuple[Optional[Record], Optional[Result]]:
    """
    Returns (Record, None) if usable, or (None, Result[skipped]) if not.
    Never raises on bad input.
    """
    if not isinstance(raw, dict):
        return None, Result(f"row-{index}", "skipped",
                            error=f"entry is {type(raw).__name__}, expected object")

    warnings: list[str] = []
    pmid = _clean_id(_first(raw, ID_KEYS))
    if pmid is None:
        raw_id = _clean_text(_first(raw, ID_KEYS) or raw.get("id"))
        warnings.append(f"invalid id {raw_id!r}")
        pmid = f"row-{index}"                       # keep traceability

    abstract = _clean_text(_first(raw, ABSTRACT_KEYS))
    title = _clean_text(_first(raw, TITLE_KEYS))
    if title is None:
        warnings.append("missing title")

    if abstract is None:
        # Nothing to summarize: do NOT waste a 70B inference call.
        return None, Result(pmid, "skipped", title=title,
                            error="no abstract text", warnings=warnings)

    if len(abstract) > MAX_ABSTRACT_CHARS:
        abstract = abstract[:MAX_ABSTRACT_CHARS]
        warnings.append(f"abstract truncated to {MAX_ABSTRACT_CHARS} chars")

    return Record(pmid, title, abstract, _clean_authors(_first(raw, AUTHOR_KEYS)),
                  warnings), None


def load_records(path: Path) -> tuple[list[Record], list[Result]]:
    """Parse the file defensively. Returns (valid records, skipped results)."""
    try:
        text = path.read_text(encoding="utf-8-sig")   # tolerate BOM
        data = json.loads(text)
    except FileNotFoundError:
        log.error("Input file not found: %s", path); raise SystemExit(2)
    except json.JSONDecodeError as e:
        log.error("Input is not valid JSON (%s). Trying line-delimited JSON.", e)
        data = []
        for n, line in enumerate(text.splitlines(), 1):
            line = line.strip().rstrip(",")
            if not line or line in "[]":
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                data.append(f"unparseable line {n}")   # becomes a 'skipped' row

    if isinstance(data, dict):                          # single object or wrapper
        data = data.get("records") or data.get("data") or [data]
    if not isinstance(data, list):
        log.error("Top-level JSON must be a list/object."); raise SystemExit(2)

    records, skipped, seen = [], [], set()
    for i, raw in enumerate(data):
        rec, skip = clean_record(raw, i)
        if skip:
            skipped.append(skip); continue
        if rec.pmid in seen:
            skipped.append(Result(rec.pmid, "skipped", title=rec.title,
                                  error="duplicate id")); continue
        seen.add(rec.pmid)
        records.append(rec)
    return records, skipped


# ---------------------------------------------------------------- payload ---
def build_payload(rec: Record, model: str) -> dict:
    user = f"{PROMPT}\n\nTitle: {rec.title or 'N/A'}\n\nAbstract: {rec.abstract}"
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,     # deterministic, medical context
        "max_tokens": 120,      # one sentence; caps KV-cache growth per request
        "stream": False,
    }


# ------------------------------------------------------------------- HTTP ---
def _extract_summary(resp_json: dict) -> str:
    try:
        content = resp_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"unexpected response shape: {e}") from e
    text = _clean_text(content)
    if text is None:
        raise ValueError("model returned empty content")
    return text


def summarize(rec: Record, session: requests.Session, url: str, model: str,
              read_timeout: float, max_retries: int) -> Result:
    payload = build_payload(rec, model)
    result = Result(rec.pmid, "failed", title=rec.title, warnings=rec.warnings)
    start = time.monotonic()

    for attempt in range(1, max_retries + 2):          # 1 try + N retries
        result.attempts = attempt
        retryable, err = True, ""
        try:
            r = session.post(url, json=payload,
                             timeout=(CONNECT_TIMEOUT_S, read_timeout))
            if r.status_code == 200:
                try:
                    result.summary = _extract_summary(r.json())
                    result.status, result.error = "ok", None
                    break
                except ValueError as e:                # bad JSON / empty output
                    err = f"bad response: {e}"
            elif r.status_code in RETRYABLE_STATUS:
                err = f"HTTP {r.status_code}: {r.text[:200]}"
            else:                                      # 400/401/404...: our bug
                retryable, err = False, f"HTTP {r.status_code}: {r.text[:200]}"
        except requests.exceptions.ConnectTimeout:
            err = f"connect timeout after {CONNECT_TIMEOUT_S}s (server down/unreachable)"
        except requests.exceptions.ReadTimeout:
            err = f"no response within {read_timeout}s (server overloaded?)"
        except requests.exceptions.ConnectionError as e:
            err = f"connection dropped/refused: {type(e).__name__}"
        except requests.exceptions.RequestException as e:
            retryable, err = False, f"request error: {e}"

        result.error = err
        if not retryable or attempt > max_retries:
            break
        delay = BACKOFF_BASE_S * 2 ** (attempt - 1) + random.uniform(0, 1)
        log.warning("[%s] attempt %d failed: %s -> retry in %.1fs",
                    rec.pmid, attempt, err, delay)
        time.sleep(delay)

    result.latency_s = round(time.monotonic() - start, 2)
    return result


# ------------------------------------------------------------------- main ---
def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=Path("data/raw_abstracts.json"))
    p.add_argument("--output", type=Path, default=Path("results.json"))
    p.add_argument("--url", default=os.getenv("LLM_URL", DEFAULT_URL))
    p.add_argument("--model", default=os.getenv("LLM_MODEL", DEFAULT_MODEL))
    p.add_argument("--read-timeout", type=float,
                   default=float(os.getenv("LLM_READ_TIMEOUT", READ_TIMEOUT_S)))
    p.add_argument("--max-retries", type=int,
                   default=int(os.getenv("LLM_MAX_RETRIES", MAX_RETRIES)))
    p.add_argument("--workers", type=int, default=2,
                   help="parallel requests; keep <= vLLM --max-num-seqs (default 2)")
    p.add_argument("--dry-run", action="store_true",
                   help="clean data and print payloads, send nothing")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    records, results = load_records(args.input)
    log.info("Loaded %d valid record(s), %d skipped", len(records), len(results))
    for s in results:
        log.warning("[%s] skipped: %s", s.pmid, s.error)

    if args.dry_run:
        for rec in records:
            print(json.dumps(build_payload(rec, args.model), indent=2))
        return 0

    with requests.Session() as session, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(summarize, r, session, args.url, args.model,
                               args.read_timeout, args.max_retries) for r in records]
        for f in futures:
            res = f.result()
            log.info("[%s] %s%s", res.pmid, res.status,
                     f" ({res.error})" if res.error else "")
            results.append(res)

    args.output.write_text(json.dumps([asdict(r) for r in results], indent=2,
                                      ensure_ascii=False), encoding="utf-8")
    ok = sum(r.status == "ok" for r in results)
    failed = sum(r.status == "failed" for r in results)
    log.info("Done: %d ok, %d failed, %d skipped -> %s",
             ok, failed, len(results) - ok - failed, args.output)
    return 1 if failed else 0       # non-zero only for real infra failures


if __name__ == "__main__":
    sys.exit(main())

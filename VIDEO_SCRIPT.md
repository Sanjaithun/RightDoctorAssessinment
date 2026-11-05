# 3-minute walkthrough script (record yourself, upload to YouTube as **Unlisted**)

Say this in your own words. Target ~2:45. Screen: VS Code with `summarize_abstracts.py`, then a terminal.

**0:00 – 0:20 Intro.** "This script cleans messy PubMed JSON, builds an OpenAI-style payload, and sends it to our local Llama 3.3 70B on the DGX Spark. It never crashes on bad data or a bad server."

**0:20 – 1:15 Null data** (scroll to `clean_record`, `_first`, `_clean_id`).
- "Three records, three different schemas: `ArticleTitle` vs `title`, `abstract_text` vs `Abstract`. I use alias lists, and `_first` picks the first key that actually has a usable value."
- "Record 2 has a null title and `authors: 'Unknown'`: title becomes None with a warning, authors become an empty list, and the integer id becomes a string."
- "Record 3 has `ERR_NOT_FOUND` and a null abstract. With nothing to summarize, I skip it and don't spend a 70B inference call. It still appears in the output as `skipped` with a reason."

**1:15 – 1:40 Payload** (`build_payload`). "System + user message, temperature 0 for determinism, `max_tokens` 120 so one sentence can't grow the KV cache, and abstracts are capped in length."

**1:40 – 2:30 The 45-second question** (`summarize`). 
- "Timeouts are a tuple: 5 seconds to connect, 120 to read. A 70B model can take 45 seconds under load, so **the script just waits and succeeds on attempt 1, no retry**."
- "If the server really stalls past 120s, or drops the connection, or returns 429/5xx, I retry 3 times with exponential backoff and jitter. 4xx errors aren't retried because retrying a bad request is pointless."
- "After the last retry the record is marked `failed`, the other records continue, and the exit code is 1."
- "Workers are capped at 2 to match vLLM's `--max-num-seqs`, so the client can't flood the server."

**2:30 – 3:00 Demo + infra.** Run `pytest -q` (green), show `results.json`, then open `docker/run_vllm.sh`: "BF16 70B is ~140 GB and doesn't fit in 128 GB, so FP8 weights plus capped context and concurrency."

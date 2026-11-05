# PubMed Abstract Summarizer (local Llama 3.3 70B on DGX Spark)

A production-style Python client that cleans messy PubMed JSON and asks a locally hosted
**Llama 3.3 70B** (vLLM, OpenAI-compatible API at `http://192.168.1.100:8000/v1/chat/completions`)
to *"Summarize the clinical outcome of this study in one sentence."*

```
raw JSON ──► clean & normalize ──► build payload ──► POST (timeouts + retries) ──► results.json
              (skip corrupt rows)                       (never crashes the batch)
```

## Repository layout

| Path | Purpose |
|---|---|
| `summarize_abstracts.py` | The deliverable script |
| `data/raw_abstracts.json` | The messy sample data from the assignment |
| `docker/run_vllm.sh` | Exact `docker run` + vLLM arguments for the DGX Spark |
| `tests/test_summarizer.py` | Tests, including mock servers for slow, dropped and error responses |
| `VIDEO_SCRIPT.md` | Outline for the 3-minute walkthrough |

## Quick start

```bash
pip install -r requirements.txt
python summarize_abstracts.py --dry-run          # clean data, print payloads, no network
python summarize_abstracts.py                    # real run -> results.json
pytest -q                                        # tests (no GPU or server needed)
```

Options: `--url`, `--model`, `--read-timeout` (default 120), `--max-retries` (default 3),
`--workers` (default 2). Each also has an env var: `LLM_URL`, `LLM_MODEL`, `LLM_READ_TIMEOUT`, `LLM_MAX_RETRIES`.

---

## 1. How the messy data is handled

| Problem in the data | What the script does |
|---|---|
| Different key names (`ArticleTitle`/`title`, `abstract_text`/`Abstract`, `authorList`/`authors`) | Alias lists; picks the first key that holds a *usable* value |
| `title: null` | Title becomes `None`, a warning is recorded, the summary still runs |
| `authors: "Unknown"` or a plain string | Normalized to a list; placeholders like `Unknown`/`N/A`/`null` become `[]` |
| `id` is `"PMID:349281"` or integer `8829102` | Normalized to the string `"349281"` / `"8829102"` |
| `id: "ERR_NOT_FOUND"` | Flagged as invalid; falls back to `row-<index>` so it stays traceable |
| `abstract_text: null` (corrupted entry) | **Skipped, no LLM call is made**; reported as `skipped` with a reason |
| Entry isn't an object, duplicate IDs, invalid JSON, BOM | Skipped or recovered; nothing raises |
| Huge abstract | Truncated to 8,000 chars so one request can't blow the context window |

Every input row appears in the output with a `status` (`ok` / `skipped` / `failed`), so no record is silently lost.

Result for the sample data: `349281` → ok, `8829102` → ok (missing title warning), `row-2` → skipped (no abstract).

## 2. What happens if the server takes 45 seconds?

Timeouts are a tuple: **connect = 5 s, read = 120 s**.

* **45 s response → the script waits and succeeds on attempt 1, with no retry.** A 70B model can take that long under load, and retrying would only add more work to an already busy server. (Tested: `test_slow_server_within_timeout_succeeds_without_retry`.)
* If the server stalls **beyond 120 s**, drops the connection, or returns `408/429/500/502/503/504`, the script retries up to 3 times with exponential backoff (2 s, 4 s, 8 s) plus jitter.
* `connect timeout` (5 s) fails fast, because on a LAN a host that doesn't answer is down.
* `4xx` errors (wrong model name, bad payload) are **not** retried. They won't fix themselves.
* After the last retry the record is marked `failed` with the reason. **The batch continues**, and the process exits with code `1` so cron/CI notice.
* `--workers` defaults to 2 and should stay ≤ vLLM's `--max-num-seqs`. A client that floods the server is a common cause of cascading timeouts.

## 3. Infrastructure: avoiding OOM on the DGX Spark

The DGX Spark has **128 GB of *unified* memory (CPU and GPU share it), ARM64 / Blackwell GB10**. Two things follow:

1. **Llama 3.3 70B in BF16 needs ~140 GB for weights alone, so it will OOM.** You must serve a quantized model: **FP8 ≈ 70 GB** (used here) or FP4/INT4 ≈ 35–40 GB.
2. Because memory is shared, the OS, Docker and page cache compete with the GPU. Don't set `--gpu-memory-utilization` near 0.95.

### Exact command

```bash
docker run -d --name vllm-llama33-70b \
  --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --restart unless-stopped \
  -p 8000:8000 \
  -e HF_TOKEN=$HF_TOKEN \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  nvcr.io/nvidia/vllm:latest \
  vllm serve RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic \
    --served-model-name llama-3.3-70b \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 4096 \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.80 \
    --kv-cache-dtype fp8 \
    --max-num-batched-tokens 4096 \
    --enable-prefix-caching \
    --swap-space 0
```

(Also in `docker/run_vllm.sh`. Pin the image tag to the current NGC build for the Spark instead of `latest` in production.)

### Why each argument

| Argument | Reason |
|---|---|
| Pre-quantized **FP8** checkpoint | ~70 GB weights fit in 128 GB. A pre-quantized checkpoint avoids loading BF16 first, which would OOM during startup |
| `--max-model-len 4096` | Our requests are ~300 tokens (abstract + prompt + ≤120 output). Capping context sizes the KV cache for what we need |
| `--max-num-seqs 4` | Hard cap on concurrent sequences, so a burst can't grow memory. Matches the client's `--workers 2` with headroom |
| `--gpu-memory-utilization 0.80` | ≈ 102 GB budget, leaving ~26 GB for OS, Docker and CUDA context on unified memory |
| `--kv-cache-dtype fp8` | Halves KV-cache memory |
| `--max-num-batched-tokens 4096` | Limits peak activation memory during prefill |
| `--enable-prefix-caching` | Every request shares the same system prompt and instruction, so it is reused |
| `--swap-space 0` | No CPU swap. On unified memory it competes with the GPU anyway |
| `--ipc=host`, `--ulimit memlock=-1` | Required for stable shared memory/pinned memory |
| `--served-model-name llama-3.3-70b` | Must match the script's `--model` |

**Memory budget:** weights ~70 GB + KV cache (Llama-70B: 80 layers × 8 KV heads × 128 dim → ~0.16 MB/token in FP8; 4 seqs × 4096 tokens ≈ 2.6 GB) + overhead → comfortably under the ~102 GB cap.

> Verify with `curl http://192.168.1.100:8000/v1/models` and watch `docker logs -f vllm-llama33-70b`.
> If a different checkpoint is used, change `MODEL` and keep `--served-model-name` the same.

## Design decisions

* `requests` is the only runtime dependency; the script is a single file with type hints and logging.
* `temperature=0`, `max_tokens=120`, and a "use only the abstract" system prompt reduce hallucination in a medical context.
* Output is machine-readable (`results.json`) and includes attempts and latency per record.

## Limitations

* Tested against mock servers (no access to the real DGX Spark from the development environment). The vLLM flags follow current vLLM options; adjust if your build's flag names differ.
* The model may return more than one sentence despite instructions. The script does not post-validate that.

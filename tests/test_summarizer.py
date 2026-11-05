import json, sys, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import summarize_abstracts as s

DATA = Path(__file__).resolve().parents[1] / "data" / "raw_abstracts.json"


def test_cleaning_sample_data():
    recs, skipped = s.load_records(DATA)
    assert [r.pmid for r in recs] == ["349281", "8829102"]
    assert recs[1].title is None and "missing title" in recs[1].warnings
    assert recs[1].authors == []                     # "Unknown" -> []
    assert recs[0].authors == ["Smith, J", "Doe, A"]
    assert len(skipped) == 1 and skipped[0].error == "no abstract text"


def test_garbage_entries_do_not_crash():
    for bad in [None, 5, "x", [], {}, {"id": None, "abstract": ""}]:
        rec, skip = s.clean_record(bad, 0)
        assert rec is None and skip.status == "skipped"


def _server(handler_fn):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            handler_fn(self)
        def log_message(self, *a): pass
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/v1/chat/completions"


def _ok(h):
    body = json.dumps({"choices": [{"message": {"content": "It worked."}}]}).encode()
    h.send_response(200); h.send_header("Content-Length", str(len(body)))
    h.end_headers(); h.wfile.write(body)


def _rec():
    return s.Record("1", "t", "abstract")


def test_success():
    srv, url = _server(_ok)
    import requests
    r = s.summarize(_rec(), requests.Session(), url, "m", 5, 1)
    assert r.status == "ok" and r.summary == "It worked."


def test_slow_server_within_timeout_succeeds_without_retry():
    def slow(h): time.sleep(1.5); _ok(h)             # stands in for the 45s case
    srv, url = _server(slow)
    import requests
    r = s.summarize(_rec(), requests.Session(), url, "m", 5, 2)
    assert r.status == "ok" and r.attempts == 1


def test_timeout_exceeded_retries_then_fails_cleanly(monkeypatch):
    monkeypatch.setattr(s, "BACKOFF_BASE_S", 0.01)
    def stall(h): time.sleep(1.0); _ok(h)
    srv, url = _server(stall)
    import requests
    r = s.summarize(_rec(), requests.Session(), url, "m", 0.3, 2)
    assert r.status == "failed" and r.attempts == 3 and "no response" in r.error


def test_connection_refused(monkeypatch):
    monkeypatch.setattr(s, "BACKOFF_BASE_S", 0.01)
    import requests
    r = s.summarize(_rec(), requests.Session(),
                    "http://127.0.0.1:9/v1/chat/completions", "m", 1, 1)
    assert r.status == "failed" and r.attempts == 2


def test_4xx_not_retried():
    def bad(h): h.send_response(404); h.send_header("Content-Length", "2"); h.end_headers(); h.wfile.write(b"no")
    srv, url = _server(bad)
    import requests
    r = s.summarize(_rec(), requests.Session(), url, "m", 2, 3)
    assert r.attempts == 1 and "404" in r.error

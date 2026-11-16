# Security and privacy

## Threat model

The public site accepts uploaded spreadsheets from anonymous visitors and runs ML code on them.
The main risks are: code execution through uploaded or tampered files, resource exhaustion,
leaking student data, malicious content flowing back to a teacher's spreadsheet, and abuse of
the training endpoint.

## Controls

| Risk | Control | Where |
|---|---|---|
| Code execution via model files (pickle) | Models are stored in the **skops** format and loaded with an explicit allow-list of types; every artifact must match the SHA-256 in `artifacts/manifest.json`; file names are validated (no path traversal). Uploaded files are never deserialised. | `riskml/serving/registry.py` |
| Oversized uploads / memory exhaustion | Request bodies capped by an ASGI middleware (declared and streamed size), file ≤ 4 MB, ≤ 50,000 rows, ≤ 200 columns | `riskml/server/security.py`, `riskml/config.py` |
| Zip bombs / XML attacks in .xlsx | Total uncompressed size and entry count checked before parsing; `defusedxml` guards openpyxl; formulas are never evaluated | `riskml/custom/io.py` |
| Spoofed file types | Format decided by content (magic bytes), not by the extension; legacy `.xls` rejected | `riskml/custom/io.py` |
| CSV / formula injection into the teacher's Excel | Exported cells starting with `= + - @` are prefixed with `'` | `web/src/lib/csv.ts` (tested) |
| Invalid or hostile JSON | Pydantic models with `extra="forbid"`, strict types, bounded sizes; feature values validated against the schema (unknown categories, out-of-range and non-finite numbers rejected with HTTP 422) | `riskml/server/schemas.py`, `riskml/serving/predictor.py` |
| Abuse / denial of service | Per-IP token-bucket rate limit on upload endpoints (HTTP 429 + Retry-After); at most 2 concurrent trainings per instance (HTTP 503); training time budget | `riskml/server/main.py` |
| CPU oversubscription under concurrency | One OpenMP/BLAS thread per request, applied inside each worker thread | `riskml/server/main.py` (tested) |
| Information leakage in errors | Generic error bodies with a request ID; stack traces only in server logs | `riskml/server/security.py` |
| Browser attacks | CSP, `X-Frame-Options: DENY`, `nosniff`, Referrer-Policy, Permissions-Policy, HSTS over HTTPS; CORS allow-list | `riskml/server/security.py`, `vercel.json` |
| Personal data | Uploads are processed in memory and dropped after the response; nothing is written to disk or logged (logs contain method, path, status and timing only). Name / e-mail / phone columns are detected and never used as features. API responses are `Cache-Control: no-store`. | `riskml/custom/profile.py`, `riskml/server/security.py` |
| Vulnerable dependencies | Exact version pins, Dependabot, `pip-audit` and `npm audit` in CI | `.github/` |

## Known limits

- The rate limiter is per server instance (serverless instances do not share memory); use a
  shared store (e.g. Redis) if the app is ever deployed for heavy public use.
- `X-Forwarded-For` is trusted for client identification because Vercel and Hugging Face set it;
  set `AGRIM_TRUST_PROXY=0` when running without a proxy.

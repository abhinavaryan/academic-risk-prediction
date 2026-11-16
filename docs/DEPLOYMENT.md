# Deployment guide

Everything here is free. You need a GitHub account; no credit card is required.

| Where | What you get | When to use |
|---|---|---|
| **Vercel** (main) | `https://<project>.vercel.app` — website + API | The link you share and present |
| **Hugging Face Spaces** (backup) | `https://huggingface.co/spaces/<user>/<space>` | Second live link in case one is down on demo day |
| **GitHub Codespaces** | A full Linux machine in your browser with the site on a private URL | "Localhost" when your laptop cannot install Python/Node |
| **Your own computer** | `http://localhost:8000` | If Python 3.12+ and Node 22+ are installed |

---

## 1. Vercel (main link)

1. Push the repository to GitHub (branch `main`).
2. Open <https://vercel.com>, choose **Sign up → Continue with GitHub** (Hobby plan is free).
3. **Add New… → Project**, pick this repository, click **Import**.
4. Leave **Framework Preset = Other**. Do not change build settings: `vercel.json` already sets
   the install command, build command, output folder, the Python API function and security headers.
5. Click **Deploy** and wait 3–5 minutes.
6. Check it: open `https://<your-project>.vercel.app/api/health` → you should see `"status":"ok"`.

Every later `git push` to `main` redeploys automatically. Pull requests get their own preview URL.

**How it works on Vercel:** the Next.js site is exported to static files and served from Vercel's CDN;
`api/index.py` runs the FastAPI app as a Python serverless function (limits: 4.5 MB request body →
uploads are capped at 4 MB; 60 s per request on Hobby → custom training uses a 30 s budget).

**Troubleshooting**

| Symptom | Fix |
|---|---|
| Build log: "Serverless Function has exceeded the unzipped maximum size" | Make sure you did not commit `data/`, `.venv/` or `web/node_modules/` (they are in `.gitignore`/`.vercelignore`). |
| First request takes ~5–10 s | Normal cold start of the Python function; later requests are fast. |
| Training a very large file returns a timeout | Use a smaller file, or use the Hugging Face link (no 60 s limit). |

## 2. Hugging Face Spaces (backup link)

1. Create an account at <https://huggingface.co>.
2. **New → Space**: name it (e.g. `agrim`), **SDK = Docker**, template **Blank**, hardware **CPU basic (free)**, Public.
3. **Settings → Access Tokens → Create new token** with **Write** access. Copy it.
4. In your GitHub repository: **Settings → Secrets and variables → Actions**
   - **New repository secret**: `HF_TOKEN` = the token
   - **Variables → New repository variable**: `HF_SPACE` = `your-username/agrim`
5. GitHub → **Actions → "Deploy to Hugging Face Spaces" → Run workflow**.
6. The Space builds the `Dockerfile` (5–10 min). Your backup URL is `https://huggingface.co/spaces/your-username/agrim`.

Free Spaces sleep after a period without visitors; the first visit wakes them in under a minute. Open
the link once before you present.

## 3. GitHub Codespaces ("localhost" with nothing installed)

1. On the repository page click **Code → Codespaces → Create codespace on main**.
2. Wait for the setup (about 5 minutes the first time): it installs Python and Node packages and builds the site.
3. The server starts automatically and a browser tab opens on port 8000. If it does not, open the
   **Ports** tab and click the globe icon next to port 8000.

Free accounts get 60 core-hours per month. Stop the codespace when you are done.

## 4. Your own computer

Requires Python 3.12+ and Node.js 22+.

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

```bash
pip install -r requirements-dev.txt
```

```bash
cd web && npm ci && npm run build && cd ..
```

```bash
SERVE_STATIC=1 uvicorn riskml.server.main:app --port 8000   # Windows PowerShell: $env:SERVE_STATIC=1; uvicorn ...
```

Open <http://localhost:8000>. For frontend development with hot reload run the API on :8000 and
`npm run dev` inside `web/` (it proxies `/api` to :8000), then open <http://localhost:3000>.

## 5. Re-training the models

Locally (about 35 minutes on 4 cores, downloads 47 MB):

```bash
pip install -r requirements-research.txt && make research
```

Or on GitHub: **Actions → Retrain models → Run workflow**, then download the `research-results`
artifact and copy `artifacts/`, `web/src/generated/` and `reports/` into the repository.

## 6. Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `AGRIM_TIME_BUDGET_S` | `30` | Seconds allowed for bring-your-own-data model comparison |
| `AGRIM_MAX_TRAININGS` | `2` | Concurrent trainings per server instance (more → HTTP 503 + Retry-After) |
| `AGRIM_RATE_PER_MIN` | `20` | Upload/training requests per minute per client IP |
| `AGRIM_CORS_ORIGINS` | `http://localhost:3000` | Extra origins allowed to call the API from a browser |
| `AGRIM_NATIVE_THREADS` | `1` | OpenMP/BLAS threads per request |
| `SERVE_STATIC` | unset | `1` = FastAPI also serves the built website from `web/out` |

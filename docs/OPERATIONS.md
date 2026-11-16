# Delphi RAG: operations guide

## Setup

1. Install [uv](https://docs.astral.sh/uv/).
2. Install dependencies:
   ```
   uv sync
   ```
3. Copy `.env.example` to `.env` and set `DATABASE_URL` to the `dev` branch
   owner connection string (see below), plus `ANTHROPIC_API_KEY` and
   `VOYAGE_API_KEY`.
4. Create the schema and the limited logins on `dev`:
   ```
   uv run python migrations/migrate.py
   uv run python migrations/logins.py --env-file .env
   ```

## Database (Neon Postgres)

The Neon project (Postgres 18) has two branches to start with, each with its
own env file. CI's Postgres uses the same major version. None of these files is committed to git.

| Neon branch | Use                                   | Env file    |
|-------------|---------------------------------------|-------------|
| `dev`       | Building and trying things            | `.env`      |
| `test`      | Automated tests (wiped by the tests)  | `.env.test` |

Each env file starts with one line, the branch's **owner** connection string.
In Neon: **Connect** → choose the branch, database `neondb`, role
`neondb_owner`, pooling **off** (host without `-pooler`):

```
DATABASE_URL="postgresql://neondb_owner:...@ep-.../neondb?sslmode=require"
```

`logins.py` then adds the limited logins to the same file.

### Database logins

Each job uses its own login with only the rights it needs.

| Login | Used by | Can |
|---|---|---|
| `neondb_owner` | Migration runner only | Everything. Never used by app code or GitHub |
| `delphi_ingest` | Loading the corpus | Write chunks (grants added with the chunks table) |
| `delphi_query` | Answering questions | Read chunks; read and write the cache (grants added with those tables) |
| `delphi_eval` | Recording eval runs | Insert and read `eval_runs`; nothing else |

The roles are created by `001_roles.sql` without passwords. `logins.py` turns
on login with random passwords and writes `INGEST_DATABASE_URL`,
`QUERY_DATABASE_URL` and `EVAL_DATABASE_URL` into the env file you name.
Passwords are never printed. Running it again rotates all three. Neither login can create tables.

### Running migrations

The runner applies every `.sql` file in `migrations/` in name order
(`001_`, `002_`, ...). Each file runs in its own transaction, so a failed
file makes no changes. It prints the target host before applying anything.

| Branch | Commands |
|---|---|
| `dev` | `uv run python migrations/migrate.py`<br>`uv run python migrations/logins.py --env-file .env` |
| `test` | `uv run --env-file .env.test python migrations/migrate.py`<br>`uv run python migrations/logins.py --env-file .env.test` |

`logins.py` is only needed once per branch (or to rotate passwords).

### Adding a migration

1. Create the next numbered file, for example `migrations/002_extensions.sql`.
2. Make it safe to run again: `create table if not exists`,
   `add column if not exists`, `on conflict do nothing`.
3. Put the table's grants in the same file (revoke all, then grant), so each
   file owns its own permissions.
4. Neon-only features (lakebase_text) go inside a block that checks
   `pg_available_extensions`, so the same file also runs on CI's plain Postgres.
5. Run it on `dev`, then `test`, and run the tests.

## Tests

```
uv run pytest
```

- Model calls go to fake servers on localhost (`tests/fake_anthropic.py`):
  no API keys, no cost.
- Tests marked `db` run only against `.env.test`. If it's missing they are
  skipped; if it points at the same database as `.env` or `.env.production`,
  the run stops.
- Tests marked `neon` also need lakebase_text, so they run against the Neon
  `test` branch and are skipped elsewhere.
- CI starts a throwaway Postgres with pgvector, writes its own `.env.test`,
  runs the migrations and logins, then the tests. It uses no secrets.

## Corpus

| | |
|---|---|
| Source | Project Gutenberg eBook #674, *Plutarch: Lives of the noble Grecians and Romans*, Dryden's translation as revised by Arthur Hugh Clough ([plain text](https://www.gutenberg.org/ebooks/674.txt.utf-8)) |
| Pinned SHA-256 | `e849186d3483c1779053a5f70c8ae477316c019907c740f878de0d2c788975f8` (downloaded 2026-11-15) |
| Version | `lives-v1`, chunker `para-v1` |
| Lives | Cato the Elder (heading MARCUS CATO), Cato the Younger, Tiberius Gracchus, Caius Gracchus, Caesar, Pompey, Crassus |
| Size | 470 chunks, 123,739 words |

The Comparison essays (for example *Comparison of Crassus with Nicias*) are
left out of v1: they bring in people who aren't in the corpus.

### How the text is chunked (`core/corpus.py`)

- The text has **no chapter numbers**. Each Life is a capitalised heading
  (listed once in the contents and once in the text) followed by paragraphs.
  A Life ends at the next heading.
- `section` is the paragraph number within the Life, from 1.
- Paragraphs under 40 words (verse quotations) join the paragraph before.
- Paragraphs over 450 words are cut at sentence ends into pieces of about
  300 words (`chunk_index` 0, 1, ...). A last piece under 100 words joins the
  one before.
- Stored `text` is exactly the source's words. What gets **embedded** starts
  with `Plutarch, Life of <life>, section <n>.`, and the BM25 column indexes
  the Life's name with the text, because most paragraphs say "he", not the
  person's name.

### Loading it

```
uv run python -m scripts.fetch_corpus            # download once, check the hash
uv run python -m scripts.ingest --dry-run        # chunk counts per Life; no API, no database
uv run python -m scripts.ingest                  # embed and load into dev (.env)
```

`ingest` embeds only the chunks the database doesn't already have, one batch
per transaction, so a second run costs nothing and an interrupted run resumes.
It stops if the spend passes `DAILY_SPEND_CAP_USD`.

Voyage accounts without a payment method are limited to **3 requests and
10K tokens per minute** (Voyage's own 429 message). Adding a payment method
raises the limits, and the free tokens (200M for the Voyage 3 series) still
apply. Without one, the default batch of 64 chunks is refused. Pace the calls:

```
uv run python -m scripts.ingest --batch-size 10 --pause 62   # ~4K tokens per call, one call a minute
```

20-chunk batches (~8K tokens by word count) were still refused: Voyage counts
more tokens per word than the estimate, so leave plenty of room.

A 429 that still gets through is waited out (65 s, up to 5 times); other
errors stop the run, and the next run resumes where it stopped. `--fake-embeddings` loads
free hashed vectors under a separate version (`lives-v1-fake`), so they never
mix with real ones.

### First load (dev, 2026-11-15)

- `lives-v1`: 470 chunks, 160 on a first run that hit Voyage's free-tier
  limit and 310 on the resumed run (`--batch-size 10 --pause 62`, no 429s).
  About 150K tokens in all; the resumed run cost $0.0066.
- A second run: 0 new chunks, 0 tokens.
- BM25 ranked the right Life first for "Utica", "Opimius", "Carrhae Surena"
  and "Tiberius Octavius" (the last uses the Life's name in `tsv`).
- **Vector search can't separate the Gracchi.** "Which brother was killed on
  the Capitol?": Tiberius §25 at 0.5743, Caius §1 at 0.5745 (cosine
  distance), a gap of 0.0002, 50 times closer than in the spike. Hybrid search
  has to fix this, and its eval has to measure it.

### Changing the corpus

Any change to the source file, the Lives or the chunking rules needs a new
`CORPUS_VERSION` (and a new `CHUNKER` name if the rules changed). `ingest`
refuses to load an existing version from a different source or chunker, and
cached answers are keyed to the version, so old answers are never served for
new text.

## Search

Two queries in `core/search.py`; lower scores are better for both, but the
scores can't be compared with each other, so hybrid search combines them by rank.

| Query | Index | Score |
|---|---|---|
| `vector_search` | HNSW on `embedding` (pgvector, cosine) | cosine distance, 0 = identical |
| `bm25_search` | `lakebase_bm25` on the generated `tsv` column (Neon only) | negative BM25; 0 = no shared words |
| `hybrid_search` | both: top 20 from each, fused with RRF | RRF score, higher is better |

### Fusion (RRF)

Each chunk scores `sum(weight / (60 + rank))` over the lists it appears in
(ranks from 1). Ties go to the better single rank, then the lower chunk id, so
results never vary between runs. `FusedHit.ranks` keeps each list's rank, which
is what explains a miss. `SEARCH_VERSION` names the current behaviour and is
saved with every eval result.

### Retrieval eval

```
uv run python -m scripts.retrieval_eval            # against dev; one Voyage call, < $0.0001
uv run python -m scripts.retrieval_eval --compare  # every saved version side by side
```

`evals/retrieval.json` holds 33 questions in four categories: `near_miss` (13:
the right Gracchus or Cato), `named` (8), `paraphrase` (8, no rare keywords),
`shared` (4, events told in more than one Life). Each lists the sections that
count as correct and an exact quote from one of them; `test_retrieval_eval.py`
checks every quote against the text. Results go to
`evals/results/retrieval/<SEARCH_VERSION>.json`.

Metrics: **hit@1** and **hit@5** (an expected section in the top 1 / top 5),
**MRR@10**, and **life@1** (the top hit is from the Life the answer is about,
the near-miss measure).

### Results on `lives-v1` (2026-11-15)

| | hit@1 | hit@5 | MRR | life@1 | near-miss life@1 |
|---|---|---|---|---|---|
| BM25 alone | 0.61 | 0.73 | 0.67 | 0.88 | 0.92 |
| **vector alone** | **0.76** | **0.97** | **0.83** | 0.85 | 0.92 |
| hybrid, `rrf-v1` (equal weights) | 0.70 | 0.91 | 0.79 | 0.85 | 0.92 |

Labels v2. Two expected sections were added after the answer eval cited them:
Elder s7 for nm12 and Pompey s30 for n05; each question's `note` says why.
Under labels v1, hybrid `rrf-v1` scored hit@5 0.85, and `rrf-v2-weighted` (BM25
at 0.5, saved with v1 labels) scored 0.88 with worse hit@1, MRR and near-miss
life@1. 33 questions, so one question is 0.03.

What this shows:

- **Vector search wins on this corpus, and hybrid is worse than vector alone.**
  BM25 locks onto a word that appears in the wrong sections: "How did Cato die
  at Utica?" ranks the many sections that mention Utica (s74 to s84) above the
  death scene (s89 to s93), which never uses the word. "Pompey" matches every
  Pompey chunk, because the Life's name is in `tsv`. Equal-weight RRF lets these
  push correct sections out of the top 5.
- **The near-miss problem is mostly handled already.** Vector search picks the
  right Gracchus or Cato in 12 of 13 near-miss questions, including "Which
  brother was killed on the Capitol?". The likely reason is the
  `Plutarch, Life of <name>` line ingest puts in front of every embedded chunk.
- **Halving BM25's weight didn't fix it.** It was set once, before measuring
  (not tuned on these questions), and kept out: one more question in the top 5,
  but worse hit@1, MRR and near-miss life@1. The default stays `rrf-v1`;
  `hybrid_search(weights=...)` remains for experiments.
- **Answers use hybrid** (decided for Phase 5): it keeps exact matching for rare
  names, which may matter more on the full 52 Lives. `SEARCH_VERSION` is saved
  with every answer, so this can be re-measured.

### Spike results (Neon test branch, 2026-11-15)

Run with `uv run --env-file .env.test python -m scripts.neon_spike` (Voyage
embeddings; needs `VOYAGE_API_KEY`) or with `--fake-embeddings` (no key).

- Neon Postgres 18.7 offers `vector` 0.8.7 and `lakebase_text` 0.1.3. Both
  install with the owner login, through the normal migration runner.
- **The read-only `delphi_query` login can run BM25 queries** (`<@>` and
  `to_bm25query` need no extra grants). The rank_bm25 fallback isn't needed.
- **Whether non-matching rows (score 0) come back depends on the plan.**
  Checked with `EXPLAIN` on 6 rows: the planner chose a sequential scan, which
  returned every row, zeros included; with `enable_seqscan = off` it used the
  `chunks_tsv_bm25` index scan, which returned only the match. `bm25_search`
  always filters out `score = 0`, so a non-match never earns a rank in fusion
  whichever plan runs.
- BM25 told the near-miss passages apart: "Opimius Aventine" ranked Caius
  Gracchus first, "Utica" ranked Cato the Younger first and returned nothing else.
- **Real embeddings (voyage-3.5) rank the near-miss brothers correctly, but
  barely.** For "Which brother was killed on the Capitol by the senators?":
  Tiberius Gracchus 0.5505, Caius Gracchus 0.5597 (cosine distance), a gap
  under 0.01. This is the case the project is built around: BM25 on exact
  names, and the cache's name check, have to cover what vector search alone
  can't separate. Seven short texts cost 241 tokens, $0.000014.
- BM25 statistics refresh at `VACUUM`; the ingest login has `MAINTAIN` on
  `chunks`, so a load can vacuum without the owner login.
- Use the **direct** connection string for `DATABASE_URL` (no `-pooler` in the
  host). The migration runner and `logins.py` need a session, not the pooler.

## Answers (`agent/answer.py`)

```
uv run python -m scripts.ask "How did Cato die at Utica?"
uv run python -m scripts.answer_eval              # 38 questions, about $0.02
uv run python -m scripts.answer_eval --rescore    # re-check saved answers after a label change; no API calls
uv run python -m scripts.answer_eval --compare
```

`answer_question()` embeds the question, takes the top 5 from `hybrid_search`,
and makes one structured call to Claude Haiku 5.5 (`LLM_MODEL`) with prompt
`a1` (`agent/prompts.py`). Passages go in as `<passage id="P1" life=...
section=...>`, the question in `<question>`, all escaped; the model never sees
database ids.

**Every citation is checked in code** before an answer is accepted: it must
name a passage it was given and quote it word for word (whitespace, case,
curly quotes and HTML escapes are forgiven). An answer must cite something; a
decline (`answerable: false`) needn't. A failed check goes back to the model
once, without the quoted text. Claude's built-in citations feature isn't
used: it needs document blocks and can't be combined with structured output.

Each `AnswerResult` records `prompt_version`, `search_version`,
`corpus_version` and model, for the cache to key on. The budget
(`DAILY_SPEND_CAP_USD`) is checked after the embedding and after the answer.

### Answer eval (`a1`, labels v2, 2026-11-15)

| | |
|---|---|
| answered (33 answerable) | 0.94 |
| cites an expected section | 0.91 |
| main citation from the right Life | 0.97 (near-miss: 1.00) |
| declined (5 unanswerable) | 1.00, including two general-knowledge traps |
| sent back once by the citation check | 0.05 |
| cost, time | $0.022 for 38 questions; 2.6 s each |

Checked in code, with no judge model: whether an answer is *correct* beyond
its citations isn't measured yet.

- **The two answerable misses are retrieval misses, and the model said so.**
  "How did Cato die at Utica?" and "What did Pompey do when his army broke at
  Pharsalia?": hybrid search didn't retrieve the scene (the BM25 problem
  above), and the model answered that the passages don't describe it instead
  of answering from memory.
- **`right_life` checks the first citation only.** Answers often add a second
  citation from another Life that retells the event (Pompey s78 for the
  Rubicon, Crassus s14 for the reconciliation) or that tells the brothers apart
  (Caius s1 alongside Tiberius s24). The first version counted those as
  wrong-Life citations (0.76); they aren't.
- Seen while testing: an answer can over-hedge ("killed on the Capitol"
  followed by "the passage does not say he was killed on the Capitol itself").
  A prompt change would be `a2`, measured against `a1`.

## Cache (`agent/cached_answer.py`)

```
uv run python -m scripts.ask "How did Tiberius Gracchus die?"      # uses the cache
uv run python -m scripts.ask --no-cache "..."                       # always fresh
uv run python -m scripts.cache_eval                                 # 38 labelled pairs
uv run python -m scripts.cache_eval --pairs heldout                 # 10 held-out pairs
```

Every question is embedded once. The 5 nearest cached questions for the same
corpus, prompt, search and model versions are fetched (`answer_cache`, HNSW),
the nearest one with the same signature is preferred, and the gate decides:

| Check | Result | Model call |
|---|---|---|
| names, ordinals or numbers differ (`core/signature.py`) | miss | none |
| similarity >= 0.95 | **hit**: the stored answer is served | none |
| similarity < 0.85 | miss | none |
| in between | the judge (Haiku 5.5, prompt `j2`) decides | one small call |

A miss answers fresh and stores the answer; a hit only increments
`hit_count`. Changing the corpus, the answer prompt, the search or the model
starts a fresh cache. The `query` login can read, add and count hits, but not
change or delete answers.

**Signatures** map aliases to one name (Gaius -> Caius; Marcus Cato, the
Censor -> Cato the Elder; Cato of Utica -> Cato the Younger; the elder/younger
Gracchus -> Tiberius/Caius) and treat any other capitalised word as a name. A
bare "Cato" matches neither Cato. They are deliberately strict: "what ransom did
the pirates *first* ask" reads "first" as an ordinal and misses its paraphrase.
A missed hit costs half a cent; a wrong cached answer costs trust.

### Cache eval (2026-11-15)

| | main (38 pairs) | held-out (10 pairs) |
|---|---|---|
| **false hits** (different questions, same answer) | **0** | **0** |
| hit rate on true paraphrases | 0.86 (12 of 14) | 1.00 (4 of 4) |
| judge calls | 11 | 7 |
| cost | $0.0018 | $0.0011 |

- **The signature check did most of the work.** It stopped all 20 pairs that
  differ in a person, ordinal or number, including Cato the Elder vs the
  Younger at 0.969 similarity, which thresholds alone would have served.
- **Judge `j1` gave one false hit** (`c1`, saved): "How many talents did the
  pirates ask from Caesar?" vs "...did Caesar promise to pay the pirates?" It
  assumed both named the same sum; Plutarch has 20 asked and 50 promised.
  `j2` adds one rule: a different action, role or direction (asked vs offered,
  promised vs paid, lost vs had) makes a different question, and the judge
  must not assume two actions are the same fact.
- **How j2 was checked:** the 10 held-out pairs were written and committed
  *before* j2 (commit `3496b7f`). j1 already scored 0 false hits on them; j2
  kept that and fixed nm08, with no loss of paraphrase hits on either set.
  j2's examples are close to nm08's wording, so the held-out set is what
  shows it generalises. The next gate change needs new held-out pairs.
- The two missed paraphrases: pa03 (the "first" ordinal above) and pa14 ("Marcus
  Cato" vs "Cato the Elder"), where the judge rightly noted that both Catos were
  named Marcus.
- On `dev`: "How did Tiberius Gracchus die?" answered fresh ($0.00064); "In what
  way did Tiberius Gracchus die?" served from the cache (0.989, $0); "How did
  Caius Gracchus die?" missed on its signature and was answered fresh.

## Evals

Three evals, each saving one JSON file per version under `evals/results/`:

| Eval | Measures | Cost | Results |
|---|---|---|---|
| `scripts.retrieval_eval` | BM25, vector and hybrid on 33 labelled questions | < $0.0001 | `retrieval/<SEARCH_VERSION>.json` |
| `scripts.answer_eval` | answers and citations on 33 + 5 unanswerable | ~$0.02 | `answers/<PROMPT_VERSION>.json` |
| `scripts.cache_eval [--pairs heldout]` | the cache gate on 38 (+10 held-out) pairs | < $0.002 | `cache/<CACHE_VERSION>[-heldout].json` |

**Where results live.** The JSON files are the source of truth and are
committed with the change they measure. Each run is also recorded in the
`eval_runs` table through the `eval` login, with the commit it ran on (and
whether there were uncommitted changes), so runs that were never committed
aren't lost. Recording never fails an eval: with no `EVAL_DATABASE_URL` it warns
and carries on. `--no-record` skips it.

```
uv run python -m scripts.eval_report                  # current results, from the files
uv run python -m scripts.eval_report --history        # every recorded run
uv run python -m scripts.eval_report --import         # add saved files to eval_runs (safe to repeat)
uv run python -m scripts.eval_report --write-readme   # regenerate the README results tables
```

**The README tables are generated** from the result files for the versions in
the code (`SEARCH_VERSION`, `PROMPT_VERSION`, `CACHE_VERSION`), between the
`<!-- results:start -->` and `<!-- results:end -->` markers. CI runs
`--check-readme` and fails if they don't match, so after any eval that changes a
current version, run `--write-readme` and commit the README with the results.
Imported (backfilled) runs have no commit, because the commit they ran on
isn't known.

**Rules the evals follow:**

- Bump the version (`SEARCH_VERSION`, `PROMPT_VERSION`, `CACHE_VERSION`,
  `JUDGE_PROMPT_VERSION`) whenever behaviour changes, so results never mix.
- Set thresholds and weights before measuring, not by tuning on the eval set.
- Before fixing a failure, write and commit **new held-out cases** that the fix
  hasn't seen; keep the fix only if it holds on them too.
- Corrections to labels are made openly: bump `labels` in the eval file, add a
  `note` to each changed question, and rerun (or `answer_eval --rescore`, which
  re-checks saved answers without API calls).
- The Voyage free tier allows 3 requests a minute: leave about a minute between
  eval runs and `scripts.ask`.

## Website

Live: **https://delphi-rag.delphi-rag.workers.dev**

```text
browser ── static site (web/, Cloudflare static assets: delphi-rag)
   │  POST /api/ask {question, turnstileToken}
   ▼
Worker (worker/, delphi-api.delphi-rag.workers.dev)
   1. origin must be the site     4. Turnstile token checked
   2. 1-300 characters            5. site-wide limit: 3 a minute
   3. per-IP limit: 5 a minute
   │  X-Delphi-Key: <shared secret>
   ▼
Python API (api/app.py, Google Cloud Run: delphi-api, us-west1)
   daily cap $0.10 (SITE_DAILY_CAP_USD) -> over it: cached answers only
   answer_with_cache() -> Neon production branch (delphi_query login)
```

| Piece | Where | Secrets |
|---|---|---|
| Site | `web/`, Cloudflare Worker `delphi-rag` (static assets only) | none (Worker URL and Turnstile site key are public, in `js/config.js`) |
| Worker | `worker/`, Cloudflare Worker `delphi-api` | `API_SHARED_SECRET`, `TURNSTILE_SECRET_KEY` (wrangler secrets) |
| API | Cloud Run service `delphi-api`, project `delphi-rag` | `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`, `QUERY_DATABASE_URL`, `API_SHARED_SECRET` (Secret Manager) |
| Database | Neon branch `production` (copied from `dev`) | owner URL and logins in `.env.production` |

**Limits.** The Worker refuses another site's requests, empty or over-long
questions, more than 5 questions a minute from one visitor, requests without
a valid Turnstile token, and more than 3 a minute for the whole site (Voyage's
free tier). Cloudflare's rate limits are approximate: in a local burst the
site limit refused the 3rd question instead of the 4th. When one too many
gets through, Voyage refuses the embedding and the API answers 503 "Busy, try
again in a minute", so visitors still get a friendly message. The API checks
the length again and enforces the daily cap: past $0.10 (UTC day) it serves
only clear cache hits, with no judge or answer call, and otherwise answers
`over_cap`. Two requests arriving together can both pass the cap check, so a
day can end up to about one answer over. `daily_spend` on production counts
fresh answers, cache hits and refusals per day.

### Deploying

```
# API (Cloud Run). Secrets come from Secret Manager; refresh them from .env.production with:
scripts/deploy_api.sh --create-secrets      # adds new versions, prints nothing secret
scripts/deploy_api.sh                       # build and deploy (~3-5 minutes)

# Worker
cd worker && npx wrangler deploy
printf '%s' "<value>" | npx wrangler secret put API_SHARED_SECRET   # or let wrangler prompt

# Site
cd web && npx wrangler@4.155.0 deploy
```

After any address change, update all four places: `web/js/config.js`
(`workerUrl`), the `connect-src` in `web/_headers`, `ALLOWED_ORIGINS` in
`worker/wrangler.toml`, and the hostname list of the Turnstile widget.

**Rolling back.** API: `gcloud run services update-traffic delphi-api
--region us-west1 --to-revisions <revision>=100` (list them with `gcloud run
revisions list --service delphi-api --region us-west1`). Worker or site:
`npx wrangler rollback` in its folder.

**Cache-only mode, by hand:** `gcloud run services update delphi-api --region
us-west1 --update-env-vars SITE_DAILY_CAP_USD=0`, and back with `=0.10`.

### Lessons from the first deploy (2026-11-16)

- **Cloud Build needs a role on new projects.** The build runs as the
  default compute service account, which new projects no longer give broad
  permissions; grant it `roles/run.builder` once. Cloud Run also needs
  `roles/secretmanager.secretAccessor` on each of the four secrets.
- **`wrangler pages project create` now creates a Worker** from the current
  folder. Run from `worker/`, it published a copy of the API Worker under the
  site's name (no secrets, so it could only refuse requests); deploying the
  site under the same name replaced it. The site now deploys as a static-assets
  Worker from `web/`, with `.assetsignore` keeping `wrangler.toml` unpublished.
- **The pinned wrangler's runtime only supports compatibility dates up to
  2026-10-30**, and the Workers runtime treats every named export of the entry
  module as a handler: exporting a constant made the Worker fail at startup.
  Both were caught by running it locally (`wrangler dev`) before deploying.
- **The workers.dev subdomain is per account** and appears in every address;
  it was renamed from the account's default to `delphi-rag`, after which DNS
  took a few minutes, and Turnstile needed the new hostname.
- **Testing the Worker locally on Windows:** vitest can fail with "failed to
  find the runner" when the working directory's drive letter is lower case
  (`c:\`); run it from `C:\` (PowerShell `Set-Location 'C:\...'`). CI is unaffected.
- Turnstile only passes in a real browser, so live tests of the full path are
  manual; the refusal paths are checked with curl, and the API directly with
  the shared secret.


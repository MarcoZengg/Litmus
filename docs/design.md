# Litmus design

> **Status:** complete draft, ready for review. This file is public: no client
> names or client content.

## 1. Problem

Teams building LLM apps change prompts, models and retrieval settings all the
time. To check a change, most run their test questions once and compare the
score with the previous one. But an LLM's answers vary from run to run, and a
small test set's score depends on which questions happen to be in it. So a few
points up or down is often just noise. Teams ship regressions they never saw,
or roll back improvements that were real.

Litmus answers one question: **did this change really make my app better or
worse?** It runs each question several times, compares two runs question by
question, and reports every score with a range. A change is called real only
when the evidence supports it.

**Why it exists.** Litmus started while preparing a client's customer-support
agent for launch. Deciding whether it was ready needed more than one run's
score. It needed evidence that the answers were correct, that the agent didn't
make up answers it shouldn't give, and that the numbers would hold up on a
second run.

## 2. Who it's for

**Main user:** a small team or a solo developer running a RAG support or Q&A
bot, without a dedicated evaluation team. They have a few dozen to a few
hundred test questions, a paid model API and limited time.

What they want from Litmus:

- **Before a change ships:** to know whether it helped, hurt or made no
  difference, and to have CI fail on a real regression.
- **Before launch:** a go/no-go check against a bar set in advance, including
  how often the bot answers when it shouldn't.
- **When switching models:** evidence that a cheaper or fine-tuned model is
  good enough.
- **With private data:** to evaluate on a client's data without that data
  reaching providers the client hasn't approved.

**Not the main user in v1:** teams evaluating multi-step agents, and
researchers benchmarking models on academic tasks (tools like
lm-evaluation-harness serve them).

## 3. Scope and non-goals

**In scope for v1:**
- Any LLM app, through the HTTP, OpenAI-compatible or Python function adapter,
  with generic checks: exact match, JSON schema, rubric judge (sections 4
  and 6).
- Deep metrics for RAG apps: recall@k, MRR, faithfulness, correctness and
  behavior (section 6).
- Comparisons that separate real changes from noise, plus CI gates (section 7).
- Private datasets (section 8) and a public demo (section 9).

**Non-goals for v1:**
- **Agent evaluation:** tool calls and multi-step trajectories.
- **A web UI or hosted service.** Litmus runs locally from the command line
  and in CI.
- **Competing on number of metrics.** promptfoo, DeepEval and Ragas cover
  breadth. Litmus goes deep on trustworthy comparisons.
- **Multi-turn conversations and streaming responses.**
- **Generating test questions automatically.**

### Commands

| Command | What it does |
|---|---|
| `litmus validate <dataset> [--app app.yaml]` | Checks the dataset against section 5's rules. With `--app`, also checks the app config's request templates against every record. Makes no calls |
| `litmus run <dataset> --app app.yaml --name baseline` | Runs every check `validate` does, with unset secrets as errors. Shows the estimated number of app and judge calls, asks to continue, then runs and writes `runs/<date>_<name>/`. Repeat 1 covers every question before repeat 2 starts, so a run stopped early covers as many questions as possible. Never overwrites an existing run. `--repeats` overrides the default of 3. `--yes` skips the question in CI. `--judge judge.yaml` grades the run when the app calls are done. Exits non-zero if the run or its judging is incomplete |
| `litmus judge <run> --judge judge.yaml` | Grades a finished run with the judge model (section 6), without calling the app again, and writes `judgments.jsonl` and `judging.yaml` in the run folder. Shows the number of judge calls and asks to continue. Refuses a run that's already judged unless `--replace` is given. `--yes` skips the question |
| `litmus label <run> --metric correctness` | Writes `labels-<metric>.csv` in the run folder: a sample (default 50) of the replies the judge grades for that metric, with the question, reference and answer, for you to label pass or fail by hand. Never shows the judge's verdicts, and never overwrites a sheet unless `--replace` is given (section 7) |
| `litmus report <run>` | Shows the metrics for one run, with breakdowns and the IDs of questions to look at, and saves aggregate numbers (no IDs, no text) as `metrics.json` in the run folder. Never shows question or answer text |
| `litmus compare <baseline> <candidate>` | Paired comparison with a verdict per metric (section 7). Shows numbers only, no question IDs or text. `--gates gates.yaml` makes it exit non-zero when a gate fails, so CI can use it. `--force` overrides the comparability check |

### Stack

Python 3.12, Typer (commands), httpx (HTTP adapter), PyYAML, jmespath, NumPy
(bootstrap) and pytest.

## 4. How apps connect

**Decided:**
- Three adapters: HTTP endpoint, OpenAI-compatible API and Python function.
- Every adapter returns the same normalized response, so metrics never depend
  on which adapter was used.
- The HTTP adapter finds fields in the app's JSON reply with
  [JMESPath](https://jmespath.org/) expressions.
- Behavior comes from the app's own status field when it has one, and from the
  judge otherwise.
- Failed calls are not retried by default. They count as failures, and the
  error rate is reported separately.
- Dataset records can carry `metadata` that request templates use (section 5).

### Two integration levels

| Level | The app returns | Metrics available |
|---|---|---|
| Black-box | An answer | Generic checks, correctness, behavior accuracy, latency, cost |
| Glass-box | An answer plus the passages it retrieved: IDs in rank order, optionally text | All of the above, plus recall@k, MRR and faithfulness |

Cited passages are not retrieved passages. Some apps return only the passages
their answer cites. Those aren't the ranked list the retriever produced, and
treating them as retrieved would make recall@k and MRR misleading. A separate
citation metric is an open question (section 11).

### The normalized response

| Field | Meaning | Filled by |
|---|---|---|
| `actual_output` | The answer text | Adapter (required) |
| `retrieved_context_ids` | Passage IDs, in rank order | Adapter (glass-box) |
| `retrieval_context` | Passage texts | Adapter (optional) |
| `status` | The app's own status value, unchanged | Adapter (optional) |
| `observed_behavior` | One of the five behaviors | Litmus |
| `behavior_source` | `status` or `judge`: how the behavior was decided | Litmus |
| `cached` | True if the app served a saved answer | Adapter (optional) |
| `input_tokens`, `output_tokens`, `cost` | Usage | Adapter (optional) |
| `kb_version` | Knowledge-base version the app used | Adapter (optional) |
| `latency_ms` | Wall-clock time around the call | Litmus |
| `error` | Error type and message | Litmus |

Litmus measures latency itself, around the call, so it is comparable across
adapters and includes network time.

### App config

Each app is described in a YAML file. Fields the config doesn't map are simply
unavailable; for example, no `retrieved_context_ids` means black-box.

Example for an invented app:

```yaml
name: x200-support-bot
adapter: http
request:
  method: POST
  url: https://staging.example.com/api/chat
  headers:
    Authorization: "Bearer ${X200_BOT_TOKEN}"   # read from an environment variable
  body:
    message: "{{input}}"
    category: "{{metadata.category?}}"          # left out when a record has none
  timeout_s: 60
response:                    # JMESPath expressions into the JSON reply
  actual_output: reply.answer
  retrieved_context_ids: debug.retrieved[*].id
  status: reply.status
  cached: reply.from_cache
  input_tokens: usage.input_tokens
  output_tokens: usage.output_tokens
behavior_map:                # app status -> Litmus behavior
  ok: answer
  partial: partial
  need_more_info: clarify
  out_of_scope: refuse
  escalate: handoff
limits:
  concurrency: 1
  requests_per_minute: 20
  max_requests: 300          # hard stop for this run
```

Templates can use `{{input}}` and `{{metadata.<key>}}`. A `?` makes a key
optional: `{{metadata.category?}}` leaves that field out of the request when a
record has no `category`. That matters for questions that are deliberately
vague, like `clarify` questions without a product. If a record lacks a
*required* key, the dataset fails validation before the run starts, so no paid
calls are wasted.

### OpenAI-compatible adapter

Evaluates a model directly, without a RAG app. It sends `input` as the user
message, with an optional system prompt, and reads the reply. It is
black-box only. Use it to compare models on the same dataset, for example a
base model against a fine-tuned one.

Config: `base_url`, `model`, the API key's environment variable,
`system_prompt`, `temperature`.

### Python function adapter

Config: `function: my_app.eval:answer`. Litmus calls the function with the
dataset record (`input` and `metadata`) and expects a dict with the normalized
fields. Useful for apps you can import, and for Litmus's own tests.

### Behavior detection

1. If the config has a `behavior_map` and the response has a `status`, Litmus
   maps it. A status missing from the map falls back to the judge, with a
   warning.
2. Otherwise the judge reads the question and answer and picks one of the five
   behaviors, using the definitions in section 5.

Each record stores `behavior_source`, so reports can show how many behaviors
the judge decided.

### Errors, timeouts and limits

- **No retries by default.** With `retries: N`, only timeouts, connection
  errors, HTTP 429 and 5xx are retried, waiting 1, 2, 4… seconds (at most 10)
  in between. Every attempt counts toward `max_requests`.
- **Redirects aren't followed.** A 3xx reply is an error, so a request can't be
  sent on to a host outside `allowed_hosts`.
- **GET requests** send the body as query parameters; other methods send it as
  JSON.
- **These count as errors:** a non-2xx HTTP status, a timeout, invalid JSON,
  or an `actual_output` expression that finds nothing.
- **A failed call** sets `error`. The question is scored as failed for answer
  and behavior metrics, and left out of latency percentiles. Each run reports
  its error rate.
- **Limits:** `concurrency` (default 1), `requests_per_minute` and
  `max_requests`. If a run reaches `max_requests`, it stops and is marked
  incomplete. This protects paid APIs and apps with daily caps.

### Secrets

Secrets come only from environment variables (`${NAME}`) and are never written
in config files. `run.yaml` saves the config as written, so it holds the
`${NAME}` references, not the values. If a secret-looking header or URL
parameter is written out literally anyway, Litmus warns and saves it as
`<redacted>`.

### Caching

Some apps return a saved answer when they see the same question again. Then
repeated runs (section 7) measure nothing. If the adapter maps `cached`,
Litmus warns when repeats came from the cache. It also warns when almost
every question gave identical answers on every repeat, which is a hint to
check for caching.

## 5. Dataset format

Background: [research/dataset-formats.md](research/dataset-formats.md).

**Decided:**
- One question per line, in JSONL.
- Field names follow DeepEval (`input`, `expected_output`, `actual_output`),
  plus Ragas's `reference_context_ids` / `reference_contexts`. Importing from
  either tool is a simple rename.
- Only `id` and `input` are required. Everything else is optional, so a new
  user can start with a list of questions and add labels later.
- Two fields neither tool has: `expected_behavior` and `source`.
- Correct passages are written as groups of alternatives.
- Run results are written next to the dataset.

### File layout

```text
my-dataset/
├── dataset.yaml       # manifest: name, version, kb_version, language, privacy (section 8)
├── dataset.jsonl      # the questions, written by hand
└── runs/              # written by Litmus, one folder per run
    └── 2026-11-02_baseline/
        ├── run.yaml         # app, adapter, model settings, dataset hash, Litmus version
        ├── records.jsonl    # one line per question per repeat
        ├── judging.yaml     # judge model, settings and prompt hashes (litmus judge)
        ├── judgments.jsonl  # one line per question, repeat and judge metric
        ├── labels-correctness.csv  # your hand labels, to check the judge (litmus label)
        └── metrics.json     # aggregate numbers only (litmus report)
```

Because results live beside the dataset, a private dataset's results stay in
the private folder automatically. A run can override the output path.

### Dataset record

| Field | Required | Type | Meaning |
|---|---|---|---|
| `id` | yes | string | Unique, stable ID, e.g. `q-0001` |
| `input` | yes | string | The question, as the user would type it |
| `expected_output` | no | string | Reference answer |
| `reference_context_ids` | no | list | Correct passages, as groups (see below) |
| `reference_contexts` | no | list of strings | Text of correct passages, for apps that don't expose IDs |
| `expected_behavior` | no | string | One of the five behaviors below |
| `tags` | no | list of strings | Topics, e.g. `["battery"]` |
| `language` | no | string | e.g. `zh`, `en`. Defaults to the manifest's language |
| `source` | no | string | `staff`, `real_traffic`, `faq`, `synthetic` or `mixed` |
| `metadata` | no | object | App-specific inputs, e.g. `{"category": "air purifier"}`. Request templates use them as `{{metadata.category}}` (section 4) |
| `notes` | no | string | Labeler's notes |

### Expected behaviors

| Value | The app should... |
|---|---|
| `answer` | Answer fully from the knowledge base |
| `partial` | Answer part of it and say what it can't confirm |
| `clarify` | Ask the user for missing details, e.g. which model |
| `refuse` | Decline: out of scope or not allowed |
| `handoff` | Direct the user to a human instead of answering |

Labeling rule: pick the single best label. If the reply should contain any
real answer, it's `answer` or `partial`, even if it also suggests contacting a
human.

### Correct passages: groups

`reference_context_ids` is a list of groups. Each group is a list of
alternative passage IDs, and any one of them counts. A plain ID is shorthand
for a group of one.

- `["p12"]`: passage p12 is needed.
- `[["p4", "faq-7"], "p15"]`: needs (p4 **or** faq-7) **and** p15.
- `[]`: no passage should answer this. Typical for `clarify`, `refuse` and
  `handoff`.
- Field missing: not labeled. Retrieval metrics skip this question.

With groups, retrieval metrics are exact (details in section 6):
- **recall@k:** the share of groups with at least one ID in the top k results.
- **MRR:** 1 / the rank of the first retrieved passage that belongs to any group.

Passage IDs are opaque strings. They must match the IDs the app returns, and
Litmus never parses them. If the knowledge base is re-chunked, IDs can change,
so `dataset.yaml` records a `kb_version`. If the app reports a different one,
Litmus warns that the labels may be stale.

### Examples (invented, not client data)

```jsonl
{"id": "q-0001", "input": "How long does the X200's battery last on a full charge?", "expected_output": "About 6 hours at normal fan speed.", "reference_context_ids": ["x200-manual#p12"], "expected_behavior": "answer", "tags": ["battery"], "language": "en", "source": "real_traffic"}
{"id": "q-0002", "input": "How loud is the X200, and how often do I change the filter?", "expected_output": "About 38 dB on low; replace the filter every 6 months.", "reference_context_ids": [["x200-manual#p4", "x200-faq#7"], "x200-manual#p15"], "expected_behavior": "answer", "tags": ["noise", "maintenance"], "source": "staff"}
{"id": "q-0003", "input": "What's the battery life?", "reference_context_ids": [], "expected_behavior": "clarify", "source": "real_traffic", "notes": "No product named; the agent should ask which one."}
{"id": "q-0004", "input": "Does the X200 at my site have the HEPA upgrade installed?", "reference_context_ids": [], "expected_behavior": "handoff", "source": "staff", "notes": "The manual can't know one site's setup."}
```

### Run record

Litmus writes one line per question per repeat: `id` (the dataset record),
`repeat` (1, 2, 3...), every field of the normalized response from section 4
(`actual_output`, `observed_behavior`, `retrieved_context_ids`, `latency_ms`,
`error` and so on), and `scores` (metric name → value).

### Checks when loading a dataset

- Every `id` is unique and every `input` is non-empty.
- `expected_behavior` and `source` use the allowed values.
- All records are single-turn. Multi-turn comes after v1.
- Warn when `expected_behavior` is `answer` but there's no `expected_output`
  and no `reference_context_ids`, because only a rubric judge can score it.

### Importing

| From | Rename |
|---|---|
| Ragas | `user_input` → `input`, `reference` → `expected_output`; `reference_context_ids` and `reference_contexts` stay the same |
| DeepEval | `input` and `expected_output` stay the same; `context` → `reference_contexts`; `additional_metadata` → `metadata` |

## 6. Metrics

**Decided:**
- Judge-based metrics give **pass/fail plus a one-sentence reason** per
  question. The aggregate is a pass rate.
- Faithfulness uses **one judge call per answer**. The reason lists any
  unsupported claims.
- **No overall score.** Each metric is reported on its own. CI pass/fail comes
  from gate rules (section 7).
- Behavior is reported as an **expected-vs-observed matrix plus two named
  error rates**.

### Rules that apply to every metric

- **Each metric scores each question, then aggregates.**
- **Coverage is always shown.** A question that lacks what a metric needs is
  skipped for that metric, and the report says so, e.g. "correctness: 52 of
  70 questions scored".
- **Failed calls count as failures** in every metric except latency (section 4).
  For the two behavior rates, a failed call gave the user no answer: it counts
  as a missed answer, but not as an unsafe one.
- **Unknown behavior waits for the judge.** A repeat whose behavior couldn't be
  read from the app's status is left out of the behavior metrics, not counted
  as wrong.
- **MRR** uses the whole ranked list the app returned, not just the top 5.
- **Retrieval metrics need a glass-box app.** If the app config doesn't read
  `retrieved_context_ids`, recall and MRR aren't measured at all, and a failed
  call doesn't count as a retrieval score of 0.
- **Breakdowns:** every metric is reported overall, by `source` and by `tags`.
  This is how "real customer questions" and "staff questions" get separate
  numbers.

### v1 metrics

| Metric | Level | Needs | Per question | Aggregate | Judge? |
|---|---|---|---|---|---|
| `exact_match` | black-box | `expected_output` | pass/fail after trimming whitespace and case | pass rate | no |
| `json_schema` | black-box | a JSON schema in the run config | pass/fail: parses and validates | pass rate | no |
| `rubric` | black-box | criteria written in the judge config | pass/fail + reason | pass rate | yes |
| `behavior_accuracy` | black-box | `expected_behavior` | match / no match | accuracy, plus the matrix and the two rates below | via status or judge |
| `unsafe_answer_rate` | black-box | `expected_behavior` of `clarify`, `refuse` or `handoff` | answered or not | rate (lower is better) | via status or judge |
| `missed_answer_rate` | black-box | `expected_behavior` of `answer` | didn't answer or did | rate (lower is better) | via status or judge |
| `correctness` | black-box | `expected_output`; expected behavior `answer` | pass/fail + reason | pass rate | yes |
| `recall@k` | glass-box | `reference_context_ids` | share of groups with an ID in the top k | mean, at k = 1, 3, 5 | no |
| `mrr` | glass-box | `reference_context_ids` | 1 / rank of first passage in any group (0 if none) | mean | no |
| `faithfulness` | glass-box | passage text (`retrieval_context`); app answered | pass/fail + unsupported claims | pass rate | yes |
| `latency` | both | successful calls | ms | p50, p95 | no |
| `errors` | both | — | failed or not | error rate | no |
| `cost` | both | token usage or reported cost | app cost | per 1,000 questions | no |

Judge calls have their own cost, reported separately from the app's cost, so a
user can see what the evaluation itself costs.

### Behavior: matrix and two error rates

The matrix counts expected behavior (rows) against observed behavior (columns)
for all five behaviors. Two rates summarize the failures that matter most:

- **Unsafe answer rate:** of the questions where the app should have clarified,
  refused or handed off, the share where it answered or partially answered
  instead. This is the overconfident failure: a support agent making up an
  answer. It's usually fixed with prompts and guardrails.
- **Missed answer rate:** of the questions where the app should have answered,
  the share where it clarified, refused or handed off. This is the overcautious
  failure. It usually means retrieval didn't find the passage.

Mixing up `answer` and `partial` shows in the matrix but counts in neither rate.

### The judge

- **Judging is a separate step** (`litmus judge`, or `litmus run --judge`). It
  reads a finished run folder, so a run can be judged, or re-judged with a new
  model or prompt, without calling the app again.
- **Any OpenAI-compatible chat model can judge.** A judge config names the
  provider, base URL, model, the environment variable holding the API key, the
  metrics to compute (default: correctness, faithfulness, behavior), rubric
  criteria and limits. The provider must be in the dataset's
  `allowed_providers` (section 8).
- **Each judge metric has a written prompt in the repo**, with temperature 0.
  Each prompt's hash goes into `judging.yaml`, so results from different prompt
  versions are never compared by mistake.
- **Users can override a prompt per dataset,** for example to write it in the
  dataset's language.
- **Correctness:** the judge sees the question, the reference answer and the
  app's answer. It passes when the app's answer states the same facts the
  question asks about, without contradicting the reference. Wording can differ.
  Only questions expected to be answered in full are scored: a `partial`
  answer leaves out part of the reference by design, so it's checked by
  behavior instead.
- **Faithfulness:** the judge sees the answer and the retrieved passage text. It
  passes when every factual claim in the answer is supported by those passages.
- **Behavior (fallback when there's no status):** the judge sees the question
  and answer, picks one of the five behaviors using the definitions in
  section 5, and gives a reason. The report marks these as decided by the judge.
- **Some verdicts need no judge call.** A failed app call fails correctness,
  faithfulness and rubric. A reply that clarified, refused or handed off fails
  correctness and isn't checked for faithfulness, since it makes no claims.
  These lines are marked `auto` in `judgments.jsonl`.
- **A failed judge call scores nothing.** It's recorded with its error type,
  left out of the metric, and counted in a warning on the report. An
  unreadable verdict counts as a failed call.
- **How far the judge can be trusted** is measured in section 7.

## 7. Trustworthy comparisons

This is what sets Litmus apart: it says whether a change is real or noise.

**Decided:**
- **3 repeats** per question by default. Configurable, and Litmus shows the
  estimated number of app and judge calls before starting.
- Two runs are compared with a **paired bootstrap** (10,000 resamples, 95%
  interval).
- A **regression** needs both conditions: the interval for the change is
  entirely below zero, *and* the drop is at least a minimum size (default 2
  percentage points for rates and scores; 10% of the baseline for latency,
  cost and tokens).
- **The judge is measured, not corrected.** Litmus compares it with hand
  labels, reports agreement and Cohen's kappa, and warns when they're low.

### Two sources of uncertainty

1. **Which questions happen to be in the dataset.** A different 70 questions
   would give a different score. Resampling questions (the bootstrap) covers
   this.
2. **Run-to-run randomness** in the app and judge. Repeats cover this.

A question's score is the mean over its repeats: passing 2 of 3 scores 0.67.
The bootstrap resamples whole questions, not single repeats, because repeats
of the same question aren't independent.

### Numbers for a single run

- **Every metric gets a 95% interval,** e.g. "correctness 81% (74–87%),
  70 questions × 3 repeats", from 10,000 resamples of the questions, seeded
  with the run's seed so a report can be reproduced exactly. Rates over calls
  (errors, cost, tokens) and latency percentiles resample questions too, taking
  all of each drawn question's calls.
- **Flip rate:** the share of questions with two or more scored repeats whose
  score changed between repeats. A high flip rate means the app is flaky,
  whatever its average. It's shown for every metric scored per repeat.

### Comparing two runs

**Runs must be comparable.** Litmus refuses to compare runs that used a
different dataset, or a judge metric graded with a different provider, model
or prompt. `run.yaml` stores a hash of `dataset.jsonl` and `judging.yaml` a
hash of each judge prompt, so an edited dataset counts as different even if
nobody bumped its version. A `--force` option overrides this, with a warning.
A different `kb_version` gives a warning only. If only one run was judged,
judge metrics aren't compared.

**How the comparison works:**
1. For each question scored in both runs, take the difference: candidate score
   minus baseline score.
2. Resample the questions with replacement 10,000 times, and take the mean
   difference each time. Each resample draws the same questions from both
   runs. Call metrics are recomputed on the drawn questions' calls in each run.
3. The 95% interval runs from the 2.5th to the 97.5th percentile of those
   means. Both runs' seeds from `run.yaml` seed it, so the result can be
   reproduced exactly.

Pairing compares each question with itself. Because some questions are just
harder than others, pairing cancels that out and detects smaller real changes
than comparing two separate pass rates.

**Verdicts for each metric** (flip the signs for metrics where lower is
better: unsafe and missed answer rates, errors, latency, cost):

| Verdict | Condition | Fails CI? |
|---|---|---|
| Regression | Interval entirely below zero, and drop ≥ minimum size | Yes, if the metric is a gate |
| Small but real drop | Interval entirely below zero, but drop < minimum size | No, shown as a note |
| No detectable change | Interval includes zero | No |
| Small but real gain | Interval entirely above zero, but gain < minimum size | No |
| Improvement | Interval entirely above zero, and gain ≥ minimum size | No |

**Smallest detectable change:** the comparison also gives the interval's
half-width, e.g. "with this dataset, changes smaller than about ±5 points can't
be told apart from noise". That tells users when they need more questions.

### Gates for CI

A gates file, passed with `--gates`, lists which metrics gate CI. A
recommended starting set:

```yaml
gates:
  correctness:        {min_effect: 0.02}
  behavior_accuracy:  {min_effect: 0.02}
  unsafe_answer_rate: {min_effect: 0.02}   # lower is better
  errors:             {max: 0.05}          # absolute cap, no comparison needed
```

- A gate with `min_effect`, or with nothing (`{}`, the default minimum
  change), fails on a **regression**. Its `min_effect` also sets that
  metric's verdict in the table.
- `max` and `min` are **absolute caps** on the candidate's value. A gate with
  only caps doesn't compare the runs.
- **A gate whose metric wasn't measured fails,** e.g. `correctness` when a run
  wasn't judged, so a misconfigured run can't pass CI by accident.
- Without `--gates`, compare shows the verdicts and always exits 0.

**Only overall numbers gate.** Breakdowns by `source` and `tags` are shown but
never fail CI. With many breakdowns, some would show a "significant" drop by
chance alone.

### How far the judge can be trusted

- **Hand-label a sample** for each judge metric: aim for 50+ replies; for the
  behavior judge, cover all five behaviors. `litmus label` picks the sample
  from the replies the judge grades (not the ones decided without a call),
  one repeat of every question before a second, using the run's seed. Label
  before looking at the judge's verdicts.
- **Labels belong to a run's replies, not to a judge.** Re-judging a run with a
  new model or prompt reuses its labels, which is how a new judge prompt gets
  checked.
- **Report agreement and Cohen's kappa.** Kappa corrects for agreement by
  chance. If 90% of answers pass, a judge that always says "pass" agrees 90% of
  the time but has a kappa of 0. For pass/fail metrics the report also counts
  where the judge was too lenient (passed what you failed) and too strict.
- **Warn when kappa is below 0.6.**
- **Every judge metric in a report shows its judge's agreement,** and which
  runs and dataset version it was checked on. The check pools the labels from
  every run of the dataset that was judged with the same provider, model and
  prompt hash, so a second run gets the first run's check. Changing the judge
  model or prompt triggers a warning that the judge hasn't been validated yet.
- **Later version:** use the measured judge error rates to correct pass rates
  and widen their intervals.

## 8. Privacy rules

**Decided:**
- **Private datasets live outside the repo** and are loaded by path. Their run
  records are written beside them (section 5), never into the repo.
- **`private: true`** in `dataset.yaml` hides question, answer and passage
  text from everything Litmus shows.
- **`allowed_providers`** in `dataset.yaml` is enforced: Litmus refuses to
  start a run that would send the data to any other provider.
- **Private datasets never run in public CI.** Public GitHub Actions logs are
  public.
- **Secrets only come from environment variables** (section 4).
- **Litmus sends no telemetry.**

Example manifest for a private dataset:

```yaml
name: support-eval
version: 3
kb_version: "2026-10-15"
language: zh
private: true
allowed_providers: [openai]
allowed_hosts: [staging.example.com]
```

### What `private: true` hides

| Where | Shows | Hidden |
|---|---|---|
| Console output | IDs, verdicts, numbers | Question, answer and passage text |
| Reports | Aggregate tables; per-question IDs with pass/fail | All text |
| Error messages | Error type, question ID | Request and response bodies |
| Run records (in the private folder) | Everything | Nothing |

Output from a private run is therefore safe to paste into an issue or share as
a report.

### What `allowed_providers` covers

- **Every model Litmus calls itself:** the judge, and the OpenAI-compatible
  adapter. Each config names its `provider`, and a run is refused if any of
  them isn't on the list.
- **The app's URL,** for the HTTP adapter, through the optional
  `allowed_hosts` list.
- **Not the app's internals.** Where an app sends data internally is outside
  Litmus's control. That stays the app owner's responsibility.

## 9. Public demo

**Decided:** an invented product manual, in English and Chinese, with recorded
runs so the demo works without an API key, plus a live mode.

### What's in `examples/x200/`

| Part | Contents |
|---|---|
| `docs/` | A ~20-page manual for the X200, an invented portable air purifier, in English and Chinese. Written for the demo, under the repo's license |
| `dataset/` | `dataset.yaml` and ~30 hand-written questions covering all five behaviors in both languages, including some that need passage groups |
| `app/` | A tiny glass-box RAG app: a simple retriever plus any OpenAI-compatible model. Returns the answer, retrieved passage IDs and text, and a status |
| `runs/` | Two recorded runs: `baseline`, and `broken`, where the retriever returns only one passage and the prompt never hands off |

Suggested question mix: about 14 answer, 5 partial, 4 clarify, 4 refuse,
3 handoff; about 20 in English and 10 in Chinese.

### The five-minute path

1. Clone and install.
2. Compare the two recorded runs. No key is needed.
3. Litmus reports regressions in recall@k and the unsafe answer rate, with
   intervals, and the CI gate fails. This shows in seconds what Litmus is for.
4. Optional live mode: set an API key for any OpenAI-compatible endpoint and
   run the app for real.

The recorded runs note which model and date produced them, in `run.yaml`.

### Why an invented manual

- No license or attribution issues.
- It has the shape of a real support case: specs, maintenance, models.
- You decide which facts exist. That makes it easy to write questions whose
  correct behavior is clarify, refuse or hand off.
- Two languages show that Litmus isn't English-only.

## 10. Milestones

About six weeks. "Case study" means the first private app evaluated with
Litmus.

| When | Milestone | Done means |
|---|---|---|
| Week 1 | Design and data | This doc reviewed; first 50 case-study questions labeled privately; demo manual drafted |
| Week 2 | Baseline | Litmus runs the case-study app through the HTTP adapter; behavior, retrieval, latency, error and cost metrics recorded |
| Week 3 | Trustworthy evals | Judge metrics (correctness, faithfulness, rubric) added and checked against hand labels; intervals, paired comparisons and gates working |
| Weeks 3–4 | Public demo | Recorded runs in place; the five-minute path works from a fresh clone |
| Weeks 3–4 | Fine-tuning gate | Small open models scored against the case-study baseline; go/no-go decision |
| Before the case study launches | Launch check | The case-study app passes gates set in advance |
| Weeks 4–5 | Fine-tune (if go) | The fine-tuned model is compared in the same way |
| Week 6 | v0.1 release | README, demo, and a write-up with aggregate case-study numbers |

## 11. Open questions

- A citation metric for apps that return cited passages (`cited_context_ids`).
- Streaming responses: not in v1.
- Multi-turn dataset format, after v1.
- Which judge model per dataset, and what does it cost?
- A `contains` / keyword check would need a new dataset field. Is it worth
  adding?
- Should the minimum change size for latency and cost be relative (e.g. 10%)
  rather than in absolute units?
- Correcting judge-metric scores for measured judge error (after v1).
- **Comparing answer models inside a RAG app** (the fine-tuning gate). The
  OpenAI-compatible adapter only sends the question, without retrieved
  passages, so it isn't a fair test of a RAG answer model. Either switch the
  model inside the app (e.g. a staging setting) and call it through the HTTP
  adapter, or replay saved retrieval results to the model. Decide before
  week 3.
- For apps with daily call caps, are 3 repeats affordable, or does the dataset
  need splitting across days?

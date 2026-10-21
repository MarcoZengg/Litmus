# Dataset formats: how Ragas and DeepEval store test cases

Research notes, October 2026. Both libraries change quickly, so check the linked
docs before relying on exact field names.

## The one idea to get first

Evaluation tools work with two kinds of record:

- **The expected result** (DeepEval calls it a *golden*). You write it by hand,
  before running anything: the question, plus what a good result looks like. You
  reuse it on every run.
- **What actually happened** (a *test case* or *sample*). This is the expected
  result plus what the app did on one run: its answer, the passages it
  retrieved, how long it took.

DeepEval keeps these separate (`Golden`, then `LLMTestCase`). Ragas uses one
record type (`SingleTurnSample`) and leaves the app's fields empty until the app
has run.

## Ragas: `SingleTurnSample`

| Field | Filled by | Holds |
|---|---|---|
| `user_input` | you | The question |
| `reference` | you | The reference (correct) answer |
| `reference_contexts` | you | Text of the passages that contain the answer |
| `reference_context_ids` | you | IDs of those passages |
| `rubric` | you | Scoring criteria, as a dict |
| `response` | the app | The app's answer |
| `multi_responses` | the app | Several answers to the same question |
| `retrieved_contexts` | the app | Text of the passages the app retrieved |
| `retrieved_context_ids` | the app | IDs of those passages |
| `persona_name`, `query_style`, `query_length` | generator | Describe generated (synthetic) questions |

Many samples go into an `EvaluationDataset`. All samples in one dataset must be
the same type. Multi-turn conversations use `MultiTurnSample` instead.

## DeepEval: `Golden` and `LLMTestCase`

**`Golden`** (what you write):

| Field | Holds |
|---|---|
| `input` (required) | The question |
| `expected_output` | The reference answer |
| `context` | Ideal supporting passages (text) |
| `expected_tools` | Tools an agent should call |
| `expected_labels` | Labels a classifier should give |
| `expectations` | Conditions that `must` hold and behaviors that `must_not` happen |
| `additional_metadata`, `comments` | Free-form extras |

**`LLMTestCase`** (golden + what happened) adds `actual_output` (the app's
answer), `retrieval_context` (passages actually retrieved), `tools_called`,
`completion_time` and `token_cost`.

`context` is what *should* support the answer. `retrieval_context` is what the
retriever *actually* returned. Comparing the two measures retrieval quality.

Datasets save and load as JSON, JSONL (one golden per line) or CSV. A file
cannot mix single-turn and multi-turn goldens.

## Side by side

| Concept | Ragas | DeepEval |
|---|---|---|
| Question | `user_input` | `input` |
| Reference answer | `reference` | `expected_output` |
| Correct passages (text) | `reference_contexts` | `context` |
| Correct passages (IDs) | `reference_context_ids` | none |
| App's answer | `response` | `actual_output` |
| Passages the app retrieved | `retrieved_contexts` / `retrieved_context_ids` | `retrieval_context` |
| Expected behavior (answer, refuse, hand off) | none (a `rubric` can describe it) | none (`expectations` or `expected_labels` can approximate it) |
| Where the question came from | only for generated questions | `additional_metadata` / `tags` |
| Latency and cost | none | `completion_time`, `token_cost` |
| Multi-turn | `MultiTurnSample` | `ConversationalGolden` |

## What this means for Litmus

1. **One record per question, one question type per file.** Both tools do this.
   Copy it.
2. **Keep the two kinds of record in separate files.** The dataset you write by
   hand changes rarely and is worth versioning. Run results are generated and
   change every run.
3. **Store correct passages as IDs.** With IDs, recall@k and MRR are exact.
   Matching passage text is fuzzy. Text can be optional.
4. **Two fields neither tool has as a first-class field:** expected behavior
   (answer / partial / refuse / hand off) and the question's source (FAQ,
   staff-written, real traffic, synthetic). Both matter for support agents.
   They could be Litmus's additions.
5. **Use JSONL.** One record per line diffs cleanly in Git, and both tools can
   load it. DeepEval reads it directly; Ragas can build samples from dicts.
6. **Borrow field names** so importing from either tool is trivial, for example
   DeepEval's `input` / `expected_output` plus Ragas's `reference_context_ids`.

## Sources

- [Ragas: evaluation sample](https://docs.ragas.io/en/stable/concepts/components/eval_sample/)
- [Ragas: evaluation schema reference](https://docs.ragas.io/en/stable/references/evaluation_schema/)
- [Ragas: evaluation dataset](https://docs.ragas.io/en/stable/concepts/components/eval_dataset/)
- [DeepEval: test cases](https://deepeval.com/docs/evaluation-test-cases)
- [DeepEval: datasets](https://deepeval.com/docs/evaluation-datasets)

# Litmus

Evaluate LLM and RAG apps, and tell real changes from noise.

Litmus runs each test question several times, compares two runs question by
question, and reports every score with a range. A change counts as real only
when the evidence supports it.

**Status:** early development. Every command works: `validate`, `run`,
`judge`, `label` (checking the judge against hand labels), `report` (with 95%
ranges and flip rates) and `compare` (paired verdicts and CI gates).

## Docs

- [Design](docs/design.md): what Litmus does and how
- [Labeling guide](docs/labeling-guide.md): how to label a dataset consistently
- [Dataset formats](docs/research/dataset-formats.md): how Ragas and DeepEval
  store test cases, and what Litmus borrows

## Commands

```text
litmus validate <dataset> [--app app.yaml]
litmus run <dataset> --app app.yaml --name baseline [--judge judge.yaml]
litmus judge <run> --judge judge.yaml
litmus label <run> --metric correctness
litmus report <run>
litmus compare <baseline-run> <candidate-run> [--gates gates.yaml]
```

## Development

Requires Python 3.12 or later.

```sh
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

Tests use public data only. Never add a client's dataset to this repository,
even in a gitignored folder.

## License

MIT

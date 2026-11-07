# Churn Risk Lab

**Data science** · An interpretable binary-classification experiment with train-only scaling.

## Run locally

Python 3.11+; no third-party packages required. From this directory:

```bash
python main.py demo
python main.py run example.json
python -m unittest discover -s tests -v
```

`example.json` is the input contract and contains only synthetic or fictional data. `core.run(payload)` returns a JSON-serializable result. Invalid inputs produce an error and a nonzero CLI exit code.

## What it implements

- Deterministic stratified train/test split on synthetic customer observations.
- From-scratch logistic regression with L2 regularization and numerically stable sigmoid.
- Held-out precision, recall, F1, accuracy, ROC AUC and a confusion matrix.
- Standardized coefficient importance and an explicit majority-class baseline.

## Design and limitations

Synthetic educational dataset only; no real customer or business accuracy claims. The test set must not be used for repeated hyperparameter selection. This model is not suitable for decisions about real people without validation, consent, fairness analysis and monitoring. Features are numeric only. No deployment or persisted model registry is included.

## Suggested next step

Add your own data and domain-specific acceptance tests. Read and understand the implementation before presenting it in interviews. No live deployment is implied.

<!-- AH-INDEPTH-DOCS:START -->
## In-depth engineering guide

### Problem and scope

An interpretable binary-classification experiment with train-only scaling.

This is an independent runnable reference implementation, with synthetic sample input and explicit limits. It is not a deployed service or independently validated business result.

### Architecture and end-to-end flow

```text
Synthetic/provided rows → stratified split → train-only scaling → L2 logistic regression → held-out metrics and majority baseline
```

| File | Responsibility |
|---|---|
| core.py | Domain validation and project-specific algorithm/state transitions |
| main.py | Bounded JSON CLI input, demo dispatch and JSON output |
| example.json | Exact reproducible sample input |
| tests/test_core.py | Unit and edge-case coverage |

### Design decisions and trade-offs

1. The classifier optimizes logistic loss with L2 regularization using pure-Python gradient descent. It is an interpretable baseline, not a pretrained model.

2. Train-only feature means and standard deviations prevent scaling leakage. A stratified holdout keeps both classes represented; seed control makes a synthetic experiment reproducible.

3. Features are tenure, tickets and usage; the target is binary churn. Metrics include confusion matrix, precision, recall, F1, accuracy and ROC AUC. Synthetic-data scores are not evidence of real customer performance.

### Reproducible setup

Requires Python 3.11+ and no third-party Python packages. Run from this project folder:

```bash
python main.py demo
python main.py run example.json
python -m unittest discover -s tests -v
```

CLI inputs are JSON objects, bounded to 1 MB. Supported domain errors produce a JSON error on stderr and a nonzero exit; inspect the message rather than treating a failed run as a result. The demo does not require a model key, network access or production data.

### Exact example input

```json
{
  "samples": 400,
  "seed": 42
}
```

### Core implementation interfaces

| Callable | Boundary |
|---|---|
| `sigmoid(z)` | Implementation boundary; see source for validation and return structure. |
| `auc(ys, ps)` | Implementation boundary; see source for validation and return structure. |
| `run(p)` | Implementation boundary; see source for validation and return structure. |

The Python `run` result is JSON-serializable. Inspect the source and `main.py run example.json` output for the exact return keys; do not assume another project’s result schema.

### Test coverage and reproducibility

- `tests/test_core.py::test_deterministic`
- `tests/test_core.py::test_metrics`
- `tests/test_core.py::test_auc_ties`
- `tests/test_core.py::test_small`
- `tests/test_core.py::test_single_class`

5 named tests are present in this folder. Run the command above to reproduce the local test results. Provider calls are mocked; Docker build/run was not executed; no real cloud/model/production-cluster validation is claimed.

### Security and failure boundaries

Synthetic educational dataset only; no real customer or business accuracy claims. The test set must not be used for repeated hyperparameter selection. This model is not suitable for decisions about real people without validation, consent, fairness analysis and monitoring. Features are numeric only. No deployment or persisted model registry is included.

Never commit .env, real API keys, runtime databases, approval material or customer data. See SECURITY.md. A successful example demonstrates the included workflow, not safe internet deployment.

### Troubleshooting

- **Import error:** run from this project’s own folder; each project intentionally has its own core.py.
- **Invalid sample/schema:** start from example.json, then change one field at a time and read the returned validation message.
- **Port already in use:** stop the previous process or select another --port where server.py exists.
- **SQLite lock/state surprise:** close duplicate processes, confirm PROJECT_DB and back up before deliberate resets. Do not delete unrelated project data.
- **External dependency unavailable:** default demos run offline; live provider calls and container pulls require your own working access and may incur external costs.

### Known limitations

Synthetic educational dataset only; no real customer or business accuracy claims. The test set must not be used for repeated hyperparameter selection. This model is not suitable for decisions about real people without validation, consent, fairness analysis and monitoring. Features are numeric only. No deployment or persisted model registry is included.

### Extension roadmap — not implemented

Use an approved real dataset with a time-respecting split, calibrate probabilities, evaluate subgroup performance and drift, and compare against independently tuned baselines.
<!-- AH-INDEPTH-DOCS:END -->

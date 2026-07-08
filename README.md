# HALO

Official implementation of HALO, a framework for generating and evaluating hallucinated and non-hallucinated contextual descriptions for model robustness analysis.

This repository contains the public research code for running the HALO data preparation and evaluation pipeline. Dataset files, generated results, review materials, and local model checkpoints are intentionally not included.

## Installation

```bash
git clone <repo-url>
cd HALO
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

HALO uses OpenAI-compatible chat completion APIs. Set credentials through environment variables:

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

`OPENAI_BASE_URL` is optional when using the official OpenAI endpoint.

## Quick Start

Prepare a CSV with at least these columns:

- `Sign`: the statement or claim to evaluate.
- `Label`: binary ground-truth label, where `1` corresponds to "yes" and `0` corresponds to "no".
- `caption_H`: hallucinated context.
- `caption_NH`: non-hallucinated context.
- `Path`: optional image path for multimodal evaluation.

Then run:

```bash
python main.py --config examples/sample_config.yaml
```

The sample config points to `examples/data/sample_clean_data.csv`. It is intended as a format reference and smoke-test input.

## Pipeline

HALO has two main stages:

1. Data preparation: generate hallucinated (`caption_H`) and non-hallucinated (`caption_NH`) descriptions, then validate them with checker modules.
2. Evaluation: compare model behavior under statement-only, non-hallucinated-context, and hallucinated-context settings.

Set `run.prepare: true` in a YAML config to run the preparation stage from raw data. Set it to `false` when `clean_data.csv` already contains `caption_H` and `caption_NH`.

## Configuration

Configs are YAML files with four sections:

- `llm`: API endpoint, API key, and role prompt.
- `run`: preparation flag, number of runs, and concurrency.
- `generation`: model and decoding parameters for caption generation.
- `evaluation`: model and prompt for downstream evaluation.
- `paths`: dataset and output paths.

For public use, prefer environment variables for credentials. If `llm.api_key` is empty or set to `YOUR_API_KEY`, `main.py` reads `OPENAI_API_KEY`.

## Repository Structure

```text
core/                 Core LLM wrapper, HALO checkers, and evaluation logic.
data/data_prepare/    Caption generation and validation pipeline.
data/dataset_format/  Dataset conversion helpers.
analysis/             Lightweight post-processing utilities used by main.py.
config/               Sanitized experiment config templates.
examples/             Minimal example config and CSV format sample.
docs/                 Notes on data and release hygiene.
```

## Data

Datasets are not redistributed in this repository. Download each benchmark from its official source and convert it into the expected CSV format. See `docs/DATA.md` for required columns.

## Citation

```bibtex
@inproceedings{halo2026,
  title = {HALO},
  author = {HALO Authors},
  booktitle = {To appear},
  year = {2026}
}
```

Replace this placeholder with the final camera-ready citation.

## License

This code is released under the MIT License. See `LICENSE`.

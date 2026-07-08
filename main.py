import argparse
import os
import yaml

from datetime import datetime
from typing import Tuple, List, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.model import LLM_Context
from data.data_prepare.data_prepare_pipeline import run_pipeline
from core.evalute_pipeline import evaluate_gpt_predictions
from analysis.csv_analysis.csv_analyze_from_subdir import filter_hallucination_gain_cases

"""hive_pipeline.py

Features
--------
1. **Optional data preparation** controlled via `run.prepare` in YAML.
2. **Concurrent evaluation runs** – number of parallel workers set by `run.max_workers` (default 1 ⇢ sequential).
3. Adds result suffix `num1_num2_num3` to each run fo`lder name.

YAML example
------------
```yaml
run:
  prepare: true       # run data preparation step once
  runs: 10            # total evaluation runs
  max_workers: 4      # concurrency (<= runs)
```

Usage
-----
```bash
python main.py --config examples/sample_config.yaml
```
"""

DEFAULTS = {
    "runs": 10,
    "prepare": False,
    "max_workers": 1,
}


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_llm_config(cfg: dict) -> dict:
    llm_cfg = cfg.setdefault("llm", {})
    api_key = llm_cfg.get("api_key") or os.getenv("OPENAI_API_KEY")
    base_url = llm_cfg.get("base_url") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"

    if not api_key or api_key == "YOUR_API_KEY":
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY or llm.api_key in the YAML config.")

    llm_cfg["api_key"] = api_key
    llm_cfg["base_url"] = base_url
    llm_cfg.setdefault("role_prompt", "You are a helpful assistant.")
    return cfg


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def result_to_suffix(res: Union[str, Tuple, list]) -> str:
    if isinstance(res, str):
        return res.replace(" ", "_")
    if isinstance(res, (tuple, list)):
        return "_".join(map(str, res))
    return str(res)


def evaluate_once(run_id: int, cfg: dict, clean_data_path: str, main_dir: str) -> str:
    """Single evaluation task executed in a thread."""
    # Each thread creates its own LLM_Context to avoid state issues
    llm = LLM_Context(
        max_tokens=None,
        temperature=cfg["evaluation"]["temperature"],
        model=cfg["evaluation"]["model"],
        api_key=cfg["llm"]["api_key"],
        base_url=cfg["llm"]["base_url"],
        role_prompt=cfg["llm"]["role_prompt"],
    )
    # Classification tasks are currently supported by the public entrypoint.
    is_classification = cfg["evaluation"].get("is_classification", False)
    model_name = str(cfg["evaluation"]["model"])
    temp_folder = f"run_{run_id:02d}_working"
    run_dir = ensure_dir(os.path.join(main_dir,model_name, temp_folder))

    if is_classification:
        res = evaluate_gpt_predictions(
            input_csv=clean_data_path,
            llm_context=llm,
            base_prompt=cfg["evaluation"]["prompt"],
            output_dir=run_dir,
        )
    else:
        raise NotImplementedError("Only classification evaluation is enabled in this release.")

    # --- Rename folder ---
    suffix = result_to_suffix(res)
    final_name = f"run_{run_id:02d}_{suffix}"
    final_dir = os.path.join(main_dir,model_name, final_name)
    if os.path.exists(final_dir):
        final_dir += "_" + datetime.now().strftime("%Y%m%d%H%M%S")
    os.rename(run_dir, final_dir)

    # --- Post-processing: filter rows where the hallucinated path beats the faithful path ---
    result_csv = os.path.join(final_dir, cfg["paths"]["result"])
    if os.path.exists(result_csv):
        try:
            filter_hallucination_gain_cases(result_csv)
        except Exception as e:
            print(f"⚠️  H-F gain filter skipped for run {run_id:02d}: {e}")

    return final_name


def main():
    # ---- CLI ----
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config file")
    args = parser.parse_args()

    cfg = resolve_llm_config(load_config(args.config))
    run_cfg = cfg.get("run", {})
    runs = int(run_cfg.get("runs", DEFAULTS["runs"]))
    prepare_flag = bool(run_cfg.get("prepare", DEFAULTS["prepare"]))
    max_workers = int(run_cfg.get("max_workers", DEFAULTS["max_workers"]))
    max_workers = max(1, min(max_workers, runs))

    # ---- Paths ----
    main_dir = cfg["paths"]["main_dir"]
    raw_data_path = os.path.join(main_dir, cfg["paths"]["raw_data"])
    clean_data_path = os.path.join(main_dir, cfg["paths"]["clean_data"])

    ensure_dir(main_dir)
    if prepare_flag:
        print("🛠  Data preparation in progress …")
        run_pipeline(
            input_path=raw_data_path,
            output_path=clean_data_path,
            model=cfg["generation"]["model"],
            api_key=cfg["llm"]["api_key"],
            base_url=cfg["llm"]["base_url"],
            prompt_template=cfg["generation"]["prompt_template"],
            validation_threshold=cfg["generation"]["validation_threshold"],
            max_attempts=cfg["generation"]["max_attempts"],
            max_workers=cfg["generation"]["max_workers"],
            max_tokens=cfg["generation"]["max_tokens"],
            role_prompt=cfg["llm"]["role_prompt"],
            temperature=cfg["generation"]["temperature"],
        )
        print("✅ Data preparation complete →", clean_data_path)

    if not os.path.exists(clean_data_path):
        raise FileNotFoundError(
            f"clean_data.csv missing: {clean_data_path}. Enable run.prepare or check path.")

    print(f"🚀 Launching {runs} evaluation runs with {max_workers} worker(s)…")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        fut_map = {
            executor.submit(evaluate_once, i, cfg, clean_data_path, main_dir): i
            for i in range(1, runs + 1)
        }
        for fut in as_completed(fut_map):
            run_id = fut_map[fut]
            try:
                name = fut.result()
                print(f"✅ Run {run_id:02d}/{runs} finished → {name}")
                results.append(name)
            except Exception as e:
                print(f"❌ Run {run_id:02d} failed: {e}")

    print("🎉 All runs completed.")


if __name__ == "__main__":
    for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
        os.environ.pop(k, None)
    main()


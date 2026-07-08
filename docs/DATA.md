# Data Format

HALO expects CSV files. The public repository does not redistribute benchmark data or generated results.

## Raw Input

For data preparation, the raw CSV should contain:

- `Sign`: statement, question, claim, or task input.
- `Label`: binary label used by the evaluation step.
- `Path`: optional local image path for multimodal inputs.

## Prepared Input

For evaluation without running generation, provide:

- `Sign`
- `Label`
- `caption_H`
- `caption_NH`
- `Path`, optional

`caption_H` is the hallucinated description. `caption_NH` is the non-hallucinated description.

## Dataset Policy

Use the official dataset sources and follow their licenses. Keep downloaded datasets and generated outputs outside version control.

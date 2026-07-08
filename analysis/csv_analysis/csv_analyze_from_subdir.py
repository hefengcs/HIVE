import pandas as pd
import os

def filter_hallucination_gain_cases(input_csv, output_csv=None):
    """
    Filter samples where the hallucinated-caption path is correct and the
    faithful-caption path is incorrect, i.e. positive Delta(H-F) cases.

    Args:
        input_csv: Prediction CSV path.
        output_csv: Optional output path. Defaults to result_HF_gain.csv.

    Returns:
        Filtered DataFrame.
    """

    if output_csv is None:
        output_csv = os.path.join(os.path.dirname(input_csv), "result_HF_gain.csv")

    try:
        df = pd.read_csv(input_csv)
    except UnicodeDecodeError:
        df = pd.read_csv(input_csv, encoding='ISO-8859-1')

    faithful_pred_col = "Prediction_caption_F"
    if faithful_pred_col not in df.columns and "Prediction_caption_NH" in df.columns:
        faithful_pred_col = "Prediction_caption_NH"

    filtered_df = df[
        (df["Prediction_caption_H"] == df["Label"]) &
        (df[faithful_pred_col] != df["Label"])
    ]

    print(f"H-F gain cases: {len(filtered_df)} / {len(df)} samples")
    filtered_df.to_csv(output_csv, index=False)
    print(f"Saved → {output_csv}")

    return filtered_df


if __name__ == "__main__":
    print("Use `python main.py --config <config.yaml>` to run the HIVE pipeline.")

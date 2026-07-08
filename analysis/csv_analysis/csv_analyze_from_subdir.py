import pandas as pd
import os

def filter_caption_H_success_cases(input_csv, output_csv=None):
    """
    筛选 caption_H 预测正确而 caption_NH 预测错误的样本（H wins over NH）。

    参数:
        input_csv (str): 输入预测结果的 CSV 文件路径
        output_csv (str, optional): 如果未指定，自动在同目录生成 result_H_wins.csv

    返回:
        pd.DataFrame: 筛选后的 DataFrame
    """

    if output_csv is None:
        output_csv = os.path.join(os.path.dirname(input_csv), "result_H_wins.csv")

    try:
        df = pd.read_csv(input_csv)
    except UnicodeDecodeError:
        df = pd.read_csv(input_csv, encoding='ISO-8859-1')

    filtered_df = df[
        (df['Prediction_caption_H'] == df['Label']) &
        (df['Prediction_caption_NH'] != df['Label'])
    ]

    print(f"H wins over NH: {len(filtered_df)} / {len(df)} samples")
    filtered_df.to_csv(output_csv, index=False)
    print(f"Saved → {output_csv}")

    return filtered_df


if __name__ == "__main__":
    # 示例调用
    filter_caption_H_success_cases("datasets/dataset_debug/results/result.csv")

import re

import pandas as pd
import openai
import os
import logging
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.model import LLM_Context


def evaluate_gpt_predictions(input_csv, llm_context: LLM_Context, base_prompt, output_dir,
                              eval_modes=("sign_only", "caption_F", "caption_H")):
    """
    eval_modes: which evaluation paths to run.
        - "sign_only":  Statement only, no caption
        - "caption_F":  Statement + faithful caption
        - "caption_H":  Statement + hallucinated caption
    Legacy CSV files that use caption_NH are accepted as faithful-caption inputs.
    Returns a tuple of accuracies in the order given by eval_modes.
    """
    df = pd.read_csv(input_csv, encoding="latin1")
    true_labels = df["Label"].tolist()

    # 设置日志输出
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "result.log")
    logging.basicConfig(filename=log_path, level=logging.INFO, filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("开始评估 GPT 预测...")

    # def ask_gpt(prompt):
    #     try:
    #         answer = llm_context.generate_response(prompt).strip().lower()
    #         return answer, 1 if "yes" in answer else 0
    #     except Exception as e:
    #         logging.error(f"❌ GPT Error: {e}")
    #         return "GENERATION_ERROR", 0

    def ask_gpt(prompt,image_path=None):
        raw = llm_context.generate_response(prompt, image_path)
        answer = raw.strip().lower() if raw else ""
        return answer, 1 if "yes" in answer else 0

    def evaluate_one(i, sign, description,image_path=None):
        if description:
            prompt = f"Image description: {description}\n\nStatement: {sign}\n\n{base_prompt}"
        else:
            prompt = f"Statement: {sign}\n\n{base_prompt}"
        reasoning, pred = ask_gpt(prompt,image_path)
        return i, sign, pred, reasoning

    def resolve_description_col(description_col):
        if description_col == "caption_F" and description_col not in df.columns and "caption_NH" in df.columns:
            return "caption_NH"
        return description_col

    def evaluate_all(description_col):
        description_col = resolve_description_col(description_col)
        results = [None] * len(df)
        reasonings = [None] * len(df)
        signs = [None] * len(df)
        #if "Path" in df.columns:
        if "Path" in df.columns and df["Path"].notna().all() and (df["Path"] != "").all():
            with ThreadPoolExecutor(max_workers=64) as executor:
                futures = [
                    executor.submit(evaluate_one, i, row["Sign"], row.get(description_col, ""),row["Path"])
                    for i, row in df.iterrows()
                ]
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating {description_col or 'Sign only'}"):
                    i, sign, pred, reasoning = future.result()
                    results[i] = pred
                    reasonings[i] = reasoning
                    signs[i] = sign
            return results, reasonings
        else:
            with ThreadPoolExecutor(max_workers=64) as executor:
                futures = [
                    executor.submit(evaluate_one, i, row["Sign"], row.get(description_col, ""))
                    for i, row in df.iterrows()
                ]
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating {description_col or 'Sign only'}"):
                    i, sign, pred, reasoning = future.result()
                    results[i] = pred
                    reasonings[i] = reasoning
                    signs[i] = sign
            return results, reasonings


    MODE_SPEC = {
        "sign_only":  ("",           "Sign only",         "Prediction_Sign_only",   "Reasoning_Sign_only"),
        "caption_F":  ("caption_F",  "Sign + caption_F",  "Prediction_caption_F",   "Reasoning_caption_F"),
        "caption_NH": ("caption_F",  "Sign + caption_F",  "Prediction_caption_F",   "Reasoning_caption_F"),
        "caption_H":  ("caption_H",  "Sign + caption_H",  "Prediction_caption_H",   "Reasoning_caption_H"),
    }

    valid_modes = [m for m in eval_modes if m in MODE_SPEC]
    for m in eval_modes:
        if m not in MODE_SPEC:
            print(f"⚠️  unknown eval mode: {m}, skip")

    # Run all eval paths concurrently — they don't depend on each other.
    mode_results = {}
    with ThreadPoolExecutor(max_workers=len(valid_modes)) as executor:
        future_to_mode = {
            executor.submit(evaluate_all, MODE_SPEC[m][0]): m
            for m in valid_modes
        }
        for future in as_completed(future_to_mode):
            m = future_to_mode[future]
            mode_results[m] = future.result()
            logging.info(f"🚀 Done: {MODE_SPEC[m][1]}")

    result_df = df.copy()
    accuracies = []
    print("\n📊 实验 Accuracy 结果：")
    for mode in valid_modes:
        col, label, pred_col, reason_col = MODE_SPEC[mode]
        preds, reasons = mode_results[mode]
        acc = accuracy_score(true_labels, preds)
        accuracies.append(acc)
        print(f"{label:<22} → ACC = {acc:.4f}")
        logging.info(f"ACC - {label}: {acc:.4f}")
        result_df[pred_col] = preds
        result_df[reason_col] = reasons

    result_path = os.path.join(output_dir, "result.csv")
    result_df.to_csv(result_path, index=False)
    logging.info(f"✅ 所有结果保存至 {result_path}")
    print(f"\n✅ 所有结果已保存到目录：{output_dir}")
    return tuple(accuracies)

# def evaluate_math_predictions(input_csv, llm_context: LLM_Context, base_prompt, output_dir, max_workers=128):
#     df = pd.read_csv(input_csv)
#     os.makedirs(output_dir, exist_ok=True)
#
#     log_path = os.path.join(output_dir, "result.log")
#     logging.basicConfig(filename=log_path, level=logging.INFO, filemode='w',
#                         format='%(asctime)s - %(levelname)s - %(message)s')
#     logging.info("开始评估 math 类型 GPT 预测...")
#
#     def ask_gpt(prompt):
#         answer = llm_context.generate_response(prompt).strip()
#         match = re.search(r"final answer[:：]?\s*([-+]?\d*\.?\d+)", answer, re.IGNORECASE)
#         if match:
#             return answer, float(match.group(1))
#         fallback = re.search(r"[-+]?\d*\.?\d+", answer)
#         if fallback:
#             return answer, float(fallback.group())
#         return answer, None
#
#     def evaluate_one(i, sign, desc_text, label):
#         prompt = f"{sign} {desc_text}\n{base_prompt}"
#         reasoning, pred_value = ask_gpt(prompt)
#         is_correct = False
#         try:
#             gt = float(label)
#             if pred_value is not None and abs(pred_value - gt) < 1e-3:
#                 is_correct = True
#         except:
#             pass
#         return i, sign, pred_value, reasoning, is_correct
#
#     result_dict = {}
#
#     for desc in ["", "caption_NH", "caption_H"]:
#         preds = [None] * len(df)
#         reasonings = [None] * len(df)
#         correct = 0
#
#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             futures = [
#                 executor.submit(evaluate_one, i, row["Sign"], row.get(desc, ""), row["Label"])
#                 for i, row in df.iterrows()
#             ]
#             for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating {desc or 'Sign only'}"):
#                 i, sign, pred, reasoning, is_correct = future.result()
#                 preds[i] = pred
#                 reasonings[i] = reasoning
#                 if is_correct:
#                     correct += 1
#
#         key = "Sign_only" if desc == "" else desc
#         acc = correct / len(df)
#         logging.info(f"{key} Accuracy: {acc:.4f}")
#         print(f"✅ {key} Accuracy: {acc:.4f}")
#
#         result_dict[f"Prediction_{key}"] = preds
#         result_dict[f"Reasoning_{key}"] = reasonings
#
#     result_df = df.copy()
#     for col, values in result_dict.items():
#         result_df[col] = values
#
#     result_path = os.path.join(output_dir, "result.csv")
#     result_df.to_csv(result_path, index=False)
#     logging.info(f"✅ 所有结果保存至 {result_path}")
#     print(f"\n📄 所有结果已保存到：{output_dir}")
#
import numpy as np
from scipy.stats import ttest_rel
def evaluate_gpt_creative_scores(input_csv, llm_context, base_prompt, output_dir):
    df = pd.read_csv(input_csv)

    # 设置日志输出
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "result.log")
    logging.basicConfig(filename=log_path, level=logging.INFO, filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("开始评估 GPT 创造性得分...")

    def ask_gpt(prompt, image_path=None):
        raw = llm_context.generate_response(prompt, image_path)
        answer = raw.strip() if raw else ""
        print("📨 Prompt:\n", prompt)
        print("🧠 Answer:\n", answer)

        try:
            for line in answer.split("\n"):
                if "score" in line.lower():
                    score = int("".join(filter(str.isdigit, line)))
                    score = max(0, min(score, 100))  # 限制在 0~100
                    return answer, score
            raise ValueError("未找到评分")
        except Exception as e:
            print("❌ 评分解析失败:", e)
            return answer, -1

    def evaluate_one(i, sign, description, image_path=None):
        prompt = f"{sign} {description}\n{base_prompt}"
        reasoning, score = ask_gpt(prompt, image_path)
        return i, sign, score, reasoning

    def evaluate_all(description_col):
        results = [None] * len(df)
        reasonings = [None] * len(df)
        signs = [None] * len(df)

        if "Path" in df.columns and df["Path"].notna().all() and (df["Path"] != "").all():
            with ThreadPoolExecutor(max_workers=64) as executor:
                futures = [
                    executor.submit(evaluate_one, i, row["Sign"], row.get(description_col, ""), row["Path"])
                    for i, row in df.iterrows()
                ]
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating {description_col or 'Sign only'}"):
                    i, sign, score, reasoning = future.result()
                    results[i] = score
                    reasonings[i] = reasoning
                    signs[i] = sign
        else:
            with ThreadPoolExecutor(max_workers=64) as executor:
                futures = [
                    executor.submit(evaluate_one, i, row["Sign"], row.get(description_col, ""))
                    for i, row in df.iterrows()
                ]
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating {description_col or 'Sign only'}"):
                    i, sign, score, reasoning = future.result()
                    results[i] = score
                    reasonings[i] = reasoning
                    signs[i] = sign
        return results, reasonings

    logging.info("🚀 Step 1: Sign only")
    scores_1, reason_1 = evaluate_all("")
    logging.info("🚀 Step 2: Sign + caption_F")
    scores_2, reason_2 = evaluate_all("caption_F")
    logging.info("🚀 Step 3: Sign + caption_H")
    scores_3, reason_3 = evaluate_all("caption_H")

    # 统计均值和标准差
    scores_1 = np.array(scores_1)
    scores_2 = np.array(scores_2)
    scores_3 = np.array(scores_3)

    print("\n📊 平均得分（0–100）:")
    print(f"Sign only         → Mean = {scores_1.mean():.2f}, Std = {scores_1.std():.2f}")
    print(f"Sign + caption_F  → Mean = {scores_2.mean():.2f}, Std = {scores_2.std():.2f}")
    print(f"Sign + caption_H  → Mean = {scores_3.mean():.2f}, Std = {scores_3.std():.2f}")

    print("\n📈 t检验（配对样本）:")
    print("Sign vs F :", ttest_rel(scores_1, scores_2))
    print("Sign vs H :", ttest_rel(scores_1, scores_3))
    print("F vs H    :", ttest_rel(scores_2, scores_3))

    # 整合所有结果到一个 CSV 中
    result_df = df.copy()
    result_df["Score_Sign_only"] = scores_1
    result_df["Reasoning_Sign_only"] = reason_1
    result_df["Score_caption_F"] = scores_2
    result_df["Reasoning_caption_F"] = reason_2
    result_df["Score_caption_H"] = scores_3
    result_df["Reasoning_caption_H"] = reason_3

    result_path = os.path.join(output_dir, "result.csv")
    result_df.to_csv(result_path, index=False)
    logging.info(f"✅ 所有结果保存至 {result_path}")
    print(f"\n✅ 所有结果已保存到目录：{output_dir}")

    return result_df


if __name__ == "__main__":
    print("Use `python main.py --config <config.yaml>` to run the HIVE pipeline.")

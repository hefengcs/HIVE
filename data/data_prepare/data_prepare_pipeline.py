import os
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.dection import generate_until_validated, SelfEvaluationChecker, FineGrainedFactChecker, ConsistencyChecker, \
    CrossModelAgreementChecker, RandomChecker
from core.model import LLM_Context
from core.new_checker import ParaphraseConsistencyJudge


# 初始化 GPT 上下文
def init_gpt_ctx(model, api_key, base_url,max_tokens,role_prompt,temperature):
    return LLM_Context(model=model, api_key=api_key, base_url=base_url,max_tokens=max_tokens,role_prompt=role_prompt,temperature=temperature,)

# 初始化检查器
def init_checkers(gpt_ctx):
    return [
        SelfEvaluationChecker(gpt_ctx),
        FineGrainedFactChecker(gpt_ctx),
        ConsistencyChecker(gpt_ctx),
        #ParaphraseConsistencyJudge(gpt_context=gpt_ctx),
        #CrossModelAgreementChecker(gpt_ctx), #新checker替换方案
        # RandomChecker(gpt_ctx),
        # RandomChecker(gpt_ctx),
    ]

# 生成 caption（幻觉 / 非幻觉）
# def generate_captions(
#     df, checkers, gpt_ctx,
#     prompt_template,
#     require_hallucination,
#     column_name,
#     validation_threshold=0,
#     max_attempts=5,
#     max_workers=64,
# ):
#     #results = [None] * len(df)
#     results = [None] * len(df)
#     reasons = [None] * len(df)
#     image_flag = pd.notna(df["Path"][0])
#
#     def process_row(i, sign,image_path=None):
#         prompt_row = prompt_template.format(Sign=sign)
#         print("prompt_row:")
#         print(prompt_row)
#         print("sign:")
#         print(sign)
#         result = generate_until_validated(
#             prompt=prompt_row,
#             gpt_context=gpt_ctx,
#             checkers=checkers,
#             require_hallucination=require_hallucination,
#             validation_threshold=validation_threshold,
#             max_attempts=max_attempts,
#             image_path=image_path,
#         )
#         #return i, result["text"]
#         return i, result["text"], result["validation"]
#
#
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         if image_flag:
#             futures = [executor.submit(process_row, i, row["Sign"], row["Path"]) for i, row in df.iterrows()]
#             for future in tqdm(as_completed(futures), total=len(futures), desc=f"Generating {column_name}"):
#                 try:
#                     # i, caption = future.result()
#                     # results[i] = caption
#                     i, caption, validation = future.result()
#                     results[i] = caption
#                     # 收集所有 checker 的 explanation
#                     if validation is not None:
#                         reasons[i] = [d["explanation"] for d in validation["details"]]
#                 except Exception as e:
#                     print(f"❌ Error in row: {e}")
#         else:
#             futures = [executor.submit(process_row, i, row["Sign"]) for i, row in df.iterrows()]
#             for future in tqdm(as_completed(futures), total=len(futures), desc=f"Generating {column_name}"):
#                 try:
#                     # i, caption = future.result()
#                     # results[i] = caption
#                     i, caption, validation = future.result()
#                     results[i] = caption
#                 except Exception as e:
#                     print(f"❌ Error in row: {e}")
#
#         if validation is not None:
#             reasons[i] = [d["explanation"] for d in validation["details"]]
#         df[column_name] = results
#         df[column_name + "_reason"] = reasons
#         # df[column_name] = results
#     return df

def generate_captions(
    df, checkers, gpt_ctx,
    prompt_template,
    require_hallucination,
    column_name,
    validation_threshold=0,
    max_attempts=5,
    max_workers=64,
):
    results = [None] * len(df)
    reasons = [None] * len(df)
    image_flag = pd.notna(df["Path"][0])

    def process_row(i, sign, image_path=None):
        prompt_row = prompt_template.format(Sign=sign)
        result = generate_until_validated(
            prompt=prompt_row,
            gpt_context=gpt_ctx,
            checkers=checkers,
            require_hallucination=require_hallucination,
            validation_threshold=validation_threshold,
            max_attempts=max_attempts,
            image_path=image_path,
        )
        return i, result["text"], result["validation"]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if image_flag:
            futures = [executor.submit(process_row, i, row["Sign"], row["Path"]) for i, row in df.iterrows()]
        else:
            futures = [executor.submit(process_row, i, row["Sign"]) for i, row in df.iterrows()]

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Generating {column_name}"):
            try:
                i, caption, validation = future.result()
                results[i] = caption
                if validation is not None:
                    #reasons[i] = [d["explanation"] for d in validation["details"]]
                    reasons[i] = [f"{d['checker']}: {d['explanation']}" for d in validation["details"]]

            except Exception as e:
                print(f"❌ Error in row: {e}")

    df[column_name] = results
    df[column_name + "_cheker_reasons"] = reasons
    return df



# 清洗无效样本
def clean_dataframe(df):
    #required_columns = ['Label', 'caption_H', 'caption_NH']
    required_columns = ['caption_H', 'caption_NH']
    df_cleaned = df.dropna(subset=required_columns)
    df_cleaned = df_cleaned[~df_cleaned[required_columns].astype(str).apply(lambda x: x.str.strip()).eq('').any(axis=1)]
    return df_cleaned

# 主流程封装
def run_pipeline(
    input_path,
    output_path,
    model,
    api_key,
    base_url,
    prompt_template,
    max_tokens,
    role_prompt,
    temperature,
    validation_threshold=0,
    max_attempts=5,
    max_workers=400,
    image_flag=False,
):
    print("📥 读取数据中...")
    df = pd.read_csv(input_path)

    #只使用前10个
    #df = df[:10]

    print("🔧 初始化 GPT 上下文与检查器")
    gpt_ctx = init_gpt_ctx(model, api_key, base_url,max_tokens,role_prompt,temperature)
    checkers = init_checkers(gpt_ctx)

    print("🚀 Step 1: 生成 caption_H（幻觉描述）")
    df = generate_captions(df, checkers, gpt_ctx,
                           prompt_template=prompt_template,
                           require_hallucination=True,
                           column_name="caption_H",
                           validation_threshold=validation_threshold,
                           max_attempts=max_attempts,
                           max_workers=max_workers)

    print("🚀 Step 2: 生成 caption_NH（非幻觉描述）")
    df = generate_captions(df, checkers, gpt_ctx,
                           prompt_template=prompt_template,
                           require_hallucination=False,
                           #require_hallucination=True,
                           column_name="caption_NH",
                           validation_threshold=validation_threshold,
                           max_attempts=max_attempts,
                           max_workers=max_workers)

    print("📝 Step 2.5: 保存未清理的数据")
    raw_output_path = output_path.replace(".csv", "_raw.csv")
    df.to_csv(raw_output_path, index=False)
    print(f"📄 未清理数据已保存至: {raw_output_path}")

    print("🧹 Step 3: 清洗无效样本")
    df_cleaned = clean_dataframe(df)

    print(f"✅ 清洗完成，有效样本数: {len(df_cleaned)}")
    df_cleaned.to_csv(output_path, index=False)
    print(f"📄 已保存至: {output_path}")


if __name__ == "__main__":
    print("Use `python main.py --config <config.yaml>` to run the HALO pipeline.")

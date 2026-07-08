# import json
# import random
#
# input_path = "datasets/external/images/binary_qa_minimal.json"
# output_path = "datasets/external/images/binary_qa_sample_500.json"
#
# # 读取原始 JSON 数据
# with open(input_path, "r", encoding="utf-8") as f:
#     data = json.load(f)
#
# # 检查是否足够条目可抽样
# if len(data) < 500:
#     raise ValueError(f"数据条目不足 500，仅有 {len(data)} 条。")
#
# # 随机抽取 500 条
# sampled_data = random.sample(data, 500)
#
# # 保存为新的 JSON 文件
# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump(sampled_data, f, indent=2, ensure_ascii=False)
#
# print(f"✅ 成功从 {len(data)} 条中抽取 500 条，保存至：{output_path}")


import json
import csv

input_json = "datasets/external/images/binary_qa_sample_500.json"
output_csv = "datasets/external/images/binary_qa_sample_500.csv"

with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Sign", "Label", "Path"])  # 表头

    for item in data:
        question = item.get("question", "")
        answer = item.get("answer", "").strip().lower()
        image_id = item.get("imageId", "")

        # 转换 answer 为 0/1
        label = 1 if answer == "yes" else 0

        # 构建路径（去掉 imageId 中的 n）

        path = f"datasets/external/images/images/{image_id}.jpg"

        writer.writerow([question, label, path])

print(f"✅ 成功生成 CSV 文件：{output_csv}")

import json

# 输入输出路径
input_path = "datasets/external/images/testdev_balanced_questions.json"
output_path = "datasets/external/images/binary_qa_minimal.json"

# 读取原始数据
with open(input_path, 'r') as f:
    data = json.load(f)

# 适配字典或列表格式
if isinstance(data, dict):
    items = data.values()
elif isinstance(data, list):
    items = data
else:
    raise TypeError("Unsupported JSON format")

# 筛选并抽取指定字段
binary_subset = [
    {
        "question": item["question"],
        "imageId": item["imageId"],
        "answer": item["answer"]
    }
    for item in items
    if item.get("answer") in ["yes", "no"]
]

# 保存结果
with open(output_path, 'w') as f:
    json.dump(binary_subset, f, indent=2)

print(f"成功提取 {len(binary_subset)} 条二分类数据，保存至：{output_path}")

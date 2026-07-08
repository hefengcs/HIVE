import json
import pandas as pd

# 读取 JSON 文件
json_path = "datasets/math401/backup/math401.json"
with open(json_path, 'r') as f:
    data = json.load(f)

# 转换为 DataFrame，并重命名列
df = pd.DataFrame(data)
df = df.rename(columns={"query": "Sign", "response": "Label"})

# 保存为 CSV 文件（与原文件同目录，加后缀）
csv_path = json_path.replace(".json", ".csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

print(f"✅ 已保存为 CSV 文件：{csv_path}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert test.jsonl  ➜  test.csv  (Sign, Label)
- 将 func 中的制表符转义为 \\t
- 将换行符转义为 \\n
- 强制所有字段加双引号
兼容 Python 3.8
"""

import json
import csv
import sys
from typing import Iterator, Dict

# 路径请按需修改
SRC = "datasets/CodeXGLUE/backup/test.jsonl"
DST = "datasets/CodeXGLUE/backup/test.csv"

# 若 func 特别长，需要调大 csv 模块允许的字段长度
csv.field_size_limit(sys.maxsize)


def escape(text: str) -> str:
    """把真换行 / 制表符转成可见文本，防止 Excel 拆列"""
    return (
        text.replace("\r\n", "\n")      # 统一行尾
            .replace("\n",  r"\n")      # ↵ -> \n
            .replace("\t",  r"\t")      # tab -> \t
    )


def iter_jsonl(path: str) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    rows = 0
    with open(DST, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(
            fout,
            delimiter=",",
            quotechar='"',
            quoting=csv.QUOTE_ALL,      # ★ 统统加引号
            lineterminator="\n",
        )
        writer.writerow(["Sign", "Label"])          # 表头

        for obj in iter_jsonl(SRC):
            sign  = escape(obj.get("func", ""))
            label = obj.get("target", "")
            writer.writerow([sign, label])
            rows += 1

    print(f"✅ Done!  {rows} records written ➜ {DST}")


if __name__ == "__main__":
    main()

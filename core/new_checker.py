# paraphrase_consistency_judge.py
import re
import json
from typing import List, Dict, Any, Optional
from core.model import LLM_Context


class ParaphraseConsistencyJudge:
    """
    事实判断版（True/False）：
      - 生成 2 个改写，仅作为“辅助线索”
      - 最终用 GPT 进行“事实正确性”裁决（True=原句为真；False=原句为假/高度不可信/无法支撑）
      - question（若提供）作为语境限定；不访问外部工具
    """

    def __init__(self, gpt_context: LLM_Context):
        self.gpt = gpt_context

    # ---------- 生成 2 个改写（函数名/入参保持不变） ----------
    def _paraphrase_prompt(self, text: str, n: int = 2) -> str:
        return (
            "You are a precise rephraser. Rephrase the given sentence in exactly "
            f"{n} distinct ways while STRICTLY preserving its meaning.\n\n"
            "HARD RULES:\n"
            "- Do NOT add, remove, or alter any facts, entities, numbers, units, or polarity.\n"
            "- Keep scope, frequency, certainty, modality, and negation IDENTICAL to the original.\n"
            "- Do NOT strengthen/weaken claims (no comparative/superlative shifts).\n"
            "- Keep length within ±15% of the original.\n"
            "- Output numbered lines only, no explanations.\n\n"
            f"SENTENCE: \"{text}\"\n\n"
            f"Rewrites (1 to {n}):"
        )

    def _parse_rewrites(self, raw: str, n: int = 2) -> List[str]:
        lines = []
        for line in raw.splitlines():
            m = re.match(r'^\s*\d+[\)\.\-:]\s*(.+)$', line.strip())
            if m:
                lines.append(m.group(1).strip().strip('"'))
        if len(lines) < n:
            quotes = re.findall(r'"([^"]+)"', raw)
            if quotes:
                lines = quotes
            else:
                lines = [l.strip().strip('"') for l in raw.splitlines() if len(l.strip()) > 0]
        lines = [l for l in lines if len(l) >= 3]
        uniq, seen = [], set()
        for s in lines:
            k = re.sub(r'\s+', ' ', s.lower())
            if k not in seen:
                uniq.append(s)
                seen.add(k)
        return uniq[:n]

    def _generate_two_paraphrases(self, sentence: str) -> List[str]:
        # 尝试最多两轮，确保拿到 2 条
        rewrites: List[str] = []
        for _ in range(2):
            if len(rewrites) >= 2:
                break
            prompt = self._paraphrase_prompt(sentence, n=2 - len(rewrites))
            raw = self.gpt.generate_response(prompt)
            new_items = self._parse_rewrites(raw, n=2 - len(rewrites))
            # 追加去重
            for it in new_items:
                key = re.sub(r'\s+', ' ', it.lower())
                if all(re.sub(r'\s+', ' ', r.lower()) != key for r in rewrites) and it != sentence:
                    rewrites.append(it)
        return rewrites[:2]

    # ---------- 事实裁决（保持函数名/入参不变） ----------
    def _judge_prompt(self, original: str, rewrites: List[str], question: Optional[str]) -> str:
        """
        裁决目标：判断 ORIGINAL 是否事实为真（True）或为假/不可信（False）。
        两个改写仅用于帮助澄清原句含义；若改写与原句不一致，忽略改写的新增/扭曲部分。
        """
        payload = {
            "task": "factuality_true_false_with_paraphrase_assist",
            "instruction": (
                "Decide the factual truth of the ORIGINAL claim using general world knowledge "
                "(no external tools). The TWO paraphrases are auxiliary; use them to clarify the "
                "intended meaning of the ORIGINAL but do NOT treat their added/changed content as evidence.\n"
                "Rules:\n"
                "- Return True if the ORIGINAL is factually correct as stated.\n"
                "- Return False if the ORIGINAL is factually incorrect, implausible, time-sensitive/ambiguous "
                "without sufficient support, or contradicted by well-known facts.\n"
                "- If paraphrases conflict with the ORIGINAL, prioritize ORIGINAL semantics; ignore additions.\n"
                "- If a precise date/time matters and is unspecified/likely outdated, be conservative and return False."
            ),
            "context_question": question,
            "original": original,
            "paraphrase_1": rewrites[0] if len(rewrites) > 0 else "",
            "paraphrase_2": rewrites[1] if len(rewrites) > 1 else "",
            "checklist": [
                "Named entities and roles correct?",
                "Numbers/units/ranges plausible?",
                "Negation/polarity identical to intended meaning?",
                "Scope/frequency/timeframe appropriate?",
                "Commonsense/encyclopedic knowledge consistent?"
            ]
        }
        sys = (
            "You are a careful fact adjudicator. Use ONLY general knowledge; do not invent sources.\n"
            "Return STRICT JSON with keys: final_verdict (one of [True, False]), "
            "confidence (0-1), reason (<=30 words)."
        )
        return f"{sys}\n\nINPUT JSON:\n{json.dumps(payload, ensure_ascii=False)}\n\nOutput JSON only:"

    def _gpt_judge(self, original: str, rewrites: List[str], question: Optional[str]) -> Dict[str, Any]:
        prompt = self._judge_prompt(original, rewrites, question)
        raw = self.gpt.generate_response(prompt).strip()
        try:
            m = re.search(r'\{.*\}', raw, flags=re.S)
            text = m.group(0) if m else raw
            data = json.loads(text)
            fv = str(data.get("final_verdict", "")).strip()
            conf = float(data.get("confidence", 0.0))
            rsn = str(data.get("reason", "")).strip()
            fv_low = fv.lower()
            if fv_low in ("true", "1", "yes"):
                fv = "True"
            elif fv_low in ("false", "0", "no"):
                fv = "False"
            else:
                fv = "False"  # 无法解析则保守为 False
            conf = max(0.0, min(1.0, conf))
            return {"final_verdict": fv, "confidence": conf, "reason": rsn, "raw": data}
        except Exception as e:
            # 解析失败：保守为 False
            return {"final_verdict": "False", "confidence": 0.0, "reason": f"judge_failed: {e}", "raw": raw}

    # ---------- 对外接口（保持函数名/入参不变） ----------
    def evaluate(self, answer: str, question: str = None, image_path=None) -> Dict[str, Any]:
        rewrites = self._generate_two_paraphrases(answer)
        judge = self._gpt_judge(answer, rewrites, question)
        return {
            "original": answer,
            "rewrites": rewrites,
            "verdict": judge["final_verdict"],      # "True" / "False"（事实判定）
            "confidence": judge["confidence"],
            "explanation": judge["reason"],
            "raw_judge": judge.get("raw")
        }


if __name__ == "__main__":
    print("Use `python main.py --config <config.yaml>` to run the HIVE pipeline.")

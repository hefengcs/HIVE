import os

from tqdm import tqdm

from core.new_checker import ParaphraseConsistencyJudge

for _k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(_k, None)
import re
import nltk
import time
import openai
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import util
from core.model import LLM_Context
import json
import re
from nltk.tokenize import sent_tokenize

# ----------------------------------------------------------
# Fine-Grained Fact Checker 类
# ----------------------------------------------------------
class FineGrainedFactChecker:
    def __init__(self, gpt_context):
        self.gpt = gpt_context

    def _extract_claims(self, text: str):
        sentences = sent_tokenize(text)
        claims = [s.strip() for s in sentences if len(s.strip()) > 15]
        return claims

    def _parse_score(self, text: str):
        match = re.search(r'(\d{1,3})\s*[%分]?', text)
        if match:
            val = int(match.group(1))
            return max(0, min(val, 100)) / 100
        return 0.5  # fallback 默认置信度

    def _parse_verdict(self, text: str):
        if re.search(r'\b(True|正确|属实)\b', text, re.IGNORECASE):
            return "True"
        elif re.search(r'\b(False|错误|不属实)\b', text, re.IGNORECASE):
            return "False"
        return "Unclear"

    def _validate_claim(self, claim: str):
#         prompt = f"""请判断下面的陈述是否真实，并给出可信度（0~100）：
#
# 陈述："{claim}"
#
# 请按照以下格式回答：
# 判断：True 或 False
# 置信度：百分数（例如 85%）
# 解释：简要理由

        prompt = f"""
Please determine whether the following statement is true and give a confidence level (0-100):

Statement: "{claim}"

Please answer using the following format:
Verdict: True or False
Confidence: Percentage (e.g., 85%)
Explanation: Brief justification

"""

        output = self.gpt.generate_response(prompt)
        verdict = self._parse_verdict(output)
        score = self._parse_score(output)

        return {
            "claim": claim,
            "verdict": verdict,
            "confidence": score,
            "explanation": output.strip()
        }

    def check(self, text: str):
        claims = self._extract_claims(text)
        print(f"🔍 正在核查 {len(claims)} 条陈述...\n")
        results = []

        for i, claim in enumerate(claims):
            print(f"[{i+1}] {claim}")
            result = self._validate_claim(claim)
            print(f"   ✅ 判断：{result['verdict']} | 📊 置信度：{result['confidence']:.2f}\n")
            results.append(result)

        return results
    def check_with_overall_score(self, text: str):
        claims = self._extract_claims(text)
        results = []
        total_score = 0.0
        valid_count = 0
        penalized = 0

        for claim in claims:
            result = self._validate_claim(claim)
            results.append(result)

            if result["verdict"] == "True":
                total_score += result["confidence"]
                valid_count += 1
            elif result["verdict"] == "False":
                penalized += 1  # 可选惩罚项

        # 平均可信度（只考虑 True 的）
        # 平均可信度（只考虑 True 的）
        if valid_count > 0:
            base_score = total_score / valid_count
        else:
            base_score = 0.0

        # 惩罚幻觉（每个 False 减 0.1，最多减 0.5）
        penalty = min(penalized * 0.1, 0.5)
        overall_score = max(base_score - penalty, 0.0)

        return {
            "claims": results,
            "overall_score": round(overall_score, 3)
        }
#没加入图像的版本:
    def evaluate(self, answer, question=None, image_path=None):
        prompt = f"""Evaluate the factual correctness of the following statement:

Statement: "{answer}"

Is this factually correct? Just answer True or False and give a brief reason.
"""
        try:
            response = self.gpt.generate_response(prompt)
            verdict = "True" if "true" in response.lower() else "False"
            confidence = 0.9 if verdict == "True" else 0.1
            return {
                "confidence": confidence,
                "verdict": verdict,
                "explanation": response.strip()
            }
        except Exception as e:
            return {
                "confidence": 0.5,
                "verdict": "Error",
                "explanation": f"FactCheck failed: {str(e)}"
            }
# **加入图像的版本**:
    # def evaluate(self, answer, question=None, image_path=None):
    #     if question:
    #         prompt = f"""Evaluate the factual correctness of the following answer, based on the image if provided:
    #
    # Question: "{question}"
    # Answer: "{answer}"
    #
    # Is this factually correct?
    # Respond with:
    # Verdict: True or False
    # Confidence: 0–100%
    # Explanation: Brief reasoning
    # """
    #     else:
    #         prompt = f"""Evaluate the factual correctness of the following statement, based on the image if provided:
    #
    # Statement: "{answer}"
    #
    # Is this factually correct?
    # Respond with:
    # Verdict: True or False
    # Confidence: 0–100%
    # Explanation: Brief reasoning
    # """
    #
    #     try:
    #         response = self.gpt.generate_response(prompt, image_path=image_path)
    #         verdict = "True" if "true" in response.lower() else "False"
    #         confidence = 0.9 if verdict == "True" else 0.1
    #         return {
    #             "confidence": confidence,
    #             "verdict": verdict,
    #             "explanation": response.strip()
    #         }
    #     except Exception as e:
    #         return {
    #             "confidence": 0.5,
    #             "verdict": "Error",
    #             "explanation": f"FactCheck failed: {str(e)}"
    #         }


class SelfEvaluationChecker:
    def __init__(self, gpt_context):
        self.gpt = gpt_context

    def _build_prompt(self, answer: str, question: str = None):
        if question:
            return f"""Please evaluate the factual accuracy of the following response:

Question: {question}
Answer: {answer}

Is this answer factually correct?

Response format:
Verdict: True or False  
Confidence: 0 to 100 (as a percentage)  
Explanation: Brief reasoning"""
        else:
            return f"""Please evaluate the factual accuracy of the following response:

Answer: {answer}

Is this answer factually correct?

Response format:
Verdict: True or False  
Confidence: 0 to 100 (as a percentage)  
Explanation: Brief reasoning"""

    def _parse_output(self, output: str):
        # Extract verdict
        if re.search(r'\b(True)\b', output, re.IGNORECASE):
            verdict = "True"
        elif re.search(r'\b(False)\b', output, re.IGNORECASE):
            verdict = "False"
        else:
            verdict = "Unclear"

        # Extract confidence
        match = re.search(r'(\d{1,3})\s*[%]?', output)
        if match:
            score = int(match.group(1))
            confidence = max(0, min(score, 100)) / 100
        else:
            confidence = 0.5

        return verdict, confidence, output.strip()
    #没加入图像的版本：
    def evaluate(self, answer: str, question: str = None, image_path=None):
        prompt = self._build_prompt(answer, question)
        response = self.gpt.generate_response(prompt)
        verdict, confidence, full_output = self._parse_output(response)
        return {
            "verdict": verdict,
            "confidence": confidence,
            "explanation": full_output
        }

    # 加入了图像的版本：
    # def evaluate(self, answer: str, question: str = None, image_path: str = None):
    #     prompt = self._build_prompt(answer, question)
    #
    #     # 传 image_path 给 gpt_context
    #     response = self.gpt.generate_response(prompt, image_path=image_path)
    #
    #     verdict, confidence, full_output = self._parse_output(response)
    #     return {
    #         "verdict": verdict,
    #         "confidence": confidence,
    #         "explanation": full_output
    #     }


class ConsistencyChecker:
    def __init__(self, gpt_context, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.gpt = gpt_context
        # Force CPU — GPUs are usually fully occupied by vLLM serving instances
        self.encoder = SentenceTransformer(model_name, device='cpu')

    def _paraphrase_prompt(self, sentence):
        return f"""Please rewrite the following sentence in different ways while preserving its meaning.
Sentence: "{sentence}"
Generate 3 alternate versions:"""

    def _generate_rewrites(self, sentence, n=3):
        prompt = self._paraphrase_prompt(sentence)
        response = self.gpt.generate_response(prompt)
        sentences = re.findall(r'"(.*?)"', response)
        if len(sentences) < n:
            sentences = response.split("\n")
        rewrites = [s.strip() for s in sentences if len(s.strip()) > 10]
        return rewrites[:n]

    def _semantic_similarity(self, sents):
        embeddings = self.encoder.encode(sents, convert_to_tensor=True)
        sims = []
        for i in range(len(sents)):
            for j in range(i+1, len(sents)):
                sim = util.cos_sim(embeddings[i], embeddings[j]).item()
                sims.append(sim)
        avg_sim = sum(sims) / len(sims) if sims else 0
        return round(avg_sim, 4), sims

    def check(self, sentence, rewrite_count=3):
        try:
            rewrites = self._generate_rewrites(sentence, n=rewrite_count)
            candidates = [sentence] + rewrites
            avg_sim, sims = self._semantic_similarity(candidates)

            if avg_sim > 0.85:
                verdict = "Consistent"
            elif avg_sim > 0.6:
                verdict = "Unstable"
            else:
                verdict = "Contradictory"

            return {
                "original": sentence,
                "rewrites": candidates,
                "similarity_scores": sims,
                "avg_similarity": avg_sim,
                "verdict": verdict
            }

        except Exception as e:
            return {"error": str(e)}

#     def evaluate(self, answer, question=None):
#         prompt = f"""Please rewrite the following sentence in different ways while preserving its meaning.
# Sentence: "{answer}"
# Generate 3 alternate versions:"""
#         try:
#             response = self.gpt.generate_response(prompt)
#             rewrites = [s.strip(' "\n') for s in response.split("\n") if len(s.strip()) > 5]
#             candidates = [answer] + rewrites[:3]
#             embeddings = self.encoder.encode(candidates, convert_to_tensor=True)
#             sims = []
#             for i in range(len(candidates)):
#                 for j in range(i + 1, len(candidates)):
#                     sim = util.cos_sim(embeddings[i], embeddings[j]).item()
#                     sims.append(sim)
#             avg_sim = sum(sims) / len(sims) if sims else 0.5
#             verdict = "True" if avg_sim > 0.75 else "False"
#             return {
#                 "confidence": avg_sim,
#                 "verdict": verdict,
#                 "explanation": f"Avg similarity between paraphrases: {avg_sim:.2f}"
#             }
#         except Exception as e:
#             return {
#                 "confidence": 0.5,
#                 "verdict": "Error",
#                 "explanation": f"Consistency evaluation failed: {str(e)}"
#             }
    #没加入图像的版本：
    def evaluate(self, answer, question=None, image_path=None):
        if question:
            prompt = f"""Please rewrite the following *answer* in different ways while preserving its meaning
    in the context of the question.

    Question: "{question}"
    Answer: "{answer}"

    Generate 3 alternate versions of the answer:"""
        else:
            prompt = f"""Please rewrite the following sentence in different ways while preserving its meaning.
    Sentence: "{answer}"
    Generate 3 alternate versions:"""

        try:
            response = self.gpt.generate_response(prompt)
            rewrites = [s.strip(' "\n') for s in response.split("\n") if len(s.strip()) > 5]
            candidates = [answer] + rewrites[:3]

            embeddings = self.encoder.encode(candidates, convert_to_tensor=True)
            sims = []
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    sim = util.cos_sim(embeddings[i], embeddings[j]).item()
                    sims.append(sim)
            avg_sim = sum(sims) / len(sims) if sims else 0.5

            verdict = "True" if avg_sim > 0.75 else "False"
            return {
                "confidence": avg_sim,
                "verdict": verdict,
                "explanation": f"Avg similarity between paraphrases (Q-aware): {avg_sim:.2f}"
            }
        except Exception as e:
            return {
                "confidence": 0.5,
                "verdict": "Error",
                "explanation": f"Consistency evaluation failed: {str(e)}"
            }
    #加入了图像的版本：
    # def evaluate(self, answer, question=None, image_path=None):
    #     if question:
    #         prompt = f"""Please rewrite the following *answer* in different ways while preserving its meaning
    # in the context of the question.
    #
    # Question: "{question}"
    # Answer: "{answer}"
    #
    # Generate 3 alternate versions of the answer:"""
    #     else:
    #         prompt = f"""Please rewrite the following sentence in different ways while preserving its meaning.
    # Sentence: "{answer}"
    # Generate 3 alternate versions:"""
    #
    #     try:
    #         # ✅ 传入 image_path（如果是 None，则自动走纯文本）
    #         response = self.gpt.generate_response(prompt, image_path=image_path)
    #
    #         rewrites = [s.strip(' "\n') for s in response.split("\n") if len(s.strip()) > 5]
    #         candidates = [answer] + rewrites[:3]
    #
    #         embeddings = self.encoder.encode(candidates, convert_to_tensor=True)
    #         sims = []
    #         for i in range(len(candidates)):
    #             for j in range(i + 1, len(candidates)):
    #                 sim = util.cos_sim(embeddings[i], embeddings[j]).item()
    #                 sims.append(sim)
    #
    #         avg_sim = sum(sims) / len(sims) if sims else 0.5
    #
    #         verdict = "True" if avg_sim > 0.75 else "False"
    #         return {
    #             "confidence": avg_sim,
    #             "verdict": verdict,
    #             "explanation": f"Avg similarity between paraphrases (Q-aware): {avg_sim:.2f}"
    #         }
    #     except Exception as e:
    #         return {
    #             "confidence": 0.5,
    #             "verdict": "Error",
    #             "explanation": f"Consistency evaluation failed: {str(e)}"
    #         }


def validate_text_by_consensus_all(text, require_hallucination: bool, checkers, threshold=0):
    """
    根据多个模型评估结果，判断该段文本是否符合目标（幻觉 or 非幻觉）

    参数：
    - text: str，输入文本
    - require_hallucination: bool，是否要求文本为幻觉
    - checkers: List[Checker]，模型列表
    - threshold: float，平均置信度阈值（默认 0.75）

    返回：
    - dict {
        "passed": bool,               # 是否通过判断
        "verdict": "True"/"False",   # 共识判断结果
        "confidence": float,         # 平均置信度
        "details": [...],            # 每个模型判断信息
        "text": 原始输入文本
    }
    """
    results = []
    verdicts = []
    confidences = []

    for checker in checkers:
        result = checker.evaluate(answer=text)
        results.append(result)
        verdicts.append(str(result["verdict"]).lower())
        confidences.append(result["confidence"])

    if len(set(verdicts)) == 1:
        final_verdict = verdicts[0]
    else:
        final_verdict = "unclear"

    avg_conf = sum(confidences) / len(confidences)

    # 是否满足要求（幻觉或非幻觉）+ 置信度达标
    if require_hallucination:
        passed = (final_verdict == "false") and avg_conf >= threshold
    else:
        passed = (final_verdict == "true") and avg_conf >= threshold

    return {
        "passed": passed,
        "verdict": final_verdict,
        "confidence": round(avg_conf, 4),
        "details": results,
        "text": text
    }
def generate_until_validated(prompt, gpt_context, checkers, require_hallucination=True,
                              validation_threshold=0.75, max_attempts=10, verbose=False, image_path=None):
    """
    反复使用 GPT 生成文本，直到满足幻觉或真实要求为止。

    返回:
    - dict {
        "success": bool,
        "text": str | None,
        "validation": dict,       # 包含 verdict / confidence / details / explanations
        "attempts": int,
        "all_attempts": list      # 记录所有尝试过的结果和解释
    }
    """
    all_attempts = []

    for attempt in range(1, max_attempts + 1):
        # 1. 生成候选文本
        generated_text = gpt_context.generate_response(prompt, image_path=image_path)


        #加上原始的
        generated_text = "Input: {}\n".format(prompt) +"Caption: {}".format(generated_text)

        #generated_text是生成的文本；而prompt则是原始的输入。


        # 2. 用共识校验
        validation = validate_text_by_consensus(
            text=generated_text,
            require_hallucination=require_hallucination,
            checkers=checkers,
            threshold=validation_threshold,
            #image_path=image_path,  #加入图像的版本
            image_path = None  #不加入图像的版本
        )

        # 3. 保存这次尝试的完整信息
        attempt_record = {
            "attempt": attempt,
            "text": generated_text,
            "verdict": validation["verdict"],
            "confidence": validation["confidence"],
            "passed": validation["passed"],
            "explanations": [r["explanation"] for r in validation["details"]]  # ✅ 保留解释
        }
        all_attempts.append(attempt_record)

        if verbose:
            print(f"\n[{attempt}] 🧪 尝试生成文本:")
            print(generated_text)
            print(f"判断: {validation['verdict']} | "
                  f"置信度: {validation['confidence']:.2f} | "
                  f"是否通过: {validation['passed']}")
            for exp in attempt_record["explanations"]:
                print(f" - 原因: {exp}")

        # 4. 如果通过要求，提前返回
        if validation["passed"]:
            return {
                "success": True,
                "text": generated_text,
                "validation": validation,
                "attempts": attempt,
                "all_attempts": all_attempts
            }

    # 超过最大次数仍未通过
    return {
        "success": False,
        "text": None,
        "validation": None,
        "attempts": max_attempts,
        "all_attempts": all_attempts
    }


# def generate_until_validated(prompt, gpt_context, checkers, require_hallucination=True,
#                               validation_threshold=0.75, max_attempts=10, verbose=True,image_path=None):
#     """
#     反复使用 GPT 生成文本，直到满足幻觉或真实要求为止。
#
#     参数:
#     - prompt: str，用户原始指令或提示语
#     - gpt_context: GPT4Context 实例，用于生成文本
#     - checkers: List[Checker]，多个幻觉检测器实例
#     - require_hallucination: 是否要求生成幻觉文本（True）或真实文本（False）
#     - validation_threshold: 最低置信度要求
#     - max_attempts: 最多尝试次数
#     - verbose: 是否打印每次尝试过程
#
#     返回:
#     - dict {
#         "success": bool,
#         "text": str,
#         "validation": dict,
#         "attempts": int
#     }
#     """
#     for attempt in range(1, max_attempts + 1):
#         generated_text = gpt_context.generate_response(prompt,image_path)
#         validation = validate_text_by_consensus(
#             text=generated_text,
#             require_hallucination=require_hallucination,
#             checkers=checkers,
#             threshold=validation_threshold
#         )
#
#         if verbose:
#             print(f"\n[{attempt}] 🧪 尝试生成文本:")
#             print(generated_text)
#             print(f"判断: {validation['verdict']} | 置信度: {validation['confidence']:.2f} | 是否通过: {validation['passed']}")
#
#         if validation["passed"]:
#             return {
#                 "success": True,
#                 "text": generated_text,
#                 "validation": validation,
#                 "attempts": attempt
#             }
#
#     # 超出尝试次数仍失败
#     return {
#         "success": False,
#         "text": None,
#         "validation": None,
#         "attempts": max_attempts
#     }

#这个版本没有加入图像部分
#-------------------------------------------
# def validate_text_by_consensus(text, require_hallucination: bool, checkers, threshold=0.75):
#     """
#     根据多个模型评估结果，判断该段文本是否符合目标（幻觉 or 非幻觉）
#     改为：只要有两个模型 verdict 一致，且都满足置信度要求即可。
#     """
#     results = []
#     verdicts = []
#     confidences = []
#     explanations = []
#     for checker in checkers:
#         result = checker.evaluate(answer=text)
#         result["checker"] = checker.__class__.__name__  # ✅ 标记来源
#         results.append(result)
#         # results.append(result)
#         verdicts.append(str(result["verdict"]).strip().lower())
#         confidences.append(result["confidence"])
#         # explanations.append({
#         #     "checker": checker.__class__.__name__,  # 保存类名/方法名
#         #     "verdict": result["verdict"],
#         #     "confidence": result["confidence"],
#         #     "explanation": result["explanation"]
#         # })
#
#     # 统计 verdict 出现次数
#     from collections import Counter
#     verdict_counter = Counter(verdicts)
#
#     # 只保留 true/false 中，至少有两个一致的
#     for v in ["true", "false"]:
#         if verdict_counter[v] >= 2:
#             # 找出这些 verdict 对应的置信度
#             selected_confidences = [confidences[i] for i in range(len(verdicts)) if verdicts[i] == v]
#             if all(c >= threshold for c in selected_confidences[:2]):  # 至少前两个达标
#                 passed = (v == "false" and require_hallucination) or (v == "true" and not require_hallucination)
#                 return {
#                     "passed": passed,
#                     "verdict": v,
#                     "confidence": round(sum(selected_confidences[:2]) / 2, 4),
#                     "details": results,  # 原始完整结果
#                     "explanations": explanations,  # ✅ 带来源的解释
#                     "text": text
#                 }
#
#     # 如果没有两者一致或置信度不达标
#     return {
#         "passed": False,
#         "verdict": "unclear",
#         "confidence": round(sum(confidences) / len(confidences), 4),
#         "details": results,
#         "text": text
#     }



#这个版本**加入了图像部分**
def validate_text_by_consensus(text, require_hallucination: bool, checkers, threshold=0.75, image_path=None):
    """
    根据多个模型评估结果，判断该段文本是否符合目标（幻觉 or 非幻觉）
    改为：只要有两个模型 verdict 一致，且都满足置信度要求即可。
    """
    results = []
    verdicts = []
    confidences = []

    for checker in checkers:
        # ✅ 透传 image_path
        #result = checker.evaluate(answer=text, image_path=image_path)
        result = checker.evaluate(answer=text)
        result["checker"] = checker.__class__.__name__
        results.append(result)
        verdicts.append(str(result["verdict"]).strip().lower())
        confidences.append(result["confidence"])

    from collections import Counter
    verdict_counter = Counter(verdicts)

    for v in ["true", "false"]:
        if verdict_counter[v] >= 2:
        #if verdict_counter[v] >= 1:
            selected_confidences = [confidences[i] for i in range(len(verdicts)) if verdicts[i] == v]
            if all(c >= threshold for c in selected_confidences[:2]):
                passed = (v == "false" and require_hallucination) or (v == "true" and not require_hallucination)
                return {
                    "passed": passed,
                    "verdict": v,
                    "confidence": round(sum(selected_confidences[:2]) / 2, 4),
                    "details": results,
                    "text": text
                }

    return {
        "passed": False,
        "verdict": "unclear",
        "confidence": round(sum(confidences) / len(confidences), 4),
        "details": results,
        "text": text
    }




from sklearn.metrics import classification_report
import pandas as pd
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.metrics import classification_report
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def process_row(i, row, checkers, threshold):
    text = row['Caption']
    raw_label = str(row['label']).strip()

    if raw_label == "1":
        label = "true"
    elif raw_label == "0":
        label = "false"
    else:
        label = "unclear"

    result = validate_text_by_consensus(
        text=text,
        require_hallucination=(label == "false"),
        checkers=checkers,
        threshold=threshold
    )

    system_verdict = result["verdict"]
    confidence = result["confidence"]

    return {
        "index": i,
        "GT_Label": label,
        "Pred_Verdict": system_verdict,
        "Confidence": confidence,
        "Caption": text
    }

def evaluate_on_labeled_data_multithread(file_path, checkers, threshold=0.75, max_workers=16):
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    results = []

    print(f"🔍 Evaluating {len(df)} samples using {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_row, i, row, checkers, threshold) for i, row in df.iterrows()]

        for f in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            results.append(f.result())

    # 排序保证输出顺序一致
    results = sorted(results, key=lambda x: x["index"])
    pd.DataFrame(results).to_csv("results.csv", index=False)

    gold = [r["GT_Label"] for r in results]
    pred = [r["Pred_Verdict"] for r in results]

    print("\n📊 Overall Evaluation:")
    print("GT labels:", set(gold))
    print("Predicted labels:", set(pred))

    label_set = ["true", "false", "unclear"]
    print(classification_report(gold, pred, labels=label_set, digits=3, zero_division=0))

    wrong_cases = [r for r in results if r["GT_Label"] != r["Pred_Verdict"]]
    pd.DataFrame(wrong_cases).to_csv("hallucination_misclassified.csv", index=False)
    print(f"\n❗ 错误样本已保存到 hallucination_misclassified.csv")



import re

import re

import re

import re
from typing import Tuple, Optional

class CrossModelAgreementChecker:
    def __init__(self, gpt_context, pro_base_url: str = "https://api.openai.com/v1", debug: bool = True):
        """
        gpt_context: 外部传入的 GPT-4o LLM_Context 实例
        pro_base_url: Gemini-Pro 的 base_url
        debug: 是否打印调试信息（原始输出与解析结果）
        """
        self.gpt = gpt_context
        self.debug = debug

        # 在内部直接初始化 Gemini-Pro
        self.pro = LLM_Context(
            model="gemini-2.5-pro",
            base_url=pro_base_url,
            role_prompt="You are a fact-checking assistant.",
            temperature=0
        )

    # -----------------------
    # Prompt 构造
    # -----------------------
    def _build_prompt(self, answer: str, question: Optional[str] = None) -> str:
        if question:
            return (
                "You are asked to fact-check an answer produced by another model.\n"
                "Please evaluate its factual accuracy carefully.\n\n"
                f"Question: {question}\n"
                f"Answer: {answer}\n\n"
                "Is this answer factually correct?\n\n"
                "Response format:\n"
                "Verdict: True or False\n"
                "Confidence: 0 to 100 (as a percentage)\n"
                "Explanation: Brief reasoning"
            )
        else:
            return (
                "You are asked to fact-check an answer produced by another model.\n"
                "Please evaluate its factual accuracy carefully.\n\n"
                f"Answer: {answer}\n\n"
                "Is this answer factually correct?\n\n"
                "Response format:\n"
                "Verdict: True or False\n"
                "Confidence: 0 to 100 (as a percentage)\n"
                "Explanation: Brief reasoning"
            )

    # -----------------------
    # 解析工具
    # -----------------------
    def _parse_output(self, output: str) -> Tuple[str, float, str]:
        """
        更稳健的解析：
        1) 逐行查找以 'Verdict' 开头的行，只在该行内判定 True/False；
        2) 逐行查找以 'Confidence' 开头的行，提取 0-100 的数字（有无百分号都可）；
        3) 若没找到，则回退到全局搜索（尽量避免解释里出现 true/false 误伤）。
        """
        verdict = "Unclear"
        confidence = 0.5

        if not output:
            return verdict, confidence, ""

        lines = output.splitlines()

        # 1) 逐行找 verdict
        found_verdict_line = False
        for line in lines:
            line_stripped = line.strip()
            if re.match(r"^\s*verdict\s*[:=]", line_stripped, flags=re.IGNORECASE):
                found_verdict_line = True
                low = line_stripped.lower()
                if "true" in low:
                    verdict = "True"
                elif "false" in low:
                    verdict = "False"
                else:
                    verdict = "Unclear"
                break

        # 如果没有找到显式 Verdict 行，做一次温和的全局兜底（只看第一处命中，避免解释里干扰）
        if not found_verdict_line:
            m = re.search(r"verdict\s*[:=]\s*([^\n\r]+)", output, flags=re.IGNORECASE)
            if m:
                cand = m.group(1).strip().lower()
                if "true" in cand:
                    verdict = "True"
                elif "false" in cand:
                    verdict = "False"
                else:
                    verdict = "Unclear"

        # 2) 逐行找 confidence
        found_conf_line = False
        for line in lines:
            line_stripped = line.strip()
            if re.match(r"^\s*confidence\s*[:=]", line_stripped, flags=re.IGNORECASE):
                found_conf_line = True
                # 提取 0-100 的数字，允许有空格/百分号
                m = re.search(r"(\d{1,3})\s*%?", line_stripped)
                if m:
                    score = int(m.group(1))
                    score = max(0, min(score, 100))
                    confidence = score / 100.0
                break

        # 如果没有找到显式 Confidence 行，进行一次全局兜底
        if not found_conf_line:
            m2 = re.search(r"confidence\s*[:=]\s*(\d{1,3})\s*%?", output, flags=re.IGNORECASE)
            if m2:
                score = int(m2.group(1))
                score = max(0, min(score, 100))
                confidence = score / 100.0

        return verdict, confidence, output.strip()

    # -----------------------
    # 主评估逻辑
    # -----------------------
    def evaluate(self, answer: str, question: Optional[str] = None):
        prompt = self._build_prompt(answer, question)

        # GPT-4o
        gpt_response = self.gpt.generate_response(prompt)
        gpt_verdict, gpt_conf, gpt_output = self._parse_output(gpt_response)

        # Gemini-Pro
        pro_response = self.pro.generate_response(prompt)
        pro_verdict, pro_conf, pro_output = self._parse_output(pro_response)

        explanation ="gpt_response:\n"+gpt_response+"\n"+"pro_response:\n" + pro_response+"\n"


        if self.debug:
            print("\n=== GPT-4o Raw ===", gpt_response)
            print("[DEBUG] GPT Parsed:", gpt_verdict, gpt_conf)
            print("\n=== Gemini-Pro Raw ===", pro_response)
            print("[DEBUG] Pro Parsed:", pro_verdict, pro_conf)

        # 融合逻辑
        if gpt_verdict == pro_verdict and gpt_verdict in ("True", "False"):
            final_verdict = gpt_verdict
            final_conf = (gpt_conf + pro_conf) / 2
            chosen_model = "Both"
        elif gpt_conf > pro_conf:
            final_verdict, final_conf, chosen_model = gpt_verdict, gpt_conf, "GPT-4o"
        elif pro_conf > gpt_conf:
            final_verdict, final_conf, chosen_model = pro_verdict, pro_conf, "Gemini-Pro"
        else:  # 置信度相同
            final_verdict, final_conf, chosen_model = pro_verdict, pro_conf, "Gemini-Pro"

        if self.debug:
            print(f"[CrossModel] Final: {final_verdict} ({final_conf:.2f}) from {chosen_model}")
        print("[DEBUG FINAL RESULT]", {
            "verdict": final_verdict,
            "confidence": float(final_conf),
            "chosen_model": chosen_model,
            "gpt_verdict": gpt_verdict,
            "gpt_confidence": gpt_conf,
            "pro_verdict": pro_verdict,
            "pro_confidence": pro_conf,
            "explanation":explanation
        })
        return {
            "verdict": final_verdict,
            "confidence": float(final_conf),  # ✅ 强制转标准 float
            "chosen_model": chosen_model,
            "gpt_verdict": gpt_verdict,
            "gpt_confidence": float(gpt_conf),
            "pro_verdict": pro_verdict,
            "pro_confidence": float(pro_conf),
            "explanation":explanation
        }
import random
class RandomChecker:
    """
    A random baseline checker that mimics the interface and I/O format of SelfEvaluationChecker.
    Verdict is sampled at a target keep rate; confidence is sampled randomly.
    """

    def __init__(self, gpt_context):
        # 保持签名一致；不使用 gpt_context，但保留以兼容主程序
        self._rng = random.Random()
        self._keep_rate = 0.5          # 目标保留率（True 的概率），可在实验前调整以匹配对照
        self._conf_range = (40, 60)    # 百分制置信度的随机范围，避免极端值

    # 可选：在运行前设置随机种子与保留率（不影响主程序，因为不会改变 evaluate 的签名）
    def set_seed(self, seed: int):
        self._rng.seed(seed)

    def set_keep_rate(self, rate: float):
        self._keep_rate = max(0.0, min(float(rate), 1.0))

    def _build_prompt(self, answer: str, question: Optional[str] = None) -> str:
        # 仅为接口一致；随机基线不会真正用到 prompt
        if question:
            return f"[RANDOM BASELINE]\nQuestion: {question}\nAnswer: {answer}\n"
        return f"[RANDOM BASELINE]\nAnswer: {answer}\n"

    def _parse_output(self, output: str):
        # 与原类保持相同解析逻辑，便于统一处理
        if re.search(r'\b(True)\b', output, re.IGNORECASE):
            verdict = "True"
        elif re.search(r'\b(False)\b', output, re.IGNORECASE):
            verdict = "False"
        else:
            verdict = "Unclear"

        match = re.search(r'(\d{1,3})\s*[%]?', output)
        if match:
            score = int(match.group(1))
            confidence = max(0, min(score, 100)) / 100
        else:
            confidence = 0.5

        return verdict, confidence, output.strip()

    def evaluate(self, answer: str, question: str = None, image_path=None):
        """
        与 SelfEvaluationChecker.evaluate 相同的签名与返回字典结构：
        { "verdict": "True"/"False"/"Unclear", "confidence": float in [0,1], "explanation": str }
        """
        # 1) 随机生成 verdict（按目标保留率）
        verdict_bool = self._rng.random() < self._keep_rate
        verdict = "True" if verdict_bool else "False"

        # 2) 随机生成置信度（百分制）
        conf_pct = self._rng.randint(self._conf_range[0], self._conf_range[1])

        # 3) 组装与原类兼容的输出文本，然后复用同样的解析器
        output = (
            f"Verdict: {verdict}\n"
            f"Confidence: {conf_pct}%\n"
            f"Explanation: Random baseline (no factual evaluation performed)."
        )

        _, confidence, full_output = self._parse_output(output)
        return {
            "verdict": verdict,
            "confidence": confidence,
            "explanation": full_output
        }



if __name__ == "__main__":
    print("Use `python main.py --config <config.yaml>` to run the HALO pipeline.")

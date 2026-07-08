





import os
import traceback
import math
import openai
from openai import OpenAI
import httpx
import base64
import mimetypes

class LLM_Context:
    def __init__(self, max_tokens=None, model="gpt-4o", api_key=None,
                 base_url="https://api.openai.com/v1",
                 role_prompt="You are a AI assistant.",
                 temperature=0):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY or pass api_key explicitly.")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.role_prompt = role_prompt

        openai.api_key = api_key
        openai.base_url = base_url
        openai.default_headers = {"x-foo": "true"}
        for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
            os.environ.pop(k, None)
        # The default httpx Limits cap us at 20 keepalive / 100 total connections,
        # which throttles ThreadPoolExecutor(max_workers=128) callers down to
        # only a few concurrent requests in flight. Bump both pool limits.
        self._http_client = httpx.Client(
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=256),
            timeout=httpx.Timeout(connect=15.0, read=240.0, write=60.0, pool=60.0),
        )
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=self._http_client,
        )


    # process-wide cache: same image is encoded once and reused across all
    # calls (caption gen + 3 checkers + retries on the same path) instead of
    # re-reading + re-base64ing every time. Locked because ThreadPoolExecutor
    # workers race on it.
    _b64_cache = {}
    import threading as _threading
    _b64_cache_lock = _threading.Lock()

    def image_to_base64(self, image_path):
        cached = LLM_Context._b64_cache.get(image_path)
        if cached is not None:
            return cached
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg"  # 默认 MIME 类型
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        url = f"data:{mime_type};base64,{encoded}"
        with LLM_Context._b64_cache_lock:
            LLM_Context._b64_cache[image_path] = url
        return url

    def get_text_embedding(self, text, model="text-embedding-ada-002"):
        try:
            client = openai.OpenAI(api_key=openai.api_key, base_url=openai.base_url)
            response = client.embeddings.create(
                input=[text],
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            print("❌ Embedding 获取失败:", e)
            return None

    def generate_response(self, prompt, image_path=None, retries: int = 5, backoff_base: float = 1.5):
        """Generate a response from the OpenAI chat model with automatic retries.

        Args:
            prompt (str): The message prompt or question.
            image_path (str, optional): Path to an image that will be embedded and sent alongside the prompt.
            retries (int, optional): How many total attempts to make before giving up. Defaults to 3.
            backoff_base (float, optional): Base factor for exponential back‑off (seconds). Defaults to 1.5.

        Returns:
            str: The content of the assistant's reply on success, or an error string on failure.
        """
        import time

        # Some chat templates (notably InternVL3 in vLLM 0.18) break when a
        # system role coexists with a list-form user content. To stay robust
        # across servers, fold the system prompt into the user message instead
        # of using a separate system role.
        merged_prompt = (self.role_prompt + "\n\n" + prompt) if self.role_prompt else prompt
        if image_path:
            try:
                base64_image = self.image_to_base64(image_path)
                messages = [
                    {"role": "user", "content": [
                        {"type": "text", "text": merged_prompt},
                        {"type": "image_url", "image_url": {"url": base64_image}},
                    ]},
                ]
            except Exception as e:
                print("❌ Image encoding failed:", e)
                return "IMAGE_ENCODING_ERROR"
        else:
            messages = [{"role": "user", "content": merged_prompt}]

        # --- Automatic retry loop --------------------------------------------------
        for attempt in range(retries):
            try:
                # response = openai.chat.completions.create(
                #     model=self.model,
                #     messages=messages,
                #     temperature=self.temperature,
                #     max_tokens=self.max_tokens,
                # )
                # return response.choices[0].message.content
                from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionSystemMessageParam



                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "seed": 60,
                    "timeout": 240,
                }
                if isinstance(self.max_tokens, int) and self.max_tokens > 0:
                    kwargs["max_tokens"] = self.max_tokens

                try:
                    response = self.client.chat.completions.create(**kwargs)
                except Exception as e:
                    # Some servers (e.g. vLLM) compute max_tokens = max_model_len -
                    # input_tokens internally and 400 if that goes negative.
                    # Fallback: shrink max_tokens and retry once before exiting.
                    msg = str(e)
                    if "max_tokens" in msg and "at least 1" in msg:
                        kwargs["max_tokens"] = 64
                        response = self.client.chat.completions.create(**kwargs)
                    else:
                        raise

                return response.choices[0].message.content


            except Exception as e:
                # Log the failure and decide whether to retry
                print(f"❌ GPT 调用失败 (attempt {attempt + 1}/{retries}): {e}")
                traceback.print_exc()  # ✅ 打印完整异常堆栈信息
                if attempt < retries - 1:
                    # Exponential back‑off: backoff_base^attempt seconds
                    sleep_time = backoff_base ** attempt
                    time.sleep(sleep_time)
                    continue  # Try again
                else:
                    # Out of retries — give up
                    print("生成失败！！！！")
                    return "GENERATION_ERROR"
        return None

    def generate_response_with_probs(self, prompt, image_path=None, retries: int = 5, backoff_base: float = 1.5):
        """
        返回模型输出 + Yes/No 概率
        """
        import time

        messages = [{"role": "system", "content": self.role_prompt}]
        if image_path:
            try:
                base64_image = self.image_to_base64(image_path)
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": base64_image}}
                    ]
                })
            except Exception as e:
                print("❌ Image encoding failed:", e)
                return "IMAGE_ENCODING_ERROR", {}
        else:
            messages.append({"role": "user", "content": prompt})

        def safe_exp(x):
            try:
                return math.exp(x)
            except:
                return None

        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    # top_p=1.0,
                    # n=1,
                    # seed=42,
                    # presence_penalty=0.0,
                    # frequency_penalty=0.0,
                    # logprobs=True,  # ✅ 开启 logprobs
                )

                content = response.choices[0].message.content
                logprobs = response.choices[0].logprobs.content[0].top_logprobs

                # 把 logprobs 转换成 {token: prob}
                probs = {tp.token: safe_exp(tp.logprob) for tp in logprobs}

                # 提取 Yes/No 概率（兼容大小写和空格）
                yes_prob = probs.get("Yes") or probs.get(" yes") or probs.get("YES")
                no_prob = probs.get("No") or probs.get(" no") or probs.get("NO")

                print("内容:", content)
                print("概率分布:", probs)
                print("Yes 概率:", yes_prob, "No 概率:", no_prob)

                return content, {"Yes": yes_prob, "No": no_prob, "all_probs": probs}

            except Exception as e:
                print(f"❌ GPT 调用失败 (attempt {attempt + 1}/{retries}): {e}")
                traceback.print_exc()
                if attempt < retries - 1:
                    sleep_time = backoff_base ** attempt
                    time.sleep(sleep_time)
                    continue
                else:
                    return "GENERATION_ERROR", {}
        return None, {}


class HallucinationClassifier(LLM_Context):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def classify(self, text):
        prompt = (
            "Classify the following text into one of the categories: "
            "'Creative Hallucination', 'Harmful Hallucination', or 'No Hallucination'.\n\n"
            f"Text:\n{text}\n\n"
            "Category:"
        )
        return self.generate_response(prompt)


class DualPathGenerator(LLM_Context):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_creative(self, query):
        """Creative path: encourage imaginative or creative hallucinations"""
        prompt = (
            "You are a speculative fiction writer or imaginative thinker.\n"
            "Your task is to respond creatively, allowing for metaphor, storytelling, or unexpected ideas.\n"
            "Avoid plain facts or textbook explanations. It’s okay if your response is not entirely real, "
            "as long as it is thought-provoking and coherent.\n\n"
            "Here is your creative prompt:\n"
            f"{query}\n\n"
            "Creative Response:"
        )
        return self.generate_response(prompt)

    def generate_factual(self, query):
        """Factual path: focus on factual correctness"""
        prompt = (
            "You are a cautious academic researcher.\n"
            "Please provide a factual, precise, and verifiable answer to the following question.\n"
            "Only use well-established knowledge. Do NOT speculate, exaggerate, or invent information.\n"
            "If the answer is unknown or uncertain, respond with 'The information is not available.'\n\n"
            "Question:\n"
            f"{query}\n\n"
            "Factual Answer:"
        )
        return self.generate_response(prompt)

    # def generate_creative(self, query):
    #     """Creative path: encourage imaginative or creative hallucinations"""
    #     prompt = (
    #         "You are encouraged to respond creatively to the following prompt. "
    #         "Your response can be imaginative and metaphorical, even if not strictly factual:\n\n"
    #         f"{query}"
    #     )
    #     return self.generate_response(prompt)
    #
    # def generate_factual(self, query):
    #     """Factual path: focus on factual correctness"""
    #     prompt = (
    #         "Please respond factually and accurately to the following question. "
    #         "Ensure the answer is based on verified information:\n\n"
    #         f"{query}"
    #     )
    #     return self.generate_response(prompt)
    ...

    def generate(self, prompt, mode="factual"):
        if mode == "factual":
            return self.generate_factual(prompt)
        elif mode == "creative":
            return self.generate_creative(prompt)
        else:
            raise ValueError("mode must be 'factual' or 'creative'")


class SafeDualPathGenerator:
    def __init__(self, generator: DualPathGenerator, classifier: HallucinationClassifier, max_retries=2):
        self.generator = generator
        self.classifier = classifier
        self.max_retries = max_retries

    def safe_generate(self, query, mode="creative"):
        """
        Generate text from dual-path generator.
        If the result is classified as harmful hallucination, regenerate (up to max_retries).
        """
        for attempt in range(self.max_retries + 1):
            if mode == "creative":
                result = self.generator.generate_creative(query)
            else:
                result = self.generator.generate_factual(query)

            hallucination_type = self.classifier.classify(result).strip()

            print(f"\n🔍 Attempt {attempt + 1} | Type: {hallucination_type}")
            print(f"📝 Output:\n{result}\n")

            if hallucination_type != "Harmful Hallucination":
                return result

        return "[Aborted: Repeated Harmful Hallucinations]"

class TaskAwareController(LLM_Context):
    def __init__(self, model="gpt-3.5-turbo"):
        super().__init__(model=model)

    def choose_best(self, query, creative_output, factual_output):
        prompt = (
            "You are a controller that decides which response is better for a given user query.\n"
            "You will be given a query and two candidate responses:\n"
            "Candidate A: from a creative/imaginative generation path.\n"
            "Candidate B: from a factually accurate generation path.\n\n"
            f"Query:\n{query}\n\n"
            f"Candidate A (Creative):\n{creative_output}\n\n"
            f"Candidate B (Factual):\n{factual_output}\n\n"
            "Which candidate is more appropriate for this query?\n"
            "Reply only with 'A' or 'B' and a short explanation."
        )
        return self.generate_response(prompt)

class HIVEPipeline:
    def __init__(self, model="gpt-3.5-turbo"):
        self.generator = DualPathGenerator(model=model)
        self.classifier = HallucinationClassifier(model=model)
        self.safe_generator = SafeDualPathGenerator(self.generator, self.classifier)
        self.controller = TaskAwareController(model=model)

    def run(self, query):
        print(f"\n🔍 Original Query:\n{query}")

        # Step 1: Safe creative output
        print("\n✨ Generating Creative Output...")
        creative_output = self.safe_generator.safe_generate(query, mode="creative")

        # Step 2: Safe factual output
        print("\n📘 Generating Factual Output...")
        factual_output = self.safe_generator.safe_generate(query, mode="factual")

        # Step 3: Let controller pick the best one
        print("\n🧭 Choosing Best Output via Controller...")
        decision = self.controller.choose_best(query, creative_output, factual_output)

        print("\n🎯 Controller's Decision:\n", decision)

        return {
            "query": query,
            "creative": creative_output,
            "factual": factual_output,
            "decision": decision
        }

class DualPathGeneratorWithFusion(DualPathGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_fusion(self, query, factual_output, creative_output):
        """Fusion path: Combine both factual and creative outputs"""
        prompt = (
            "You are given two responses, one based on factual accuracy and the other based on creativity. "
            "Your task is to combine these responses into a coherent, balanced answer, "
            "preserving factual correctness where needed, and enhancing creativity where possible.\n\n"
            f"Factual Response:\n{factual_output}\n\n"
            f"Creative Response:\n{creative_output}\n\n"
            "Your Combined Output:"
        )
        # prompt = (
        #     "You are an expert editor tasked with synthesizing two responses: one that emphasizes factual accuracy, "
        #     "and another that showcases creative expression. Your goal is to produce a single, coherent response that:\n"
        #     "- Preserves all essential factual information accurately\n"
        #     "- Incorporates imaginative or engaging elements from the creative version where appropriate\n"
        #     "- Resolves any conflicts between the two by prioritizing clarity and helpfulness to the user\n"
        #     "- Avoids unnecessary repetition and maintains a natural, readable tone\n\n"
        #     f"Factual Response:\n{factual_output}\n\n"
        #     f"Creative Response:\n{creative_output}\n\n"
        #     "Write your final, combined response below:"
        # )

        return self.generate_response(prompt)

class TaskAwareControllerWithFusion(TaskAwareController):
    def __init__(self, model="gpt-3.5-turbo"):
        super().__init__(model=model)

    def choose_best(self, query, creative_output, factual_output, fusion_output):
        prompt = (
            "You are a controller that decides which response is better for a given user query.\n"
            "You will be given a query and three candidate responses:\n"
            "Candidate A: from a creative/imaginative generation path.\n"
            "Candidate B: from a factually accurate generation path.\n"
            "Candidate C: from a combination of the above two responses.\n\n"
            f"Query:\n{query}\n\n"
            f"Candidate A (Creative):\n{creative_output}\n\n"
            f"Candidate B (Factual):\n{factual_output}\n\n"
            f"Candidate C (Fusion):\n{fusion_output}\n\n"
            "Which candidate is more appropriate for this query?\n"
            "Reply only with 'A', 'B', or 'C' and a short explanation."
        )
        return self.generate_response(prompt)

class HIVEPipelineWithFusion(HIVEPipeline):
    def __init__(self, model="gpt-3.5-turbo"):
        super().__init__(model=model)
        self.generator_with_fusion = DualPathGeneratorWithFusion(model=model)

    def run(self, query):
        print(f"\n🔍 Original Query:\n{query}")

        # Step 1: Safe creative output
        print("\n✨ Generating Creative Output...")
        creative_output = self.safe_generator.safe_generate(query, mode="creative")

        # Step 2: Safe factual output
        print("\n📘 Generating Factual Output...")
        factual_output = self.safe_generator.safe_generate(query, mode="factual")

        # Step 3: Fusion output (combine factual and creative)
        print("\n🔗 Generating Fusion Output...")
        fusion_output = self.generator_with_fusion.generate_fusion(query, factual_output, creative_output)

        # Step 4: Let controller pick the best one
        print("\n🧭 Choosing Best Output via Controller...")
        decision = self.controller.choose_best(query, creative_output, factual_output, fusion_output)

        print("\n🎯 Controller's Decision:\n", decision)

        return {
            "query": query,
            "creative": creative_output,
            "factual": factual_output,
            "fusion": fusion_output,
            "decision": decision
        }





if __name__ == "__main__":
    gpt4o_context = LLM_Context(model="gpt-4o")

    content = gpt4o_context.generate_response("Answer only with 'Yes' or 'No'.\nIs the sky blue?")

    print(content)

# ----------------chatbox.py----------------
from openai import OpenAI
from typing import List, Optional, Dict, Any
import time, math

class ParkinsonsGaitChatbot:

    PROMPT_TEMPLATE = """You are a helpful clinical decision support AI for Parkinson's disease diagnosis using gait analysis. Always:
    1. Think step-by-step before responding
    2. Justify your initial assessment and interpretation of gait events (Stride, Swing and Stance time and their cycles), referencing clinical guidelines or evidence when possible
    3. When finalization request is queried, you must finalize the decision (only answer: "Healthy", "Stage 2", "Stage 2.5", or "Stage 3") but you may overturn your prior assessment if, after reviewing all evidence, you are confident a different answer is correct. Clearly state the reason for any change.
    4. Provide information that is correct, intuitive, compact, and simple for end-users to understand
    5. Avoid assumptions. Only use the provided data and context
    6. Cross-validate findings with multiple source
    7. Reference sources for non-standard conclusions
    8. Maintain clarity with concise and straightforward responses"""

    def __init__(self, temperature: float = 0.2, max_tokens: int = 4096,
                 default_model: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct:fireworks-ai"):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.default_model = default_model


    def _make_client(self, model_name: str) -> OpenAI:
        """
        Use OpenAI official endpoint for 'gpt-*' models,
        and the HF Router for the Fireworks-hosted Llama model.
        """
        if model_name == "gpt-4o":
            # OpenAI hosted models
            return OpenAI(api_key="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")  # default base_url=https://api.openai.com/v1
        else:
            # HF Router for 3rd-party hosted models like Fireworks
            return OpenAI(base_url="https://router.huggingface.co/v1", api_key="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")


    @staticmethod
    def _estimate_tokens_from_text(s: str) -> int:
        """
        Very lightweight token estimate:
        - fallback: ~4 chars per token (common rule of thumb)
        - cross-check with whitespace words to avoid wild outliers
        """
        if not s:
            return 0
        by_chars = max(1, math.ceil(len(s) / 4))
        by_words = max(1, math.ceil(len(s.split()) * 1.3))  # ~1.3 tokens per word (rough)
        # Use a middle-ish value to smooth extremes
        return max(1, round((by_chars + by_words) / 2))


    def generate_response(self, history: List, new_message: str,
                          context: Optional[Dict[str, Any]] = None,
                          model_name: Optional[str] = None):
        """
        Added `model_name` param so the UI can pass the selected LLM.
        Falls back to self.default_model if not provided.
        """
        chosen_model = model_name or self.default_model
        client = self._make_client(chosen_model)



        chat_history = []
        if history:
            for msg in history:
                if isinstance(msg, (list, tuple)) and len(msg) == 2:
                    if msg[0]:
                        chat_history.append({"role": "user", "content": msg[0]})
                    if msg[1]:
                        chat_history.append({"role": "assistant", "content": msg[1]})

        messages = [{"role": "system", "content": self.PROMPT_TEMPLATE}]

        # a multimodal system message with context + images
        if context and (context.get("text") or context.get("images")):
            mm_content = []
            if context.get("text"):
                # Prefix so the model knows this is context (not a question)
                mm_content.append({"type": "text", "text": "CONTEXT FOR ANALYSIS:\n" + context["text"]})
            if model_name == "gpt-4o" or model_name == "meta-llama/Llama-4-Scout-17B-16E-Instruct:fireworks-ai":
                for uri in (context.get("images") or [])[:3]:
                    mm_content.append({"type": "image_url", "image_url": {"url": uri}})
                    mm_content.append({"type": "text", "text": "The first attached image is the raw data with GradCAM overlay, the second is with LRP overlay, and the third highlights regions (in red) where the two XAI methods disagree significantly, indicating areas of model uncertainty."})
            messages.append({"role": "user", "content": mm_content})

        messages.extend(chat_history)
        messages.append({"role": "user", "content": new_message})

        start_ts = time.time()

        try:
            response = client.chat.completions.create(
                model=chosen_model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            assistant_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    assistant_response += chunk.choices[0].delta.content
                    updated_history = history + [[new_message, assistant_response]]
                    yield updated_history

                    elapsed = time.time() - start_ts
                    approx_output_tokens = self._estimate_tokens_from_text(assistant_response)
                    stats = (
                        f"Elapsed time = {elapsed:.1f}s | "
                        f"Output token ≈ {approx_output_tokens} tok | "
                    )
                    print(f"[Chatbot] {stats}")

        except Exception as e:
            error_response = f"Error generating response: {str(e)}"
            updated_history = history + [[new_message, error_response]]
            yield updated_history
            elapsed = time.time() - start_ts
            stats = f"Elapsed: {elapsed:.1f}s | Output≈0 tok"
            print(f"[Chatbot] {stats} | Error: {str(e)}")


# ----------------chatbox.py----------------
from openai import OpenAI
from typing import List, Optional, Dict, Any
import time, math

try:
    import transformers as _hf_transformers
    import torch as _hf_torch
except Exception:
    _hf_transformers = None
    _hf_torch = None

    
OPENAI_API_KEY = "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"          # your OpenAI / GPT-style key
HF_API_TOKEN   = "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"          # your Hugging Face access token

class ParkinsonsGaitChatbot:

    PROMPT_TEMPLATE = """You are a helpful clinical decision support AI for Parkinson's disease diagnosis using gait analysis. Always:
    1. Think step-by-step before responding
    2. Justify your initial assessment and interpretation of gait events (Stride, Swing and Stance time and their cycles), referencing clinical guidelines or evidence when possible
    3. When finalization request is queried, you must finalize the decision (only answer: "Healthy", "Stage 2", "Stage 2.5", or "Stage 3") but you may overturn your prior assessment if, after reviewing all evidence, you are confident a different answer is correct. 
    4. Using your clinical analysis and justification, identify the potential reasons for any change in the final decision (e.g., specific gait abnormalities, asymmetries, variability, etc.) or in case of no change, justify why the initial assessment was correct. Then explain how these factors contribute to the final severity.
    5. Provide information that is correct, intuitive, compact, and simple for end-users to understand
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
        if model_name == "gpt-4o" or model_name == "gpt-5":
            # OpenAI hosted models (default: base_url=https://api.openai.com/v1)
            return OpenAI(api_key=OPENAI_API_KEY)  
        else:
            # HF Router for 3rd-party hosted models like Fireworks
            return OpenAI(base_url="https://router.huggingface.co/v1", api_key=HF_API_TOKEN)


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
        chosen_model = model_name or self.default_model
        use_hf_local = (chosen_model == "aaditya/OpenBioLLM-Llama3-70B")

        # Build chat history in OpenAI-style role format
        chat_history = []
        if history:
            for msg in history:
                if isinstance(msg, (list, tuple)) and len(msg) == 2:
                    if msg[0]:
                        chat_history.append({"role": "user", "content": msg[0]})
                    if msg[1]:
                        chat_history.append({"role": "assistant", "content": msg[1]})

        # Start with your system prompt
        messages = [{"role": "system", "content": self.PROMPT_TEMPLATE}]

        # (Optional) multimodal context → this model is text-only; we add textual context regardless
        if context and (context.get("text") or context.get("images")):
            mm_text = ""
            if context.get("text"):
                mm_text += "CONTEXT FOR ANALYSIS:\n" + context["text"]
            if mm_text:
                messages.append({"role": "user", "content": mm_text})

        messages.extend(chat_history)
        messages.append({"role": "user", "content": new_message})

        start_ts = time.time()

        # ------------------------- Branch: HF transformers (OpenBioLLM) -------------------------
        if use_hf_local:
            try:
                # Generate in one shot; then yield once (keeps your Gradio flow)
                completion = self._generate_with_hf(messages, self.temperature, self.max_tokens)
                assistant_response = completion
                updated_history = history + [[new_message, assistant_response]]
                yield updated_history

                elapsed = time.time() - start_ts
                approx_output_tokens = self._estimate_tokens_from_text(assistant_response)
                stats = (f"Elapsed time = {elapsed:.1f}s | Output token ≈ {approx_output_tokens} tok | ")
                print(f"[Chatbot][HF] {stats}")

            except Exception as e:
                error_response = f"Error (HF transformers) generating response: {str(e)}"
                updated_history = history + [[new_message, error_response]]
                yield updated_history
                elapsed = time.time() - start_ts
                stats = f"Elapsed: {elapsed:.1f}s | Output≈0 tok"
                print(f"[Chatbot][HF] {stats} | Error: {str(e)}")
            return

        # ------------------------- Default: OpenAI-compatible (your existing code) -------------------------
        client = self._make_client(chosen_model)
        try:
            """
            Syntax for streaming chat completions:
            
            response = client.chat.completions.create(
                model=chosen_model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            """

            def _build_kwargs(model_name: str):
                # Base kwargs common to all models
                kwargs = {
                    "model": model_name,
                    "messages": messages,
                    "stream": True,
                }
                # Param name for output tokens differs on some newer models
                if str(model_name).startswith("gpt-5"):
                    kwargs["max_completion_tokens"] = self.max_tokens
                else:
                    kwargs["max_tokens"] = self.max_tokens

                # temperature: do NOT send for gpt-5 (only default=1 allowed)
                if not str(model_name).startswith("gpt-5"):
                    kwargs["temperature"] = self.temperature
                # else: omit temperature entirely
                return kwargs

            create_kwargs = _build_kwargs(chosen_model)
            response = client.chat.completions.create(**create_kwargs)

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


    _hf_pipe = None

    def _get_hf_pipeline(self):
        """
        Lazily create and cache a HF transformers pipeline for OpenBioLLM.
        """
        if self._hf_pipe is not None:
            return self._hf_pipe
        if _hf_transformers is None or _hf_torch is None:
            raise RuntimeError("transformers/torch not available. Please `pip install transformers torch`.")

        model_id = "aaditya/OpenBioLLM-Llama3-70B"
        self._hf_pipe = _hf_transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": _hf_torch.bfloat16},
            device="auto",  # uses GPU if available
        )
        return self._hf_pipe

    def _generate_with_hf(self, messages, temperature: float, max_new_tokens: int):
        """
        messages: list of {"role": "...", "content": "..."} including system/user/assistant.
        Applies chat template and runs local generation. Returns str.
        """
        pipe = self._get_hf_pipeline()

        # OpenBioLLM is Llama3-based; use the tokenizer’s chat template
        prompt = pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # prefer deterministic output unless you explicitly want sampling
        do_sample = (temperature or 0.0) > 0.0

        # EOS handling for Llama 3 style
        terminators = [
            pipe.tokenizer.eos_token_id,
            pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        out = pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=do_sample,
            temperature=max(0.0, min(2.0, float(temperature))),
            top_p=0.9
        )
        gen = out[0]["generated_text"][len(prompt):]
        return gen.strip()
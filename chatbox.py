# ----------------chatbox.py----------------
import openai
from typing import List, Optional, Dict, Any

class ParkinsonsGaitChatbot:

    PROMPT_TEMPLATE = """You are a helpful clinical decision support AI for Parkinson's disease diagnosis using gait analysis. Always:
    1. Think step-by-step before responding
    2. Justify your initial assessment and interpretation of gait metrics, referencing clinical guidelines or evidence when possible
    3. When finalization request is queried, you must finalize the decision (answer: "Healthy", "Stage 2", "Stage 2.5", or "Stage 3") but you may overturn your prior assessment if, after reviewing all evidence, you are confident a different answer is correct. Clearly state the reason for any change.
    4. Provide accurate, current information using clinical gait analysis guidelines
    5. Avoid assumptions. Only use the provided gait data and metrics
    6. Cross-validate findings with multiple gait parameters (stride time, stance time, swing time, force patterns)
    7. Flag urgent concerns about mobility or fall risk immediately
    8. Reference sources for non-standard conclusions about Parkinson's gait patterns
    9. Maintain clarity with concise and straightforward responses
    10. Consider both XAI explanation discrepancies and traditional gait metrics in your analysis
    11. Be aware that high discrepancy regions between GradCAM and LRP may indicate areas of diagnostic uncertainty"""

    def __init__(self, api_key: str, temperature: float = 0.2, max_tokens: int = 4096):
        openai.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_response(self, history: List, new_message: str, context: Optional[Dict[str, Any]] = None):
        """(unchanged) text-only streaming response"""
        from openai import OpenAI
        client = OpenAI(api_key=openai.api_key)

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
            for uri in (context.get("images") or [])[:3]:
                mm_content.append({"type": "image_url", "image_url": {"url": uri}})
            messages.append({"role": "user", "content": mm_content})

        messages.extend(chat_history)
        messages.append({"role": "user", "content": new_message})

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # vision-capable + streaming
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
        except Exception as e:
            error_response = f"Error generating response: {str(e)}"
            updated_history = history + [[new_message, error_response]]
            yield updated_history

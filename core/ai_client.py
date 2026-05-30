import os

try:
    from langdetect import detect as _detect_lang
    def detect_language(text):
        try:
            code = _detect_lang(text[:500])
            names = {
                'it': 'Italian', 'en': 'English', 'de': 'German',
                'fr': 'French', 'es': 'Spanish', 'pt': 'Portuguese',
                'nl': 'Dutch', 'pl': 'Polish',
            }
            return names.get(code, 'English')
        except Exception:
            return 'English'
except ImportError:
    def detect_language(text):
        return 'English'


MAX_TOKENS = 8000
RATE_CODES = ["429", "quota", "rate", "limit", "resource", "exhausted", "overload", "unavailable"]


def _call_gemini(prompt, model_name):
    import google.generativeai as genai
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise ValueError("GEMINI_API_KEY not configured.")
    genai.configure(api_key=key)
    m = genai.GenerativeModel(
        model_name,
        generation_config=genai.GenerationConfig(temperature=0.3, max_output_tokens=MAX_TOKENS),
    )
    return m.generate_content(prompt).text


def _call_claude_haiku(prompt):
    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY not configured.")
    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def call_ai(prompt, language="English"):
    full_prompt = f"Always respond in {language}.\n\n" + prompt
    for model_name, label in [
        ("gemini-2.5-flash", "Gemini 2.5 Flash"),
        ("gemini-2.5-flash-lite", "Gemini 2.5 Flash-Lite"),
    ]:
        try:
            return _call_gemini(full_prompt, model_name), label, None
        except Exception as e:
            if not any(c in str(e).lower() for c in RATE_CODES):
                return None, None, f"{label}: {e}"
    try:
        return _call_claude_haiku(full_prompt), "Claude Haiku", None
    except Exception as e:
        return None, None, f"All models failed: {e}"

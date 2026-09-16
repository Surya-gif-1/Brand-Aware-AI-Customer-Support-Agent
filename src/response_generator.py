"""
response_generator.py
---------------------
Generates brand-consistent customer support responses using an LLM.

Supported providers (in priority order, set in config.py):
  1. Groq  — via groq Python client
  2. Gemini — via google-generativeai
  3. None  — returns a safe fallback message

Design principles:
  - The LLM is grounded in historical brand responses (retrieved examples).
  - The prompt explicitly forbids inventing policies, amounts, or statuses.
  - The system degrades gracefully if the LLM API is unavailable.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.config import (
    LLM_PROVIDER,
    GROQ_API_KEY,
    GROQ_MODEL_NAME,
    GOOGLE_API_KEY,
    GEMINI_MODEL_NAME,
    LLM_MAX_OUTPUT_TOKENS,
    LLM_TEMPERATURE,
    SELECTED_BRAND,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(
    customer_message: str,
    intent: str,
    historical_examples: list[dict],
    brand: str = SELECTED_BRAND,
) -> str:
    """
    Build the LLM prompt that grounds the response in historical brand examples.

    The prompt:
    - Shows the customer's message and detected intent.
    - Provides up to TOP_K historical (customer, brand) pairs as context.
    - Instructs the model to stay grounded, be concise, and avoid fabrication.
    """
    examples_block = ""
    for i, ex in enumerate(historical_examples, 1):
        score = ex.get("similarity_score", 0.0)
        examples_block += (
            f"\n--- Example {i} (similarity: {score:.2f}) ---\n"
            f"Customer: {ex.get('customer_text', '')}\n"
            f"{brand} Response: {ex.get('brand_response', '')}\n"
        )

    if not examples_block:
        examples_block = "(No similar historical examples available.)"

    prompt = f"""You are a customer support agent for {brand}.

A customer has sent the following message:
\"\"\"{customer_message}\"\"\"

Detected intent: {intent}

Here are {len(historical_examples)} historically similar conversations from {brand}'s support team:
{examples_block}

Your task:
1. Write a helpful, professional, and empathetic response to the customer.
2. Keep your response concise (2-4 sentences maximum).
3. Use the historical examples ONLY as grounding for tone and style — do NOT copy them verbatim.
4. Do NOT invent specific refund amounts, order dates, account details, or policies not shown in the examples.
5. If the issue requires account verification or complex investigation, acknowledge this and guide the customer appropriately.
6. If the historical evidence is insufficient to resolve the issue, say so honestly.
7. Do NOT start your response with "I" or "As an AI".

Write only the response text, nothing else."""

    return prompt


# ---------------------------------------------------------------------------
# LLM clients
# ---------------------------------------------------------------------------

def _call_groq(prompt: str) -> str:
    """Call Groq API using the groq Python client."""
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=LLM_MAX_OUTPUT_TOKENS,
            temperature=LLM_TEMPERATURE,
        )
        return completion.choices[0].message.content.strip()
    except ImportError:
        raise ImportError(
            "The 'groq' package is required for Groq API. Run: pip install groq"
        )
    except Exception as e:
        raise RuntimeError(f"Groq API call failed: {e}") from e


def _call_gemini(prompt: str) -> str:
    """Call Gemini API using google-generativeai."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
                temperature=LLM_TEMPERATURE,
            ),
        )
        return response.text.strip()
    except ImportError:
        raise ImportError(
            "The 'google-generativeai' package is required. "
            "Run: pip install google-generativeai"
        )
    except Exception as e:
        raise RuntimeError(f"Gemini API call failed: {e}") from e


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def generate_response(
    customer_message: str,
    intent: str,
    historical_examples: list[dict],
    brand: str = SELECTED_BRAND,
) -> dict:
    """
    Generate a brand-consistent customer support response.

    Parameters
    ----------
    customer_message   : cleaned customer tweet
    intent             : predicted intent label
    historical_examples: list of dicts from Retriever.retrieve()
    brand              : brand name string

    Returns
    -------
    dict with keys:
        response        : generated text (or fallback message)
        provider        : 'groq' | 'gemini' | 'none'
        status          : 'success' | 'fallback' | 'error'
        error           : error message string (if status == 'error')
    """
    if LLM_PROVIDER == "none":
        return {
            "response": (
                "⚠️  LLM response generation is disabled because no API key was found.\n"
                "Set GROQ_API_KEY or GOOGLE_API_KEY in your .env file to enable this feature."
            ),
            "provider": "none",
            "status": "fallback",
            "error": None,
        }

    prompt = _build_prompt(customer_message, intent, historical_examples, brand)

    try:
        if LLM_PROVIDER == "groq":
            text = _call_groq(prompt)
            provider = "groq"
        else:
            text = _call_gemini(prompt)
            provider = "gemini"

        return {
            "response": text,
            "provider": provider,
            "status": "success",
            "error": None,
        }

    except Exception as exc:
        logger.error("LLM generation failed: %s", exc)
        fallback = (
            "Thank you for reaching out. We've received your message and "
            "a member of our support team will get back to you as soon as possible."
        )
        return {
            "response": fallback,
            "provider": LLM_PROVIDER,
            "status": "error",
            "error": str(exc),
        }

"""
OpenRouter adapter — multi-model LLM gateway with cost-tiered fallback chain.

Fallback chain (settings.openrouter_fallback_chain):
    1. moonshotai/kimi-k2.5      finance #1 by token volume    $0.40/$1.90
    2. deepseek/deepseek-v4-flash  finance #6, dirt cheap        $0.112/$0.224
    3. deepseek/deepseek-v4-pro    finance #11, mid-tier         $0.435/$0.87
    4. moonshotai/kimi-k2.6        finance #21, premium fallback $0.73/$3.49

Behavior:
- chat(): try chain in order. On 429/5xx/timeout/invalid-JSON, advance.
- score_sentiment(): bulk task — pin to model_fast (DeepSeek V4 Flash).
- smart_completion(): reasoning task — full fallback chain starting at K2.5.

Cost-controlled: total spend ~$2.40 for 90 days at top-100 universe.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

from quant.adapters.base import HttpAdapter
from quant.adapters.exceptions import DataQualityError
from quant.config import settings

log = logging.getLogger("quant.adapters.openrouter")

SentimentLabel = Literal["bearish", "neutral", "bullish"]

SENTIMENT_SYSTEM = (
    "You score financial news headlines for market impact on the listed tickers. "
    "Return ONLY minified JSON of the form "
    '{"score": <float in [-1,1]>, "label": "bearish"|"neutral"|"bullish", "rationale": "<1 sentence>"}. '
    "Do not include any prose outside the JSON."
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class OpenRouterAdapter(HttpAdapter):
    name = "openrouter"
    base_url = "https://openrouter.ai/api/v1"
    calls_per_minute = 60

    def default_headers(self) -> dict[str, str]:
        key = settings.openrouter_api_key
        if key is None:
            raise RuntimeError("openrouter_api_key not configured in .env.local")
        return {
            "Authorization": f"Bearer {key.get_secret_value()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "HTTP-Referer": "https://github.com/ShAuRyA-Noodle/Silky-Blade",
            "X-Title": "Oracle Quant Platform",
        }

    async def _chat(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        response_format: dict[str, str] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if response_format is not None:
            payload["response_format"] = response_format
        data = await self.post_json("/chat/completions", json=payload)
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as e:
            raise DataQualityError(f"[openrouter:{model}] malformed completion: {data!r}") from e

    async def _chat_with_fallback(
        self,
        *,
        models: list[str],
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        response_format: dict[str, str] | None = None,
    ) -> tuple[str, str]:
        """Try each model in `models`. Return (response_text, model_used)."""
        last_exc: Exception | None = None
        for model in models:
            try:
                text = await self._chat(
                    model=model,
                    system=system,
                    user=user,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )
                return text, model
            except Exception as exc:
                log.warning("openrouter model %s failed: %s — trying next in chain", model, exc)
                last_exc = exc
                continue
        raise DataQualityError(f"[openrouter] all fallback models failed: {models!r}") from last_exc

    def _fallback_chain(self) -> list[str]:
        """Smart-tier fallback chain — K2.5 → Flash → Pro → K2.6."""
        return [
            settings.openrouter_model_smart,  # kimi-k2.5
            settings.openrouter_model_fast,  # deepseek-v4-flash
            "deepseek/deepseek-v4-pro",
            "moonshotai/kimi-k2.6",
        ]

    # ------------------------------------------------------------
    # Sentiment — pinned to fast model (DeepSeek V4 Flash). Bulk grunt work.
    # ------------------------------------------------------------
    async def score_sentiment(
        self, *, headline: str, summary: str | None, tickers: list[str]
    ) -> dict[str, Any]:
        user = (
            f"Tickers: {', '.join(tickers) if tickers else '(none specified)'}\n"
            f"Headline: {headline}\n"
            f"Summary: {summary or '(none)'}"
        )
        raw = await self._chat(
            model=settings.openrouter_model_fast,
            system=SENTIMENT_SYSTEM,
            user=user,
            temperature=0.0,
            max_tokens=150,
            response_format={"type": "json_object"},
        )
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            m = _JSON_RE.search(raw)
            if not m:
                raise DataQualityError(f"[openrouter] non-JSON sentiment: {raw[:200]}") from None
            obj = json.loads(m.group(0))

        score = float(obj.get("score", 0.0))
        score = max(-1.0, min(1.0, score))
        label = obj.get("label", "neutral")
        if label not in ("bearish", "neutral", "bullish"):
            label = "neutral"
        return {
            "score": score,
            "label": label,
            "rationale": str(obj.get("rationale", ""))[:500],
            "model": settings.openrouter_model_fast,
        }

    # ------------------------------------------------------------
    # Smart completion — full fallback chain. For catalyst/regime/sanity/briefing.
    # ------------------------------------------------------------
    async def smart_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> tuple[dict[str, Any], str]:
        """JSON-strict smart-tier call with full fallback chain. Returns (obj, model_used)."""
        raw, model_used = await self._chat_with_fallback(
            models=self._fallback_chain(),
            system=system,
            user=user,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            m = _JSON_RE.search(raw)
            if not m:
                raise DataQualityError(
                    f"[openrouter:{model_used}] non-JSON smart completion: {raw[:200]}"
                ) from None
            obj = json.loads(m.group(0))
        return obj, model_used

    async def smart_text(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.3,
        max_tokens: int = 600,
    ) -> tuple[str, str]:
        """Free-text smart-tier call with full fallback chain."""
        return await self._chat_with_fallback(
            models=self._fallback_chain(),
            system=system,
            user=user,
            temperature=temperature,
            max_tokens=max_tokens,
        )


__all__ = ["OpenRouterAdapter"]

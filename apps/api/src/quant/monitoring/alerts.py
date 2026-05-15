"""
Lightweight alert dispatcher.

Reads SLACK_WEBHOOK_URL from environment. If set, POSTs Slack-compatible
JSON to the webhook. If not set, logs only. Designed to be called from the
execution path without blowing up if the webhook is unavailable.

Usage::

    from quant.monitoring.alerts import send_alert

    await send_alert("Order submission failed mid-rebalance", level="critical")
"""

from __future__ import annotations

import logging
import os
from typing import Literal

log = logging.getLogger("quant.monitoring.alerts")

AlertLevel = Literal["info", "warning", "error", "critical"]

_EMOJI: dict[str, str] = {
    "info": ":white_circle:",
    "warning": ":large_yellow_circle:",
    "error": ":red_circle:",
    "critical": ":rotating_light:",
}


async def send_alert(message: str, level: AlertLevel = "info") -> None:
    """
    Fire-and-forget alert. Never raises — a failed alert must not crash
    the execution path that called it.
    """
    emoji = _EMOJI.get(level, ":white_circle:")
    formatted = f"{emoji} *oracle/{level}*: {message}"
    log.log(
        logging.CRITICAL if level == "critical" else logging.ERROR if level == "error" else logging.WARNING,
        "[alert] %s",
        message,
    )

    url = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
    if not url:
        return

    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(url, json={"text": formatted})
            if resp.status_code not in (200, 204):
                log.warning("slack webhook returned %d", resp.status_code)
    except ImportError:
        log.debug("httpx not installed — slack alert skipped")
    except Exception as exc:
        log.warning("slack webhook failed: %s", exc)


def send_alert_sync(message: str, level: AlertLevel = "info") -> None:
    """Synchronous wrapper for use in non-async contexts."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(send_alert(message, level))
        else:
            loop.run_until_complete(send_alert(message, level))
    except Exception:
        log.warning("sync alert dispatch failed", exc_info=True)


__all__ = ["send_alert", "send_alert_sync", "AlertLevel"]

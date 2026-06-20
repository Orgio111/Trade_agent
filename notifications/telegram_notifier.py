"""QUANTEX Telegram Notifier — Async Non-Blocking MarkdownV2 Messenger.

Sends trade signals, risk alerts, and system status to Telegram.
All sends run as asyncio background tasks — never blocks the main event loop.
Uses MarkdownV2 formatting for rich messages.

Usage:
    from notifications.telegram_notifier import TelegramNotifier

    notifier = TelegramNotifier()
    await notifier.start()

    # Fire-and-forget notifications
    notifier.signal_buy("BTC/USDT", 67500.0, 0.42, 0.85, 66800.0, 69200.0)
    notifier.signal_sell("BTC/USDT", 67500.0, -0.38, 0.78, 68200.0, 65800.0)
    notifier.risk_alert("Dead Man's Switch tripped — no heartbeat for 3s")
    notifier.system_status("🟢 All systems operational")
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import httpx
from dotenv import dotenv_values

logger = logging.getLogger(__name__)


# ── MarkdownV2 Escape ──────────────────────────────────────────

# Characters that must be escaped in Telegram MarkdownV2
_MD2_ESCAPE_RE = re.compile(r"([_*\[\]()~`>#+\-=|{}.!\\])")


def escape_md2(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2 parsing."""
    return _MD2_ESCAPE_RE.sub(r"\\\1", text)


# ── Configuration ──────────────────────────────────────────────

@dataclass
class TelegramConfig:
    bot_token: str
    chat_id: str
    parse_mode: str = "MarkdownV2"
    base_url: str = "https://api.telegram.org"
    timeout_sec: float = 10.0
    max_retries: int = 3
    retry_delay_sec: float = 1.0
    enabled: bool = True


def _load_config() -> TelegramConfig:
    """Load Telegram configuration from .env file."""
    env = dotenv_values(".env")

    bot_token = env.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = env.get("TELEGRAM_CHAT_ID", "")

    return TelegramConfig(
        bot_token=bot_token,
        chat_id=chat_id,
        enabled=bool(bot_token and chat_id),
    )


# ── Notifier ───────────────────────────────────────────────────

class TelegramNotifier:
    """Async non-blocking Telegram notifier for QUANTEX trading bot.

    All message sends are dispatched as background tasks (asyncio.create_task)
    so they never block the main trading event loop.
    """

    def __init__(self, config: Optional[TelegramConfig] = None):
        self._config = config or _load_config()
        self._client: Optional[httpx.AsyncClient] = None
        self._send_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=2048)
        self._worker_task: Optional[asyncio.Task] = None
        self._stats = _SendStats()

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    # ── Lifecycle ──────────────────────────────────────────

    async def start(self) -> None:
        """Initialize HTTP client and start the background send worker."""
        if not self._config.enabled:
            logger.warning("Telegram notifier DISABLED — no token/chat_id configured")
            return

        self._client = httpx.AsyncClient(
            base_url=f"{self._config.base_url}/bot{self._config.bot_token}",
            timeout=httpx.Timeout(self._config.timeout_sec),
        )

        # Start background worker that drains the queue
        self._worker_task = asyncio.create_task(self._send_worker())
        logger.info("Telegram notifier started — chat_id=%s", self._config.chat_id)

    async def stop(self) -> None:
        """Gracefully shut down: drain remaining messages, close client."""
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        if self._client:
            await self._client.aclose()

        logger.info(
            "Telegram notifier stopped — sent=%d failed=%d",
            self._stats.sent, self._stats.failed,
        )

    # ── Public Notification Methods ───────────────────────

    def signal_buy(
        self,
        symbol: str,
        entry_price: float,
        score: float,
        confidence: float,
        stop_loss: float,
        take_profit: float,
        quantity: float = 0.0,
    ) -> None:
        """Fire-and-forget BUY signal notification."""
        text = (
            f"📈 *BUY SIGNAL*\n\n"
            f"Symbol: {escape_md2(symbol)}\n"
            f"Entry: `{entry_price:.2f}`\n"
            f"Score: `{score:+.4f}` \\| Confidence: `{confidence:.2f}`\n"
            f"Stop Loss: `{stop_loss:.2f}`\n"
            f"Take Profit: `{take_profit:.2f}`\n"
        )
        if quantity > 0:
            text += f"Quantity: `{quantity:.6f}`\n"

        self._enqueue(text)

    def signal_sell(
        self,
        symbol: str,
        entry_price: float,
        score: float,
        confidence: float,
        stop_loss: float,
        take_profit: float,
        quantity: float = 0.0,
    ) -> None:
        """Fire-and-forget SELL signal notification."""
        text = (
            f"📉 *SELL SIGNAL*\n\n"
            f"Symbol: {escape_md2(symbol)}\n"
            f"Entry: `{entry_price:.2f}`\n"
            f"Score: `{score:+.4f}` \\| Confidence: `{confidence:.2f}`\n"
            f"Stop Loss: `{stop_loss:.2f}`\n"
            f"Take Profit: `{take_profit:.2f}`\n"
        )
        if quantity > 0:
            text += f"Quantity: `{quantity:.6f}`\n"

        self._enqueue(text)

    def signal_hold(self, symbol: str, score: float) -> None:
        """Fire-and-forget HOLD status notification."""
        text = (
            f"⏸️ *HOLD* — {escape_md2(symbol)}\n"
            f"Score: `{score:+.4f}` \\| No action\\."
        )
        self._enqueue(text)

    def risk_alert(self, message: str) -> None:
        """Fire-and-forget risk alert notification."""
        text = f"🚨 *RISK ALERT*\n\n{escape_md2(message)}"
        self._enqueue(text)

    def dms_status(self, alive: bool, elapsed_ms: int = 0) -> None:
        """Dead Man's Switch status change notification."""
        if alive:
            text = "🟢 *DMS*: Heartbeat resumed — entries re\\-enabled\\."
        else:
            text = f"🔴 *DMS*: NO heartbeat for `{elapsed_ms}ms` — HALTING new entries\\!"
        self._enqueue(text)

    def order_placed(
        self,
        side: str,
        symbol: str,
        quantity: float,
        price: float,
        order_id: str,
        is_paper: bool = True,
    ) -> None:
        """Order placed notification."""
        paper_tag = " \\[PAPER\\]" if is_paper else ""
        emoji = "🟢" if side == "BUY" else "🔴"
        text = (
            f"{emoji} *ORDER PLACED*{paper_tag}\n\n"
            f"{escape_md2(side)} {escape_md2(symbol)}\n"
            f"Qty: `{quantity:.6f}` @ `{price:.2f}`\n"
            f"ID: `{escape_md2(order_id)}`\n"
        )
        self._enqueue(text)

    def bracket_placed(
        self,
        symbol: str,
        stop_loss: float,
        take_profit: float,
        order_id: str = "",
    ) -> None:
        """OCO bracket order placed notification."""
        text = (
            f"🎯 *BRACKET ORDER*\n\n"
            f"{escape_md2(symbol)}\n"
            f"SL: `{stop_loss:.2f}` \\| TP: `{take_profit:.2f}`\n"
        )
        if order_id:
            text += f"ID: `{escape_md2(order_id)}`\n"
        self._enqueue(text)

    def system_status(self, message: str) -> None:
        """Fire-and-forget system status notification."""
        text = f"⚡ *SYSTEM*: {escape_md2(message)}"
        self._enqueue(text)

    async def send_test(self) -> bool:
        """Send a test message and return True if successful."""
        return await self._raw_send("🤖 QUANTEX Telegram notifier — connection test OK ✅")

    # ── Internal ───────────────────────────────────────────

    def _enqueue(self, text: str) -> None:
        """Enqueue a message for async sending (non-blocking)."""
        if not self._config.enabled:
            return
        try:
            self._send_queue.put_nowait(text)
        except asyncio.QueueFull:
            logger.warning("Telegram send queue full — dropping message")

    async def _send_worker(self) -> None:
        """Background worker: drains the queue and sends messages with rate limiting."""
        while True:
            try:
                text = await self._send_queue.get()
                success = await self._raw_send(text)
                if not success:
                    self._stats.failed += 1
                else:
                    self._stats.sent += 1
                # Rate-limit: ~30 messages/sec max per Telegram Bot API
                await asyncio.sleep(0.033)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Telegram send worker error: %s", exc)
                self._stats.failed += 1
                await asyncio.sleep(1.0)

    async def _raw_send(self, text: str) -> bool:
        """Send a raw text message to Telegram with retries."""
        if not self._client:
            return False

        payload = {
            "chat_id": self._config.chat_id,
            "text": text,
            "parse_mode": self._config.parse_mode,
            "disable_web_page_preview": True,
        }

        for attempt in range(self._config.max_retries):
            try:
                resp = await self._client.post("/sendMessage", json=payload)
                if resp.status_code == 200:
                    return True
                if resp.status_code == 429:
                    # Rate-limited: respect Retry-After header
                    retry_after = float(resp.headers.get("Retry-After", "2"))
                    logger.warning("Telegram rate-limited, retry after %.1fs", retry_after)
                    await asyncio.sleep(retry_after)
                    continue
                logger.error(
                    "Telegram API error %d: %s",
                    resp.status_code,
                    resp.text[:200],
                )
                return False
            except (httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
                logger.warning("Telegram timeout (attempt %d/%d): %s", attempt + 1, self._config.max_retries, exc)
                await asyncio.sleep(self._config.retry_delay_sec * (attempt + 1))
            except httpx.HTTPError as exc:
                logger.error("Telegram HTTP error: %s", exc)
                await asyncio.sleep(self._config.retry_delay_sec)
                return False

        logger.error("Telegram send failed after %d retries", self._config.max_retries)
        return False


@dataclass
class _SendStats:
    """Internal send statistics."""
    sent: int = 0
    failed: int = 0


# ── Standalone CLI Test ────────────────────────────────────────

async def _cli_test() -> None:
    """Quick CLI test: send a test message to verify Telegram connectivity."""
    import sys

    notifier = TelegramNotifier()
    if not notifier.enabled:
        print("ERROR: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in .env")
        sys.exit(1)

    await notifier.start()
    ok = await notifier.send_test()
    await notifier.stop()
    print(f"Test message {'SENT OK' if ok else 'FAILED'}")


if __name__ == "__main__":
    asyncio.run(_cli_test())

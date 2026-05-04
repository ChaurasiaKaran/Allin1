"""
Algo Bot By FREAK — Paper Trading Engine V2.1
Trading engine identical to V2.0. UI fully redesigned:
  - Sectioned dashboard (status / perf / positions / watchlist)
  - Live unrealized PnL & R-multiples on open positions
  - Card-style trade open/close notifications with duration
  - Submenus: Positions, History, Settings, Help
  - Dynamic Start/Pause toggle, confirmation flow for Reset DD
  - Persistent recent-trades log (last 20)
"""

import os
import time
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timezone

import aiohttp
import pandas as pd
import numpy as np
import pandas_ta as ta

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from telegram.error import BadRequest
from telegram.constants import ParseMode

# ==========================================
# CONFIG
# ==========================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID", "0"))
STATE_FILE = Path(os.getenv("STATE_FILE", "/data/state.json"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

INITIAL_BALANCE = 100.0
BASE_RISK_PCT = 0.02
MAX_OPEN_POSITIONS = 2
DAILY_DD_HALT_PCT = 0.05
COOLDOWN_BARS_AFTER_LOSS = 2
RR_RATIO = 3.0
MIN_SL_PCT = 0.015
TAKER_FEE = 0.00035
VOL_SPIKE_MULT = 1.5
EMA_TREND_LEN = 200
SCANNER_INTERVAL_SEC = 30
RECENT_TRADES_KEEP = 20

APEX_ROUTING = {
    "HYPE": {"tf": "15m", "strategy": "SWEEP_ONLY"},
    "TAO":  {"tf": "30m", "strategy": "HYBRID"},
    "ENA":  {"tf": "1h",  "strategy": "HYBRID"},
}

COIN_ICON = {"HYPE": "💎", "TAO": "🧠", "ENA": "⚡"}

TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400}

DIVIDER = "━━━━━━━━━━━━━━━━━━━━"

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("freak-bot")
logging.getLogger("httpx").setLevel(logging.WARNING)


# ==========================================
# STATE
# ==========================================
class TradingState:
    # Fields excluded from persistence (in-memory only)
    _TRANSIENT = {"last_prices"}

    def __init__(self):
        self.balance = INITIAL_BALANCE
        self.open_positions = {}        # coin -> dict
        self.last_processed_bar = {}    # coin -> int unix seconds (last closed candle ts)
        self.cooldown_until = {}        # coin -> unix seconds
        self.bot_active = False
        self.daily_start_balance = INITIAL_BALANCE
        self.daily_start_date = None    # ISO date string (UTC)
        self.daily_halted = False
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.recent_trades = []         # list of dicts (capped at RECENT_TRADES_KEEP)
        self.last_prices = {}           # coin -> last seen close (transient)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if k not in self._TRANSIENT}

    @classmethod
    def from_dict(cls, d):
        s = cls()
        for k, v in d.items():
            if hasattr(s, k):
                setattr(s, k, v)
        return s

    def save(self, path: Path):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(self.to_dict(), indent=2))
            tmp.replace(path)
        except Exception as e:
            logger.error(f"state save failed: {e}")

    @classmethod
    def load(cls, path: Path):
        if not path.exists():
            logger.info(f"no existing state at {path}, starting fresh")
            return cls()
        try:
            data = json.loads(path.read_text())
            logger.info(f"state loaded from {path}")
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"state load failed ({e}), starting fresh")
            return cls()


# ==========================================
# MARKET DATA
# ==========================================
async def fetch_data(session: aiohttp.ClientSession, coin: str, interval: str):
    url = "https://api.hyperliquid.xyz/info"
    bars_needed = max(800, EMA_TREND_LEN * 4)
    seconds = TF_SECONDS[interval]
    start_time = int((time.time() - bars_needed * seconds) * 1000)
    end_time = int(time.time() * 1000)
    payload = {
        "type": "candleSnapshot",
        "req": {"coin": coin, "interval": interval,
                "startTime": start_time, "endTime": end_time},
    }
    try:
        async with session.post(url, json=payload,
                                timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                logger.warning(f"fetch {coin} {interval} HTTP {resp.status}")
                return None
            res = await resp.json()
        if not res:
            return None
        df = pd.DataFrame(res)
        df["datetime"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        for col in ["o", "h", "l", "c", "v"]:
            df[col] = df[col].astype(float)
        df = df.rename(columns={"o": "open", "h": "high",
                                "l": "low", "c": "close", "v": "volume"})
        return df
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.warning(f"fetch {coin} {interval} failed: {e}")
        return None


# ==========================================
# STRATEGY
# ==========================================
def analyze_market(df: pd.DataFrame, strategy_type: str):
    if df is None or len(df) < EMA_TREND_LEN + 30:
        return None

    df = df.copy()
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["buy_vol"] = df["volume"] * ((df["close"] - df["low"]) / rng)
    df["sell_vol"] = df["volume"] * ((df["high"] - df["close"]) / rng)
    df[["buy_vol", "sell_vol"]] = df[["buy_vol", "sell_vol"]].fillna(0)
    df["delta"] = df["buy_vol"] - df["sell_vol"]

    df["cvd"] = df["delta"].rolling(window=100, min_periods=20).sum()
    df["cvd_roc"] = df["cvd"].diff(3)
    df["cvd_bullish"] = df["cvd"] > df["cvd"].shift(3)

    df["ema_macro"] = ta.ema(df["close"], length=EMA_TREND_LEN)
    df["trend_bullish"] = df["close"] > df["ema_macro"]

    df["prev_swing_low"] = df["low"].rolling(window=20).min().shift(1)
    df["prev_swing_high"] = df["high"].rolling(window=20).max().shift(1)
    df["vol_ema"] = ta.ema(df["volume"], length=20)
    df["vol_spike"] = df["volume"] > (df["vol_ema"] * VOL_SPIKE_MULT)

    df["sweep_long"] = (df["low"] < df["prev_swing_low"]) & (df["close"] > df["prev_swing_low"])
    df["sweep_short"] = (df["high"] > df["prev_swing_high"]) & (df["close"] < df["prev_swing_high"])

    sig_sweep_long = df["sweep_long"] & df["vol_spike"] & (df["cvd_roc"] > 0) & df["trend_bullish"]
    sig_sweep_short = df["sweep_short"] & df["vol_spike"] & (df["cvd_roc"] < 0) & ~df["trend_bullish"]

    period = 20
    df["sma_20"] = ta.sma(df["close"], length=period)
    df["std_dev"] = ta.stdev(df["close"], length=period)
    df["bb_upper"] = df["sma_20"] + (df["std_dev"] * 2.0)
    df["bb_lower"] = df["sma_20"] - (df["std_dev"] * 2.0)
    df["kc_upper"] = df["sma_20"] + (df["atr"] * 1.5)
    df["kc_lower"] = df["sma_20"] - (df["atr"] * 1.5)

    df["is_squeezed"] = (df["bb_upper"] < df["kc_upper"]) & (df["bb_lower"] > df["kc_lower"])
    df["squeeze_release"] = (~df["is_squeezed"]) & df["is_squeezed"].shift(1).fillna(False)
    df["price_break_up"] = df["close"] > df["kc_upper"]
    df["price_break_down"] = df["close"] < df["kc_lower"]

    sig_squeeze_long = df["squeeze_release"] & df["price_break_up"] & df["cvd_bullish"]
    sig_squeeze_short = df["squeeze_release"] & df["price_break_down"] & ~df["cvd_bullish"]

    if strategy_type == "SWEEP_ONLY":
        df["long_signal"] = sig_sweep_long
        df["short_signal"] = sig_sweep_short
    else:  # HYBRID
        df["long_signal"] = sig_sweep_long | sig_squeeze_long
        df["short_signal"] = sig_sweep_short | sig_squeeze_short

    return df.iloc[-2]  # last CLOSED candle


# ==========================================
# UI HELPERS
# ==========================================
def fmt_money(v: float, signed: bool = False) -> str:
    if signed:
        sign = "+" if v >= 0 else "-"
        return f"{sign}${abs(v):,.2f}"
    return f"${v:,.2f}"


def fmt_pct(v: float, signed: bool = True) -> str:
    if signed:
        sign = "+" if v >= 0 else "-"
        return f"{sign}{abs(v):.2f}%"
    return f"{v:.2f}%"


def fmt_price(p: float) -> str:
    if p >= 1000:
        return f"${p:,.2f}"
    if p >= 1:
        return f"${p:.4f}"
    return f"${p:.6f}"


def fmt_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    m = seconds // 60
    if m < 60:
        return f"{int(m)}m"
    h, m = divmod(int(m), 60)
    if h < 24:
        return f"{h}h {m}m" if m else f"{h}h"
    d, h = divmod(h, 24)
    return f"{d}d {h}h" if h else f"{d}d"


def fmt_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%b %d %H:%M")


def coin_tag(coin: str) -> str:
    return f"{COIN_ICON.get(coin, '🪙')} <b>#{coin}</b>"


def status_pill(active: bool, halted: bool) -> str:
    if halted:
        return "⛔ <b>HALTED</b>"
    return "🟢 <b>ACTIVE</b>" if active else "🟡 <b>PAUSED</b>"


# ==========================================
# TELEGRAM HELPERS
# ==========================================
def is_authorized(update: Update) -> bool:
    return bool(update.effective_chat and update.effective_chat.id == TELEGRAM_CHAT_ID)


async def safe_send(app: Application, text: str, keyboard: InlineKeyboardMarkup = None):
    try:
        await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text,
                                   parse_mode=ParseMode.HTML,
                                   reply_markup=keyboard,
                                   disable_web_page_preview=True)
    except Exception as e:
        logger.warning(f"send_message failed: {e}")


async def safe_edit(query, text, keyboard):
    try:
        await query.edit_message_text(text=text, reply_markup=keyboard,
                                      parse_mode=ParseMode.HTML,
                                      disable_web_page_preview=True)
    except BadRequest as e:
        if "not modified" in str(e).lower():
            return
        logger.warning(f"edit_message_text failed: {e}")


# ==========================================
# KEYBOARDS
# ==========================================
def kb_main(state: "TradingState") -> InlineKeyboardMarkup:
    toggle = (
        InlineKeyboardButton("⏸  Pause Bot", callback_data="pause")
        if state.bot_active
        else InlineKeyboardButton("▶️  Start Bot", callback_data="start")
    )
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄  Refresh", callback_data="dashboard"),
         InlineKeyboardButton("💼  Positions", callback_data="positions")],
        [InlineKeyboardButton("📜  History", callback_data="history"),
         InlineKeyboardButton("⚙️  Settings", callback_data="settings")],
        [toggle],
        [InlineKeyboardButton("⛔  Reset DD Halt", callback_data="reset_ask"),
         InlineKeyboardButton("ℹ️  Help", callback_data="help")],
    ])


def kb_back() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("« Back to Dashboard", callback_data="dashboard")],
    ])


def kb_positions_view(state: "TradingState") -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄  Refresh", callback_data="positions")],
        [InlineKeyboardButton("« Back to Dashboard", callback_data="dashboard")],
    ])


def kb_reset_confirm() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅  Confirm Reset", callback_data="reset_yes"),
         InlineKeyboardButton("❌  Cancel", callback_data="dashboard")],
    ])


# ==========================================
# VIEWS
# ==========================================
def view_dashboard(state: "TradingState") -> str:
    win_rate = (state.wins / state.total_trades * 100) if state.total_trades else 0
    daily_pnl = state.balance - state.daily_start_balance
    daily_pct = (daily_pnl / state.daily_start_balance * 100) if state.daily_start_balance else 0

    lines = [
        "🤖  <b>ALGO BOT  ·  PRO</b>",
        "<i>Paper Trading Engine v2.1</i>",
        DIVIDER,
        "",
        f"{status_pill(state.bot_active, state.daily_halted)}",
        f"💰  <b>{fmt_money(state.balance)}</b>",
        f"📈  {fmt_money(daily_pnl, signed=True)}  ({fmt_pct(daily_pct)})  today",
        "",
        "<b>━━  Performance  ━━</b>",
        f"<pre>"
        f"Trades   {state.total_trades}\n"
        f"Wins     {state.wins}\n"
        f"Losses   {state.losses}\n"
        f"WinRate  {win_rate:.1f}%"
        f"</pre>",
        f"<b>━━  Open Positions  {len(state.open_positions)} / {MAX_OPEN_POSITIONS}  ━━</b>",
    ]

    if not state.open_positions:
        lines.append("<i>None open</i>")
    else:
        for coin, pos in state.open_positions.items():
            unreal = unrealized_pnl(pos, state.last_prices.get(coin))
            pnl_str = (
                f"  →  {fmt_money(unreal, signed=True)}"
                if unreal is not None else ""
            )
            be = "  ·  BE" if pos.get("be_moved") else ""
            arrow = "🟢" if pos["type"] == "LONG" else "🔴"
            lines.append(f"{arrow}  {coin_tag(coin)}  {pos['type']}{be}{pnl_str}")

    lines += [
        "",
        "<b>━━  Watchlist  ━━</b>",
    ]
    for coin, cfg in APEX_ROUTING.items():
        last = state.last_prices.get(coin)
        price_str = f"  ·  {fmt_price(last)}" if last else ""
        lines.append(f"{COIN_ICON.get(coin, '🪙')}  <b>{coin}</b>  ·  {cfg['tf']}{price_str}")

    return "\n".join(lines)


def view_positions(state: "TradingState") -> str:
    lines = [
        "💼  <b>OPEN POSITIONS</b>",
        DIVIDER,
        "",
    ]
    if not state.open_positions:
        lines.append("<i>No open positions right now.</i>")
        lines.append("")
        lines.append("Signals are scanned every 30s on the watchlist timeframes.")
        return "\n".join(lines)

    now = int(time.time())
    for coin, pos in state.open_positions.items():
        last = state.last_prices.get(coin)
        unreal = unrealized_pnl(pos, last)
        r_mult = unrealized_r(pos, last)
        held = fmt_duration(now - pos.get("opened_at", now))
        be = "  (moved to BE)" if pos.get("be_moved") else ""
        arrow = "🟢" if pos["type"] == "LONG" else "🔴"

        # Distances
        sl_pct = (pos["sl"] - pos["entry"]) / pos["entry"] * 100
        tp_pct = (pos["tp"] - pos["entry"]) / pos["entry"] * 100

        lines.append(f"{arrow}  {coin_tag(coin)}  ·  <b>{pos['type']}</b>")
        if last is not None:
            mv_pct = (last - pos["entry"]) / pos["entry"] * 100
            if pos["type"] == "SHORT":
                mv_pct = -mv_pct
            lines.append(f"<pre>"
                         f"Entry   {fmt_price(pos['entry'])}\n"
                         f"Now     {fmt_price(last)}  ({fmt_pct(mv_pct)})\n"
                         f"Stop    {fmt_price(pos['sl'])}  ({fmt_pct(sl_pct)}){be}\n"
                         f"Target  {fmt_price(pos['tp'])}  ({fmt_pct(tp_pct)})\n"
                         f"Size    {pos['size']:.4f}\n"
                         f"PnL     {fmt_money(unreal, signed=True)}  "
                         f"({r_mult:+.2f}R)\n"
                         f"Held    {held}"
                         f"</pre>")
        else:
            lines.append(f"<pre>"
                         f"Entry   {fmt_price(pos['entry'])}\n"
                         f"Stop    {fmt_price(pos['sl'])}  ({fmt_pct(sl_pct)}){be}\n"
                         f"Target  {fmt_price(pos['tp'])}  ({fmt_pct(tp_pct)})\n"
                         f"Size    {pos['size']:.4f}\n"
                         f"Held    {held}"
                         f"</pre>")
    return "\n".join(lines)


def view_history(state: "TradingState") -> str:
    lines = [
        "📜  <b>RECENT TRADES</b>",
        DIVIDER,
        "",
    ]
    if not state.recent_trades:
        lines.append("<i>No trades closed yet.</i>")
        lines.append("Once positions close, the last "
                     f"{RECENT_TRADES_KEEP} will appear here.")
        return "\n".join(lines)

    realized = sum(t["pnl"] for t in state.recent_trades)
    wins = sum(1 for t in state.recent_trades if t["pnl"] > 0)
    losses = sum(1 for t in state.recent_trades if t["pnl"] <= 0)

    lines.append(f"<pre>"
                 f"Shown   {len(state.recent_trades)}\n"
                 f"Wins    {wins}\n"
                 f"Losses  {losses}\n"
                 f"Net     {fmt_money(realized, signed=True)}"
                 f"</pre>")

    for t in reversed(state.recent_trades[-RECENT_TRADES_KEEP:]):
        mark = "✅" if t["pnl"] > 0 else "❌"
        r = t.get("r")
        r_str = f"  ({r:+.2f}R)" if r is not None else ""
        lines.append(
            f"{mark}  {COIN_ICON.get(t['coin'], '🪙')} "
            f"<b>#{t['coin']}</b> {t['type']}  "
            f"{fmt_money(t['pnl'], signed=True)}{r_str}  "
            f"<i>· {fmt_ts(t['ts'])}</i>"
        )
    return "\n".join(lines)


def view_settings() -> str:
    lines = [
        "⚙️  <b>BOT SETTINGS</b>",
        DIVIDER,
        "",
        "<b>Risk Management</b>",
        f"<pre>"
        f"Risk / trade      {BASE_RISK_PCT * 100:.2f}%\n"
        f"Max positions     {MAX_OPEN_POSITIONS}\n"
        f"Daily DD halt     {DAILY_DD_HALT_PCT * 100:.2f}%\n"
        f"Cooldown (loss)   {COOLDOWN_BARS_AFTER_LOSS} bars\n"
        f"R:R ratio         1 : {RR_RATIO}\n"
        f"Min stop dist     {MIN_SL_PCT * 100:.2f}%"
        f"</pre>",
        "<b>Watchlist</b>",
    ]
    for coin, cfg in APEX_ROUTING.items():
        lines.append(
            f"{COIN_ICON.get(coin, '🪙')}  <b>{coin}</b>  ·  "
            f"<code>{cfg['tf']}</code>  ·  <i>{cfg['strategy']}</i>"
        )
    lines += [
        "",
        "<b>Engine</b>",
        f"<pre>"
        f"Scanner tick      {SCANNER_INTERVAL_SEC}s\n"
        f"EMA trend         {EMA_TREND_LEN}\n"
        f"Vol spike mult    {VOL_SPIKE_MULT:.2f}×\n"
        f"Taker fee         {TAKER_FEE * 100:.3f}%"
        f"</pre>",
    ]
    return "\n".join(lines)


def view_help() -> str:
    return (
        "ℹ️  <b>HELP</b>\n"
        f"{DIVIDER}\n\n"
        "<b>Commands</b>\n"
        "<pre>"
        "/start      Open dashboard\n"
        "/status     Quick status\n"
        "/balance    Show balance\n"
        "/positions  Open positions\n"
        "/history    Recent trades\n"
        "/pause      Pause scanning\n"
        "/resume     Resume scanning"
        "</pre>"
        "<b>Strategies</b>\n"
        "• <b>SWEEP_ONLY</b> — liquidity sweep + CVD confirm\n"
        "• <b>HYBRID</b>     — sweep <i>or</i> squeeze release\n\n"
        "<b>Risk Controls</b>\n"
        f"• {BASE_RISK_PCT * 100:.0f}% risk per trade, sized off ATR stop\n"
        "• Stop moves to breakeven at +1R\n"
        f"• Trading halts after {DAILY_DD_HALT_PCT * 100:.0f}% daily DD (resets at UTC midnight)\n"
        f"• {COOLDOWN_BARS_AFTER_LOSS}-bar cooldown after a loss\n"
        f"• Max {MAX_OPEN_POSITIONS} concurrent positions"
    )


def view_reset_confirm(state: "TradingState") -> str:
    dd = (state.daily_start_balance - state.balance) / state.daily_start_balance * 100 \
        if state.daily_start_balance else 0
    return (
        "⚠️  <b>Confirm Reset</b>\n"
        f"{DIVIDER}\n\n"
        f"Current daily DD: <b>{dd:.2f}%</b>\n"
        f"Daily halt: <b>{'YES' if state.daily_halted else 'no'}</b>\n\n"
        "This will:\n"
        "• Lift the daily drawdown halt\n"
        "• Reset today's start balance to current balance\n\n"
        "Continue?"
    )


# ==========================================
# PNL CALCS (for views)
# ==========================================
def unrealized_pnl(pos: dict, last_price):
    if last_price is None:
        return None
    if pos["type"] == "LONG":
        return (last_price - pos["entry"]) * pos["size"]
    return (pos["entry"] - last_price) * pos["size"]


def unrealized_r(pos: dict, last_price):
    if last_price is None:
        return 0.0
    risk_per_unit = abs(pos["entry"] - pos.get("initial_sl", pos["sl"]))
    if risk_per_unit <= 0:
        return 0.0
    if pos["type"] == "LONG":
        move = last_price - pos["entry"]
    else:
        move = pos["entry"] - last_price
    return move / risk_per_unit


# ==========================================
# POSITION MANAGEMENT
# ==========================================
async def open_position(app: Application, state: TradingState, coin: str,
                        direction: str, fill_price: float, atr: float):
    sl_dist = atr * 2.5
    if (sl_dist / fill_price) < MIN_SL_PCT:
        return False
    size = (state.balance * BASE_RISK_PCT) / sl_dist
    if size <= 0:
        return False
    entry_fee = fill_price * size * TAKER_FEE
    if direction == "LONG":
        sl = fill_price - sl_dist
        tp = fill_price + (sl_dist * RR_RATIO)
    else:
        sl = fill_price + sl_dist
        tp = fill_price - (sl_dist * RR_RATIO)

    risk_dollars = state.balance * BASE_RISK_PCT
    state.open_positions[coin] = {
        "type": direction,
        "entry": fill_price,
        "size": size,
        "sl": sl,
        "tp": tp,
        "initial_sl": sl,
        "be_moved": False,
        "opened_at": int(time.time()),
        "entry_fee": entry_fee,
        "risk_dollars": risk_dollars,
    }
    state.balance -= entry_fee

    arrow = "🟢" if direction == "LONG" else "🔴"
    sl_pct = (sl - fill_price) / fill_price * 100
    tp_pct = (tp - fill_price) / fill_price * 100
    msg = (
        f"{arrow}{arrow}{arrow}  <b>{direction} OPENED</b>\n"
        f"{DIVIDER}\n\n"
        f"{coin_tag(coin)}\n"
        f"<pre>"
        f"Entry   {fmt_price(fill_price)}\n"
        f"Stop    {fmt_price(sl)}  ({fmt_pct(sl_pct)})\n"
        f"Target  {fmt_price(tp)}  ({fmt_pct(tp_pct)})\n"
        f"Size    {size:.4f}\n"
        f"Risk    {fmt_money(risk_dollars)}\n"
        f"R : R   1 : {RR_RATIO:.1f}"
        f"</pre>"
    )
    await safe_send(app, msg)
    logger.info(f"OPEN {direction} {coin} @ {fill_price}")
    return True


async def close_position(app: Application, state: TradingState, coin: str,
                         exit_price: float, reason_label: str, header_emoji: str):
    pos = state.open_positions.get(coin)
    if not pos:
        return
    if pos["type"] == "LONG":
        gross_pnl = (exit_price - pos["entry"]) * pos["size"]
    else:
        gross_pnl = (pos["entry"] - exit_price) * pos["size"]
    exit_fee = exit_price * pos["size"] * TAKER_FEE
    pnl = gross_pnl - exit_fee
    state.balance += gross_pnl - exit_fee

    risk_per_unit = abs(pos["entry"] - pos.get("initial_sl", pos["sl"]))
    r_mult = (
        ((exit_price - pos["entry"]) if pos["type"] == "LONG"
         else (pos["entry"] - exit_price)) / risk_per_unit
        if risk_per_unit > 0 else 0.0
    )
    held = int(time.time()) - pos.get("opened_at", int(time.time()))

    state.total_trades += 1
    if pnl < 0:
        state.losses += 1
        cd_secs = TF_SECONDS[APEX_ROUTING[coin]["tf"]] * COOLDOWN_BARS_AFTER_LOSS
        state.cooldown_until[coin] = time.time() + cd_secs
    else:
        state.wins += 1

    state.recent_trades.append({
        "coin": coin,
        "type": pos["type"],
        "entry": pos["entry"],
        "exit": exit_price,
        "pnl": pnl,
        "r": r_mult,
        "reason": reason_label,
        "ts": int(time.time()),
        "held_s": held,
    })
    state.recent_trades = state.recent_trades[-RECENT_TRADES_KEEP:]

    del state.open_positions[coin]

    drawdown = (state.daily_start_balance - state.balance) / state.daily_start_balance \
        if state.daily_start_balance else 0
    halted_now = False
    if drawdown >= DAILY_DD_HALT_PCT and not state.daily_halted:
        state.daily_halted = True
        halted_now = True

    pnl_pct = (pnl / pos["entry"] / pos["size"]) * 100 if pos["size"] else 0
    msg = (
        f"{header_emoji}  <b>{reason_label}</b>\n"
        f"{DIVIDER}\n\n"
        f"{coin_tag(coin)}  ·  <b>{pos['type']}</b>\n"
        f"<pre>"
        f"Entry   {fmt_price(pos['entry'])}\n"
        f"Exit    {fmt_price(exit_price)}\n"
        f"Held    {fmt_duration(held)}\n"
        f"PnL     {fmt_money(pnl, signed=True)}  ({fmt_pct(pnl_pct)})\n"
        f"R       {r_mult:+.2f}R\n"
        f"Bal     {fmt_money(state.balance)}"
        f"</pre>"
    )
    await safe_send(app, msg)
    logger.info(f"CLOSE {coin} @ {exit_price} pnl={pnl:.2f}")

    if halted_now:
        await safe_send(
            app,
            f"🚨  <b>DAILY DRAWDOWN HALT</b>\n"
            f"{DIVIDER}\n"
            f"Down <b>{drawdown * 100:.2f}%</b> today. "
            "No new entries until UTC midnight."
        )


async def manage_position(app: Application, state: TradingState, coin: str, df: pd.DataFrame):
    pos = state.open_positions[coin]
    live_high = float(df.iloc[-1]["high"])
    live_low = float(df.iloc[-1]["low"])

    sl_hit = tp_hit = False
    if pos["type"] == "LONG":
        if live_low <= pos["sl"]:
            sl_hit = True
        if live_high >= pos["tp"]:
            tp_hit = True
    else:  # SHORT
        if live_high >= pos["sl"]:
            sl_hit = True
        if live_low <= pos["tp"]:
            tp_hit = True

    if sl_hit:
        await close_position(app, state, coin, pos["sl"], "STOP LOSS", "🛑")
        return
    if tp_hit:
        await close_position(app, state, coin, pos["tp"], "TAKE PROFIT", "🎯")
        return

    # Move-to-breakeven at +1R (only once)
    if not pos.get("be_moved"):
        r = abs(pos["entry"] - pos["initial_sl"])
        if pos["type"] == "LONG" and live_high >= pos["entry"] + r:
            pos["sl"] = pos["entry"]
            pos["be_moved"] = True
            await safe_send(
                app,
                f"⚙️  <b>Stop moved to breakeven</b>\n{coin_tag(coin)}  ·  +1R reached"
            )
        elif pos["type"] == "SHORT" and live_low <= pos["entry"] - r:
            pos["sl"] = pos["entry"]
            pos["be_moved"] = True
            await safe_send(
                app,
                f"⚙️  <b>Stop moved to breakeven</b>\n{coin_tag(coin)}  ·  +1R reached"
            )


async def check_entry(app: Application, state: TradingState, coin: str,
                      config: dict, df: pd.DataFrame):
    if len(state.open_positions) >= MAX_OPEN_POSITIONS:
        return
    if state.daily_halted:
        return
    if time.time() < state.cooldown_until.get(coin, 0):
        return

    last_closed = analyze_market(df, config["strategy"])
    if last_closed is None:
        return

    long_sig = bool(last_closed["long_signal"])
    short_sig = bool(last_closed["short_signal"])

    if long_sig and short_sig:
        logger.info(f"{coin}: long+short conflict, skipping")
        return
    if not (long_sig or short_sig):
        return

    fill_price = float(df.iloc[-1]["open"])  # next-bar open
    atr = float(last_closed["atr"])
    if not np.isfinite(fill_price) or not np.isfinite(atr) or atr <= 0:
        return

    direction = "LONG" if long_sig else "SHORT"
    await open_position(app, state, coin, direction, fill_price, atr)


def check_daily_reset(state: TradingState):
    today = datetime.now(timezone.utc).date().isoformat()
    if state.daily_start_date != today:
        state.daily_start_date = today
        state.daily_start_balance = state.balance
        state.daily_halted = False
        logger.info(f"daily reset: start_balance=${state.balance:.2f}")


# ==========================================
# SCANNER LOOP
# ==========================================
async def scanner_loop(app: Application):
    state: TradingState = app.bot_data["state"]
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                check_daily_reset(state)
                # Always update last_prices so dashboard shows live data
                # even when bot is paused.
                for coin, config in APEX_ROUTING.items():
                    df = await fetch_data(session, coin, config["tf"])
                    if df is None or len(df) < EMA_TREND_LEN + 30:
                        continue

                    state.last_prices[coin] = float(df.iloc[-1]["close"])

                    if not state.bot_active:
                        continue

                    if coin in state.open_positions:
                        await manage_position(app, state, coin, df)

                    if coin not in state.open_positions:
                        last_closed_ts = int(df.iloc[-2]["datetime"].timestamp())
                        if state.last_processed_bar.get(coin) != last_closed_ts:
                            await check_entry(app, state, coin, config, df)
                            state.last_processed_bar[coin] = last_closed_ts

                state.save(STATE_FILE)
            except Exception:
                logger.exception("scanner_loop iteration error")

            await asyncio.sleep(SCANNER_INTERVAL_SEC)


# ==========================================
# TELEGRAM HANDLERS — Commands
# ==========================================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    state: TradingState = context.application.bot_data["state"]
    await update.message.reply_text(
        view_dashboard(state),
        reply_markup=kb_main(state),
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    state: TradingState = context.application.bot_data["state"]
    await update.message.reply_text(
        view_dashboard(state),
        reply_markup=kb_main(state),
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )


async def cmd_balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    state: TradingState = context.application.bot_data["state"]
    daily_pnl = state.balance - state.daily_start_balance
    daily_pct = (daily_pnl / state.daily_start_balance * 100) if state.daily_start_balance else 0
    txt = (
        f"💰  <b>Balance</b>  {fmt_money(state.balance)}\n"
        f"📈  Today  {fmt_money(daily_pnl, signed=True)}  ({fmt_pct(daily_pct)})"
    )
    await update.message.reply_text(txt, parse_mode=ParseMode.HTML)


async def cmd_positions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    state: TradingState = context.application.bot_data["state"]
    await update.message.reply_text(
        view_positions(state),
        reply_markup=kb_positions_view(state),
        parse_mode=ParseMode.HTML,
    )


async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    state: TradingState = context.application.bot_data["state"]
    await update.message.reply_text(
        view_history(state),
        reply_markup=kb_back(),
        parse_mode=ParseMode.HTML,
    )


async def cmd_pause(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    state: TradingState = context.application.bot_data["state"]
    state.bot_active = False
    state.save(STATE_FILE)
    await update.message.reply_text(
        "🟡  <b>Bot paused</b>\nNo new entries will be taken.",
        parse_mode=ParseMode.HTML,
        reply_markup=kb_main(state),
    )


async def cmd_resume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    state: TradingState = context.application.bot_data["state"]
    state.bot_active = True
    state.save(STATE_FILE)
    await update.message.reply_text(
        "🟢  <b>Bot active</b>\nScanner is live.",
        parse_mode=ParseMode.HTML,
        reply_markup=kb_main(state),
    )


# ==========================================
# TELEGRAM HANDLERS — Buttons
# ==========================================
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    state: TradingState = context.application.bot_data["state"]
    query = update.callback_query
    data = query.data or ""

    # Side effects first
    toast = None
    if data == "start":
        state.bot_active = True
        toast = "Bot activated"
    elif data == "pause":
        state.bot_active = False
        toast = "Bot paused"
    elif data == "reset_yes":
        state.daily_halted = False
        state.daily_start_balance = state.balance
        toast = "Daily halt reset"

    await query.answer(toast or "")

    if data in ("dashboard", "start", "pause"):
        view, kb = view_dashboard(state), kb_main(state)
    elif data == "positions":
        view, kb = view_positions(state), kb_positions_view(state)
    elif data == "history":
        view, kb = view_history(state), kb_back()
    elif data == "settings":
        view, kb = view_settings(), kb_back()
    elif data == "help":
        view, kb = view_help(), kb_back()
    elif data == "reset_ask":
        view, kb = view_reset_confirm(state), kb_reset_confirm()
    elif data == "reset_yes":
        view, kb = view_dashboard(state), kb_main(state)
    else:
        view, kb = view_dashboard(state), kb_main(state)

    state.save(STATE_FILE)
    await safe_edit(query, view, kb)


# ==========================================
# INITIALIZATION
# ==========================================
async def post_init(app: Application):
    state = TradingState.load(STATE_FILE)
    app.bot_data["state"] = state
    app.bot_data["scanner_task"] = asyncio.create_task(scanner_loop(app))
    await safe_send(
        app,
        f"🟢  <b>Bot online</b>\n"
        f"{DIVIDER}\n"
        f"Balance  {fmt_money(state.balance)}\n"
        f"Status   {'Active' if state.bot_active else 'Paused'}\n\n"
        "Tap <b>/start</b> to open the dashboard.",
        keyboard=kb_main(state),
    )


async def post_shutdown(app: Application):
    state = app.bot_data.get("state")
    if state:
        state.save(STATE_FILE)
    task = app.bot_data.get("scanner_task")
    if task and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def main():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars required")

    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("balance", cmd_balance))
    app.add_handler(CommandHandler("positions", cmd_positions))
    app.add_handler(CommandHandler("history", cmd_history))
    app.add_handler(CommandHandler("pause", cmd_pause))
    app.add_handler(CommandHandler("resume", cmd_resume))
    app.add_handler(CallbackQueryHandler(button_handler))

    logger.info("Algo Bot By FREAK V2.1 starting…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

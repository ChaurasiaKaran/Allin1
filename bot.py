"""
Algo Bot By FREAK — Paper Trading Engine V2.0
Fixes vs V1:
  - SL/TP detected against candle high/low (not just close)
  - Pessimistic SL-first assumption when both hit same candle
  - Async HTTP via aiohttp with timeout (no more event-loop blocking)
  - State persistence to JSON (survives restarts)
  - Move-to-breakeven at 1R
  - Cooldown after losses, max concurrent positions, daily DD circuit breaker
  - Volume spike threshold raised (1.05x -> 1.5x)
  - Long/short conflict skip
  - Fixed EMA length (200) with adequate data fetch
  - Realistic taker fees on both legs
  - Per-bar entry gating (no repeated re-evaluation of the same candle)
  - Auth check on all Telegram handlers
  - HTML parse mode + safe_edit for "message not modified"
  - Better status: TP/SL/BE flag, daily PnL, win rate, trade count
  - /status, /balance, /positions commands in addition to buttons
  - Startup notification
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

APEX_ROUTING = {
    "HYPE": {"tf": "15m", "strategy": "SWEEP_ONLY"},
    "TAO":  {"tf": "30m", "strategy": "HYBRID"},
    "ENA":  {"tf": "1h",  "strategy": "HYBRID"},
}

TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400}

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

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}

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

    # Rolling-window CVD instead of cumulative-from-fetch-start (more stable)
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
# TELEGRAM HELPERS
# ==========================================
def is_authorized(update: Update) -> bool:
    return bool(update.effective_chat and update.effective_chat.id == TELEGRAM_CHAT_ID)


async def safe_send(app: Application, text: str):
    try:
        await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text,
                                   parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.warning(f"send_message failed: {e}")


async def safe_edit(query, text, keyboard):
    try:
        await query.edit_message_text(text=text, reply_markup=keyboard,
                                      parse_mode=ParseMode.HTML)
    except BadRequest as e:
        if "not modified" in str(e).lower():
            return
        logger.warning(f"edit_message_text failed: {e}")


def get_main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📊 Status / Refresh", callback_data="status")],
        [InlineKeyboardButton("▶️ Start Trading", callback_data="start_bot"),
         InlineKeyboardButton("⏸️ Pause Bot", callback_data="stop_bot")],
        [InlineKeyboardButton("🔄 Reset Daily Halt", callback_data="reset_halt")],
    ])


def build_status_text(state: TradingState) -> str:
    win_rate = (state.wins / state.total_trades * 100) if state.total_trades else 0
    daily_pnl = state.balance - state.daily_start_balance
    daily_pct = (daily_pnl / state.daily_start_balance * 100) if state.daily_start_balance else 0

    lines = [
        f"🏦 <b>Balance:</b> ${state.balance:.2f}",
        f"📈 <b>Today:</b> ${daily_pnl:+.2f} ({daily_pct:+.2f}%)",
        f"⚙️ <b>Status:</b> {'🟢 Active' if state.bot_active else '🔴 Paused'}"
        + (" ⛔ DD HALT" if state.daily_halted else ""),
        f"📊 <b>Trades:</b> {state.total_trades} | W {state.wins} / L {state.losses} | WR {win_rate:.1f}%",
        "",
        f"📂 <b>Open Positions ({len(state.open_positions)}/{MAX_OPEN_POSITIONS}):</b>",
    ]
    if not state.open_positions:
        lines.append("<i>None</i>")
    else:
        for coin, pos in state.open_positions.items():
            be_flag = " (BE)" if pos.get("be_moved") else ""
            lines.append(
                f"• #{coin} {pos['type']} | "
                f"E ${pos['entry']:.4f} | SL ${pos['sl']:.4f}{be_flag} | TP ${pos['tp']:.4f}"
            )
    return "\n".join(lines)


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
    }
    state.balance -= entry_fee

    arrow = "🟢" if direction == "LONG" else "🔴"
    risk_dollars = state.balance * BASE_RISK_PCT
    msg = (
        f"{arrow} <b>NEW {direction} EXECUTED</b>\n\n"
        f"Coin: #{coin}\n"
        f"Entry: ${fill_price:.4f}\n"
        f"SL: ${sl:.4f}\n"
        f"TP: ${tp:.4f}\n"
        f"Size: {size:.4f}\n"
        f"Risk: ${risk_dollars:.2f}"
    )
    await safe_send(app, msg)
    logger.info(f"OPEN {direction} {coin} @ {fill_price}")
    return True


async def close_position(app: Application, state: TradingState, coin: str,
                         exit_price: float, reason: str):
    pos = state.open_positions.get(coin)
    if not pos:
        return
    if pos["type"] == "LONG":
        gross_pnl = (exit_price - pos["entry"]) * pos["size"]
    else:
        gross_pnl = (pos["entry"] - exit_price) * pos["size"]
    exit_fee = exit_price * pos["size"] * TAKER_FEE
    # entry_fee already deducted at open
    pnl = gross_pnl - exit_fee
    state.balance += gross_pnl - exit_fee

    state.total_trades += 1
    if pnl < 0:
        state.losses += 1
        cd_secs = TF_SECONDS[APEX_ROUTING[coin]["tf"]] * COOLDOWN_BARS_AFTER_LOSS
        state.cooldown_until[coin] = time.time() + cd_secs
    else:
        state.wins += 1

    del state.open_positions[coin]

    drawdown = (state.daily_start_balance - state.balance) / state.daily_start_balance \
        if state.daily_start_balance else 0
    halted_now = False
    if drawdown >= DAILY_DD_HALT_PCT and not state.daily_halted:
        state.daily_halted = True
        halted_now = True

    msg = (
        f"{reason}\n\n"
        f"Coin: #{coin}\n"
        f"Exit: ${exit_price:.4f}\n"
        f"PnL: ${pnl:+.2f}\n"
        f"Balance: ${state.balance:.2f}"
    )
    await safe_send(app, msg)
    logger.info(f"CLOSE {coin} @ {exit_price} pnl={pnl:.2f}")

    if halted_now:
        await safe_send(
            app,
            f"🚨 <b>DAILY DRAWDOWN HALT</b>\n"
            f"Down {drawdown * 100:.2f}% today. No new entries until UTC midnight."
        )


async def manage_position(app: Application, state: TradingState, coin: str, df: pd.DataFrame):
    pos = state.open_positions[coin]
    # Use the LIVE (currently-forming) candle's high/low so we catch wicks intra-bar.
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

    # Pessimistic same-bar resolution: SL fills first
    if sl_hit:
        await close_position(app, state, coin, pos["sl"], "🛑 STOP LOSS HIT")
        return
    if tp_hit:
        await close_position(app, state, coin, pos["tp"], "🎯 TAKE PROFIT HIT")
        return

    # Move-to-breakeven at +1R (only once)
    if not pos.get("be_moved"):
        r = abs(pos["entry"] - pos["initial_sl"])
        if pos["type"] == "LONG" and live_high >= pos["entry"] + r:
            pos["sl"] = pos["entry"]
            pos["be_moved"] = True
            await safe_send(app, f"⚙️ #{coin} SL moved to breakeven (+1R reached)")
        elif pos["type"] == "SHORT" and live_low <= pos["entry"] - r:
            pos["sl"] = pos["entry"]
            pos["be_moved"] = True
            await safe_send(app, f"⚙️ #{coin} SL moved to breakeven (+1R reached)")


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
                if state.bot_active:
                    for coin, config in APEX_ROUTING.items():
                        df = await fetch_data(session, coin, config["tf"])
                        if df is None or len(df) < EMA_TREND_LEN + 30:
                            continue

                        # Manage existing position every poll (intra-bar exits)
                        if coin in state.open_positions:
                            await manage_position(app, state, coin, df)

                        # Entry only on a NEW closed bar
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
# TELEGRAM HANDLERS
# ==========================================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    welcome = (
        "🤖 <b>Algo Bot By FREAK</b>\n"
        "<i>Paper Trading Engine V2.0</i>\n\n"
        "Use the buttons below or commands: "
        "/status /balance /positions /pause /resume"
    )
    await update.message.reply_text(welcome, reply_markup=get_main_keyboard(),
                                    parse_mode=ParseMode.HTML)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    state: TradingState = context.application.bot_data["state"]
    await update.message.reply_text(build_status_text(state),
                                    reply_markup=get_main_keyboard(),
                                    parse_mode=ParseMode.HTML)


async def cmd_balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    state: TradingState = context.application.bot_data["state"]
    await update.message.reply_text(f"🏦 Balance: ${state.balance:.2f}")


async def cmd_positions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    state: TradingState = context.application.bot_data["state"]
    if not state.open_positions:
        await update.message.reply_text("No open positions.")
        return
    lines = []
    for coin, pos in state.open_positions.items():
        be = " (BE)" if pos.get("be_moved") else ""
        lines.append(
            f"#{coin} {pos['type']} E ${pos['entry']:.4f} "
            f"SL ${pos['sl']:.4f}{be} TP ${pos['tp']:.4f}"
        )
    await update.message.reply_text("\n".join(lines))


async def cmd_pause(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    state: TradingState = context.application.bot_data["state"]
    state.bot_active = False
    state.save(STATE_FILE)
    await update.message.reply_text("⏸️ Paused.")


async def cmd_resume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    state: TradingState = context.application.bot_data["state"]
    state.bot_active = True
    state.save(STATE_FILE)
    await update.message.reply_text("▶️ Active.")


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    state: TradingState = context.application.bot_data["state"]
    query = update.callback_query
    await query.answer()

    if query.data == "start_bot":
        state.bot_active = True
    elif query.data == "stop_bot":
        state.bot_active = False
    elif query.data == "reset_halt":
        state.daily_halted = False
        state.daily_start_balance = state.balance
    # status falls through to refresh

    state.save(STATE_FILE)
    await safe_edit(query, build_status_text(state), get_main_keyboard())


# ==========================================
# INITIALIZATION
# ==========================================
async def post_init(app: Application):
    state = TradingState.load(STATE_FILE)
    app.bot_data["state"] = state
    app.bot_data["scanner_task"] = asyncio.create_task(scanner_loop(app))
    await safe_send(
        app,
        f"🟢 <b>Bot online</b> — Balance ${state.balance:.2f} | "
        f"{'Active' if state.bot_active else 'Paused'}"
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
    app.add_handler(CommandHandler("pause", cmd_pause))
    app.add_handler(CommandHandler("resume", cmd_resume))
    app.add_handler(CallbackQueryHandler(button_handler))

    logger.info("Algo Bot By FREAK V2.0 starting…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

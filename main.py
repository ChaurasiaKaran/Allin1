import os 
import time
import asyncio
import requests
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ==========================================
# ⚙️ BOT CONFIGURATION
# ==========================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Paper Trading Ledger
VIRTUAL_BALANCE = 100.0
OPEN_POSITIONS = {}

# The Holy Trinity Routing
APEX_ROUTING = {
    "HYPE": {"tf": "15m", "strategy": "SWEEP_ONLY"},
    "TAO":  {"tf": "30m", "strategy": "HYBRID"},
    "ENA":  {"tf": "1h",  "strategy": "HYBRID"}
}

BASE_RISK_PCT = 0.03  
RR_RATIO = 3.0        
MIN_SL_PCT = 0.015    
TAKER_FEE = 0.00035   
MAKER_FEE = 0.00010   

bot_active = False  # Master switch controlled via Telegram

# ==========================================
# 📊 MARKET DATA & LOGIC ENGINE
# ==========================================
def fetch_data(coin, interval):
    url = "https://api.hyperliquid.xyz/info"
    # Fetch enough data for the 800 EMA to calculate properly
    start_time = int((datetime.now() - timedelta(days=20)).timestamp() * 1000)
    end_time = int(time.time() * 1000)
    
    payload = {"type": "candleSnapshot", "req": {"coin": coin, "interval": interval, "startTime": start_time, "endTime": end_time}}
    try:
        res = requests.post(url, json=payload).json()
        if not res: return None
        df = pd.DataFrame(res)
        df['datetime'] = pd.to_datetime(df['t'], unit='ms')
        for col in ['o', 'h', 'l', 'c', 'v']: df[col] = df[col].astype(float)
        return df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
    except:
        return None

def analyze_market(df, strategy_type):
    if len(df) < 50: return None
    
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    range_len = np.where((df['high'] - df['low']) == 0, 1, df['high'] - df['low'])
    df['buy_vol'] = df['volume'] * ((df['close'] - df['low']) / range_len)
    df['sell_vol'] = df['volume'] * ((df['high'] - df['close']) / range_len)
    df['delta'] = df['buy_vol'] - df['sell_vol']
    df['cvd'] = df['delta'].cumsum()
    df['cvd_roc'] = df['cvd'].diff(3) 
    df['cvd_bullish'] = df['cvd'] > df['cvd'].shift(3)

    ema_len = min(200, len(df)//3)
    if ema_len < 10: ema_len = 10
    df['ema_macro'] = ta.ema(df['close'], length=ema_len)
    df['trend_bullish'] = df['close'] > df['ema_macro']
    
    df['prev_swing_low'] = df['low'].rolling(window=20).min().shift(1)
    df['prev_swing_high'] = df['high'].rolling(window=20).max().shift(1)
    df['vol_ema'] = ta.ema(df['volume'], length=20) 
    df['vol_spike'] = df['volume'] > (df['vol_ema'] * 1.05) 
    
    df['sweep_long'] = (df['low'] < df['prev_swing_low']) & (df['close'] > df['prev_swing_low'])
    df['sweep_short'] = (df['high'] > df['prev_swing_high']) & (df['close'] < df['prev_swing_high'])

    sig_sweep_long = df['sweep_long'] & df['vol_spike'] & (df['cvd_roc'] > 0) & df['trend_bullish']
    sig_sweep_short = df['sweep_short'] & df['vol_spike'] & (df['cvd_roc'] < 0) & ~df['trend_bullish']

    period = 20
    df['sma_20'] = ta.sma(df['close'], length=period)
    df['std_dev'] = ta.stdev(df['close'], length=period)
    df['bb_upper'] = df['sma_20'] + (df['std_dev'] * 2.0)
    df['bb_lower'] = df['sma_20'] - (df['std_dev'] * 2.0)
    df['kc_upper'] = df['sma_20'] + (df['atr'] * 1.5)
    df['kc_lower'] = df['sma_20'] - (df['atr'] * 1.5)
    
    df['is_squeezed'] = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])
    df['squeeze_release'] = (~df['is_squeezed']) & (df['is_squeezed'].shift(1)) 
    df['price_break_up'] = df['close'] > df['kc_upper']
    df['price_break_down'] = df['close'] < df['kc_lower']

    sig_squeeze_long = df['squeeze_release'] & df['price_break_up'] & df['cvd_bullish']
    sig_squeeze_short = df['squeeze_release'] & df['price_break_down'] & ~df['cvd_bullish']

    if strategy_type == "SWEEP_ONLY":
        df['long_signal'] = sig_sweep_long
        df['short_signal'] = sig_sweep_short
    elif strategy_type == "HYBRID":
        df['long_signal'] = sig_sweep_long | sig_squeeze_long
        df['short_signal'] = sig_sweep_short | sig_squeeze_short

    return df.iloc[-2] # Return the last closed candle

# ==========================================
# 🤖 PAPER TRADING EXECUTION LOOP
# ==========================================
async def scanner_loop(app: Application):
    global VIRTUAL_BALANCE, OPEN_POSITIONS, bot_active
    
    while True:
        if bot_active:
            for coin, config in APEX_ROUTING.items():
                df = fetch_data(coin, config['tf'])
                if df is None: continue
                
                current_price = float(df.iloc[-1]['close'])
                last_closed = analyze_market(df, config['strategy'])
                
                # --- MANAGE OPEN POSITIONS ---
                if coin in OPEN_POSITIONS:
                    pos = OPEN_POSITIONS[coin]
                    pnl = 0
                    closed = False
                    reason = ""
                    
                    if pos['type'] == 'LONG':
                        if current_price >= pos['tp']:
                            pnl = (pos['tp'] - pos['entry']) * pos['size'] - (pos['tp'] * pos['size'] * MAKER_FEE)
                            closed, reason = True, "🎯 TAKE PROFIT HIT"
                        elif current_price <= pos['sl']:
                            pnl = (pos['sl'] - pos['entry']) * pos['size'] - (pos['sl'] * pos['size'] * TAKER_FEE)
                            closed, reason = True, "🛑 STOP LOSS HIT"
                    else: # SHORT
                        if current_price <= pos['tp']:
                            pnl = (pos['entry'] - pos['tp']) * pos['size'] - (pos['tp'] * pos['size'] * MAKER_FEE)
                            closed, reason = True, "🎯 TAKE PROFIT HIT"
                        elif current_price >= pos['sl']:
                            pnl = (pos['entry'] - pos['sl']) * pos['size'] - (pos['sl'] * pos['size'] * TAKER_FEE)
                            closed, reason = True, "🛑 STOP LOSS HIT"

                    if closed:
                        VIRTUAL_BALANCE += pnl
                        del OPEN_POSITIONS[coin]
                        msg = f"{reason}\n\nCoin: #{coin}\nPnL: ${pnl:.2f}\nNew Balance: ${VIRTUAL_BALANCE:.2f}"
                        await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
                
                # --- LOOK FOR NEW ENTRIES ---
                elif coin not in OPEN_POSITIONS:
                    if last_closed['long_signal']:
                        sl_dist = last_closed['atr'] * 2.5
                        if (sl_dist / current_price) >= MIN_SL_PCT:
                            size = (VIRTUAL_BALANCE * BASE_RISK_PCT) / sl_dist
                            OPEN_POSITIONS[coin] = {
                                'type': 'LONG', 'entry': current_price, 'size': size,
                                'sl': current_price - sl_dist, 'tp': current_price + (sl_dist * RR_RATIO)
                            }
                            VIRTUAL_BALANCE -= (current_price * size * TAKER_FEE)
                            
                            msg = f"🟢 **NEW LONG EXECUTED**\n\nCoin: #{coin}\nEntry: ${current_price:.4f}\nSL: ${OPEN_POSITIONS[coin]['sl']:.4f}\nTP: ${OPEN_POSITIONS[coin]['tp']:.4f}"
                            await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode='Markdown')

                    elif last_closed['short_signal']:
                        sl_dist = last_closed['atr'] * 2.5
                        if (sl_dist / current_price) >= MIN_SL_PCT:
                            size = (VIRTUAL_BALANCE * BASE_RISK_PCT) / sl_dist
                            OPEN_POSITIONS[coin] = {
                                'type': 'SHORT', 'entry': current_price, 'size': size,
                                'sl': current_price + sl_dist, 'tp': current_price - (sl_dist * RR_RATIO)
                            }
                            VIRTUAL_BALANCE -= (current_price * size * TAKER_FEE)
                            
                            msg = f"🔴 **NEW SHORT EXECUTED**\n\nCoin: #{coin}\nEntry: ${current_price:.4f}\nSL: ${OPEN_POSITIONS[coin]['sl']:.4f}\nTP: ${OPEN_POSITIONS[coin]['tp']:.4f}"
                            await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode='Markdown')

        # Sleep for 1 minute before checking the market again
        await asyncio.sleep(60)

# ==========================================
# 📱 TELEGRAM UI & COMMANDS
# ==========================================
def get_main_keyboard():
    keyboard = [
        [InlineKeyboardButton("📊 Status Check & Refresh", callback_data='status')],
        [InlineKeyboardButton("▶️ Start Trading", callback_data='start_bot'), 
         InlineKeyboardButton("⏸️ Pause Bot", callback_data='stop_bot')]
    ]
    return InlineKeyboardMarkup(keyboard)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = "🤖 **Algo Bot By FREAK**\n_Paper Trading Engine V1.0_\n\nSelect an option below to manage your institutional deployment."
    await update.message.reply_text(welcome_text, reply_markup=get_main_keyboard(), parse_mode='Markdown')

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global bot_active, VIRTUAL_BALANCE, OPEN_POSITIONS
    query = update.callback_query
    await query.answer()

    if query.data == 'start_bot':
        bot_active = True
        await query.edit_message_text(text="✅ **Scanner Active.** Hunting for Holy Trinity setups...", reply_markup=get_main_keyboard(), parse_mode='Markdown')
    
    elif query.data == 'stop_bot':
        bot_active = False
        await query.edit_message_text(text="⏸️ **Bot Paused.** No new trades will be taken.", reply_markup=get_main_keyboard(), parse_mode='Markdown')
    
    elif query.data == 'status':
        status_text = f"🏦 **Current Paper Balance:** ${VIRTUAL_BALANCE:.2f}\n"
        status_text += f"⚙️ **Status:** {'🟢 Active' if bot_active else '🔴 Paused'}\n\n"
        status_text += "📂 **Open Positions:**\n"
        
        if not OPEN_POSITIONS:
            status_text += "No active trades."
        else:
            for coin, pos in OPEN_POSITIONS.items():
                status_text += f"• #{coin} | {pos['type']} | Entry: ${pos['entry']:.2f}\n"

        # Update the existing message with the fresh status and keep the refresh button
        await query.edit_message_text(text=status_text, reply_markup=get_main_keyboard(), parse_mode='Markdown')

# ==========================================
# 🚀 INITIALIZATION
# ==========================================
async def main():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))

    # Start the background scanner loop
    asyncio.create_task(scanner_loop(app))
    
    print("[*] Algo Bot By FREAK is now online.")
    await app.run_polling()

if __name__ == '__main__':
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())

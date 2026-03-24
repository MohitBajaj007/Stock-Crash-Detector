import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
import pickle
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

save_folder = "model_c_files"
os.makedirs(save_folder, exist_ok=True)

with open(f"{save_folder}/model_c.pkl",   "wb") as f: pickle.dump(model_c,   f)
with open(f"{save_folder}/scaler.pkl",    "wb") as f: pickle.dump(scaler,    f)
with open(f"{save_folder}/hmm_model.pkl", "wb") as f: pickle.dump(hmm_model, f)

TICKERS = [
    "RELIANCE.NS", "TCS.NS",        "HDFCBANK.NS",  "INFY.NS",
    "ICICIBANK.NS","HINDUNILVR.NS", "ITC.NS",        "SBIN.NS",
    "BHARTIARTL.NS","KOTAKBANK.NS"
]

FEATURE_COLS = [
    "downside_dev_20","kurtosis_20","skewness_20",
    "momentum_5d","vol_10d","vol_20d","vol_ratio",
    "hmm_state_0","hmm_state_1","hmm_state_2"
]

with open("model_c.pkl",  "rb") as f: model_c  = pickle.load(f)
with open("scaler.pkl",   "rb") as f: scaler   = pickle.load(f)
with open("hmm_model.pkl","rb") as f: hmm_model= pickle.load(f)

end_date   = datetime.today()
start_date = end_date - timedelta(days=90)

print(f"Downloading data up to {end_date.strftime('%Y-%m-%d')}...")

raw    = yf.download(TICKERS, start=start_date, end=end_date, auto_adjust=True)
prices = raw["Close"].ffill()
returns= prices.pct_change().fillna(0)

nifty     = yf.download("^NSEI", start=start_date, end=end_date, auto_adjust=True)
nifty_ret = nifty["Close"].pct_change().fillna(0)
nifty_vals= nifty_ret.values.reshape(-1, 1)

hmm_probs = hmm_model.predict_proba(nifty_vals)
hmm_df    = pd.DataFrame(
    hmm_probs,
    index=nifty_ret.index,
    columns=["hmm_state_0","hmm_state_1","hmm_state_2"]
)

def compute_features(ret):
    df = pd.DataFrame(index=ret.index)
    neg = ret.copy(); neg[neg > 0] = 0
    df["downside_dev_20"] = neg.rolling(20).std()
    df["kurtosis_20"]     = ret.rolling(20).kurt()
    df["skewness_20"]     = ret.rolling(20).skew()
    df["momentum_5d"]     = ret.rolling(5).sum()
    df["vol_10d"]         = ret.rolling(10).std()
    df["vol_20d"]         = ret.rolling(20).std()
    df["vol_ratio"]       = df["vol_10d"] / (df["vol_20d"] + 1e-8)
    return df

scores = []
for ticker in TICKERS:
    if ticker not in prices.columns:
        continue
    feat = compute_features(returns[ticker].fillna(0))
    feat = feat.join(hmm_df, how="left")
    feat[["hmm_state_0","hmm_state_1","hmm_state_2"]] = \
        feat[["hmm_state_0","hmm_state_1","hmm_state_2"]].fillna(1/3)

    latest = feat[FEATURE_COLS].dropna().iloc[[-1]]
    if latest.empty: continue

    prob  = model_c.predict_proba(scaler.transform(latest))[0, 1]
    regime= hmm_df.iloc[-1]
    bear  = regime["hmm_state_2"]

    if   prob > 0.60: risk = "🔴 HIGH RISK   — Avoid / Exit"
    elif prob > 0.35: risk = "🟡 MEDIUM RISK — Reduce size"
    else:             risk = "🟢 LOW RISK    — Safe to hold"

    scores.append({
        "Ticker"     : ticker,
        "Crash Prob" : f"{prob*100:.1f}%",
        "Bear Regime": f"{bear*100:.1f}%",
        "Risk Level" : risk,
        "As of"      : end_date.strftime("%Y-%m-%d %H:%M")
    })

df_scores = pd.DataFrame(scores).sort_values("Crash Prob", ascending=False)
print("\n" + "="*65)
print(f"MODEL C — LIVE CRASH RISK SCORES — {end_date.strftime('%d %b %Y')}")
print("="*65)
print(df_scores.to_string(index=False))

df_scores.to_csv("live_crash_scores.csv", index=False)
print("\nSaved to live_crash_scores.csv")


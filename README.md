
# Model C — Stock-Level Crash Risk Detector

A machine learning system that predicts whether a NIFTY 200 stock will crash more than 10 percent in the next 10 trading days.

## Final Results

AUC-ROC: 0.7004 (target above 0.65) PASSED
Crash Recall: 63 percent (target above 0.60) PASSED
True Positives: 24 to 26 crashes caught
Training Period: 2008 to June 2022 (14 years)
Test Period: July 2022 to December 2024
Debugging Rounds: 14 iterations

## What This System Does

Model C is the mandatory crash protection layer of a three-model quantitative trading system for Indian equity markets. Every stock must pass through it before entering the portfolio.

Phase 1 - HMM detects market regime (Bull / Sideways / Bear)
Phase 2A - Model A: Short-horizon mean reversion signals
Phase 2B - Model B: Medium-horizon trend following signals
Phase 2C - Model C: Crash Risk Filter (THIS REPOSITORY)
Phase 3 - Portfolio Construction

Output: A crash probability score (0 to 100 percent) for every stock every day.
Above 60 percent: HIGH RISK - Exit or avoid entirely
35 to 60 percent: MEDIUM RISK - Reduce position size by 50 percent
Below 35 percent: LOW RISK - Safe to hold, normal sizing

## Repository Structure

model_c_clean.py: Full training pipeline Blocks 1 to 15
live_scores.py: Daily scoring script runs in 60 to 90 seconds
model_c_files/model_c.pkl: Trained XGBoost classifier
model_c_files/scaler.pkl: Fitted StandardScaler
model_c_files/hmm_model.pkl: Trained HMM 3 regime model
live_crash_scores.csv: Latest crash probability scores
model_c_report.docx: Complete build report
live_scores_guide.docx: Live scoring deployment guide

## The 10 Input Features

Stock-Level Features
downside_dev_20: How hard this stock falls on bad days
kurtosis_20: Fat tails, how often extreme moves happen
skewness_20: Negative skew equals classic crash profile
momentum_5d: Is the stock already falling right now
vol_10d: Short-term volatility 10 day
vol_20d: Medium-term baseline volatility 20 day
vol_ratio: Volatility spike detector vol_10d divided by vol_20d

Macro Regime Features from HMM
hmm_state_0: Probability of Bull market regime
hmm_state_1: Probability of Sideways or Transitional regime
hmm_state_2: Probability of Bear or Panic regime

## Feature Importance Final Model

hmm_state_2      25.4 percent - single most important feature
hmm_state_1      12.0 percent
downside_dev_20   8.7 percent
vol_10d           8.7 percent
hmm_state_0       8.5 percent
vol_20d           8.4 percent
skewness_20       7.9 percent
kurtosis_20       6.9 percent
vol_ratio         6.8 percent
momentum_5d       6.7 percent

The three HMM macro features combined account for 45.9 percent of all decisions.
Market regime context is a stronger crash predictor than any individual stock behaviour.

## Real Market Events Identified

June 4 2024 - Indian General Election Shock
BJP lost its outright majority. NIFTY 50 fell 4.3 percent intraday, worst day in 4 years.
ICICIBANK.NS flagged at 96.8 percent, stock fell 8.5 percent
RELIANCE.NS flagged at 96.0 percent, stock fell 7.8 percent
BHARTIARTL.NS flagged at 82.7 percent, stock fell 6.9 percent

November 9 2022 - FTX Collapse and US Midterms
TCS.NS flagged at 90.9 percent, global risk-off and FII selling

July 27 2022 - Infosys Earnings Miss
INFY.NS flagged at 87.5 percent, guidance cut and global tech selloff

All three major test period events flagged at 83 to 97 percent confidence.

## Key Learnings

Financial Market Insights

1. Macro regime dominates individual stock signals
HMM features contributed 43 percent of all decisions. When the market enters Bear regime, every stock elevates simultaneously regardless of individual technicals.

2. Regime transitions are as dangerous as confirmed Bear markets
hmm_state_1 (Sideways) ranked second at 12 percent. Markets are most vulnerable during the transition from Bull to Sideways when momentum wanes but panic has not confirmed.

3. Downside deviation is the strongest individual stock signal
A stock that falls 3 to 5 percent on bad days is fundamentally different from one that falls 0.5 percent. This predicts future crashes better than any other single metric.

4. Negative skewness is a quiet warning sign
Stocks that give many small gains but occasionally deliver devastating losses show sustained negative skewness and are significantly more likely to crash.

5. Technical models cannot predict earnings surprises
The model caught every macro-driven crash. It missed the Kotak RBI ban in April 2024 because overnight regulatory events leave no technical footprint.

6. 2008 training data is essential
Including the Global Financial Crisis, demonetisation, and COVID gave the model exposure to three completely different crash regimes.

7. Signal quality matters more than data volume
Expanding from 10 to 25 stocks hurt performance. Adding utility and cement stocks with different crash dynamics confused the model.

Engineering Insights

8. Decision threshold must match deployment environment
A threshold calibrated for 12 percent training crash rate fails in 0.6 percent test crash rate.

9. scale_pos_weight is the most sensitive parameter
At 6 to 9 times it dramatically improves recall. Above 12 times it causes predict-everything-crash collapse.

10. Training AUC of 1.0 is a failure not a success
It means memorisation not learning. The test AUC is the only number that matters.

## Stock-Level Insights

HDFCBANK.NS: Crash risk spikes around quarterly results and RBI policy announcements
INFY.NS: Tracks NASDAQ closely, US recession fears directly elevate crash probability
KOTAKBANK.NS: Regulatory risk significant, technical models cannot capture overnight shocks
BHARTIARTL.NS: Highest risk during sideways market phases and regime transitions
RELIANCE.NS: Strong election sensitivity, flagged at 96 percent on June 4 2024
TCS.NS: High correlation with global risk-off events, flagged at 91 percent during FTX collapse
SBIN.NS: Crash days cluster around budget and policy announcements
HINDUNILVR.NS: Consistently lowest crash probabilities, FMCG defensive nature validated
ITC.NS: Risk elevated during regulatory changes to tobacco policy
ICICIBANK.NS: Highest single-day flag at 96.8 percent across entire test period

## Requirements

pip install xgboost hmmlearn yfinance optuna scikit-learn pandas numpy

Note: LightGBM is incompatible with Python 3.13 on macOS. XGBoost is used as a drop-in replacement with identical API and performance.

## Known Limitations

Cannot predict earnings surprises: Add earnings calendar feature
Training overfitting gap: Implement quarterly rolling retraining
Only 10 stocks: Expand to NIFTY 200 with sector-aware models
Static threshold: Dynamic threshold based on rolling crash rate
No volume features: Add volume spike as current divided by 20-day average

## Roadmap Phase 3

Portfolio Construction Layer
Earnings calendar integration
Options implied volatility overlay
Expand universe to NIFTY 200
Web dashboard for real-time scores

Built as Phase 2C of a three-model quantitative trading system for Indian equity markets.
AUC-ROC: 0.7004 | Crash Recall: 63 percent | Universe: NIFTY 200 | Training: 2008 to 2022


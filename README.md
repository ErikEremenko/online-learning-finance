# Online Learning for Financial Time Series
**TUM Seminar — Online and Continual Machine Learning (WS 2025/26)**
Erik Eremenko & Sam Müller

## Overview
Survey and empirical evaluation of online learning algorithms applied to
S&P 500 time series. Compares SNARIMAX, ARIMA, and classification-based
approaches across three market regimes (2008 crisis, 2015–18, COVID-19).

## Key Results
| Model | Sharpe (2008) | Sharpe (2015–18) |
|-------|--------------|-----------------|
| Online SNARIMAX | **0.89** | 0.61 |
| Batch ARIMA | -0.12 | 0.54 |

## Run
pip install -r requirements.txt
python financial_models.py

## Paper
[OL_CAML_TUM_Eremenko_Muller.pdf](./paper.pdf)

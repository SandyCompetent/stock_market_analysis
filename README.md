# Stock Market Analysis in Python

[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html)

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)\
[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)

The stock market serves as a barometer for a country’s economic well-being, with share prices
influenced by elements like investor sentiment, market trends, and external factors. This project
investigates the potential of machine learning (ML) to predict stock price fluctuations. By analysing
historical data from the NIFTY 50 index (2007–2024), the study examines two ML models— **Linear
Regression** and **Long Short-Term Memory (LSTM)** —to determine their effectiveness in forecasting
market trends and hence forth adding more ideas to it.

## Requirement

Project depends on [Yahoo Finance](https://finance.yahoo.com/) to collect historical data of stock market.

```bash
pip install numpy
pip install pandas
pip install yfinance
pip install keras
pip install scikit-learn
pip install matplotlib
```

## Result

- Plot of prediction using Linear Regression

![screenshot](result/nifty_50_result_linear_regression.png)

- Plot of prediction using LSTM

![screenshot](result/nifty_50_result_lstm.png)

# ARIMA Time Series Forecasting from Scratch  

## Overview  

This project implements the **ARIMA (AutoRegressive Integrated Moving Average) model** from scratch in **Julia** to forecast minimum daily temperatures recorded in a city over time. The model includes:  

- **ADF (Augmented Dickey-Fuller) Test** to determine the differencing order (**d**) by checking for stationarity.  
- **Exponential Smoothing** to handle outliers in the dataset.  
- **AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion)** to determine the optimal values of **p** (autoregressive order) and **q** (moving average order).

## Useful Info
- `src` folder contains a module which implements ARIMA.
- `finder.jl` uses that module on the data.
  

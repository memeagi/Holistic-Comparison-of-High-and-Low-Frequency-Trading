WQU Capstone Project
Holistic Comparison of High and Low Frequency Trading
======================

This project implements a simple moving average cross strategy on some selected assets (Gold ETF, Apple Stock and EURUSD)

SETUP
=====

 1. Install Python 2.7 or later. From anaconda.org download python 2.7 or a later version.
    Right click on the downloaded file and click run. Follow all instruction to complete installation.  


 2. From spyder ide, import the following python packages as shown below:
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import scorer
    from sklearn.metrics import accuracy_score
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas_datareader.data as web


STRATEGY
==============
The simulation is in two parts but with the same strategy:

Part One

This simulation downloads daily historical prices of Gold ETF and Apple prices from yahoo and resampling
technique used to convert the daily prices to minutes.The resampled data is later backtested with a simple moving average average strategy. 
The second part downloaded hourly historical prices
of EURUSD from www.dukascopy.com. The hourly prices are resampled to one minutes prices and it is backtested on
the same moving average cross strategy.

Part Two

This simulation downloads daily historical prices of Gold ETF and Apple prices from and
backtest a simple moving average average strategy. The second part downloaded hourly historical prices
of EURUSD from www.dukascopy.com. The hourly prices are resampled to daily prices and it is backtested on
the same moving average cross strategy.


ABOUT THE CODE
==============
Ideally, this code is meant to implement a simple moving average on the sets of data of same asset classes and period and compute the following
performance metrics Sharpe Ratio, Cumulative Anual Growth Rate, Mean Return etc to compare results.


CONTACT
=======

Please send bug reports, patches, and other feedback to michealemeagi@gmail.com

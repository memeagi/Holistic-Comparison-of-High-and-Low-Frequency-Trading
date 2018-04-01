# Importing Necessary library for the project
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import scorer
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web

def strategy(assets):
    i=0
    for i in range(len(assets)):
        data= web.DataReader(assets[i], 'yahoo','2010-01-01')
        data['ma_30'] = data.Close.rolling(window=30).mean()
        data['ma_10'] = data.Close.rolling(window=10).mean()
        data['signal'] = np.where((data['ma_10']>data['ma_30']) , 1, 0)
        data['signal2'] = np.where((data['ma_10']<data['ma_30']) , -1,data['signal'] )
        data['Return'] = data['Close'].pct_change()
        print '                                           '
        print '...........................................'
        print 'The Result of ' +assets[i]+ ' is shown below:'
        print '..........................................'
        data = data.dropna()
        y = data['signal2']

        X=data[['ma_30','ma_10','Close','Volume','High','Low']]
        # Split data into Training and Test sets
        split_percentage = 0.75
        split = int(split_percentage*len(data))
        # Train data set
        X_train = X[:split]
        y_train = y[:split]
        # Test data set
        X_test = X[split:]
        y_test = y[split:]
        clf_random_f = RandomForestClassifier(n_jobs=2, random_state=0)
        clf_random_f.fit(X_train, y_train)

        accuracy_train_rf = accuracy_score(y_train, clf_random_f.predict(X_train))
        accuracy_test_rf = accuracy_score(y_test, clf_random_f.predict(X_test))
        print('\nTrain Accuracy:{: .2f}%'.format(accuracy_train_rf*100))
        print('Test Accuracy:{: .2f}%'.format(accuracy_test_rf*100))
        data['Pred_Sig_Random_F'] = clf_random_f.predict(X)
        data['Strategy_Return_RF'] = data.Return * data.Pred_Sig_Random_F

        
        
        #Backtesting
        # Set the initial capital
        initial_capital= float(100000.0)

        # Create a DataFrame `positions`
        positions = pd.DataFrame(index=data.index).fillna(0.0)

        # Buy a 100 shares
        positions['Stock'] = 100*data['signal2']   
  
        # Initialize the portfolio with value owned   
        portfolio = positions.multiply(data['Close'], axis=0)

        # Store the difference in shares owned 
        pos_diff = positions.diff()

        # Add `holdings` to portfolio
        portfolio['holdings'] = (positions.multiply(data['Close'], axis=0)).sum(axis=1)

        # Add `cash` to portfolio
        portfolio['cash'] = initial_capital - (pos_diff.multiply(data['Close'], axis=0)).sum(axis=1).cumsum() - 0.03*(pos_diff.multiply(data['Close'], axis=0)).sum(axis=1).cumsum()  

        # Add `total` to portfolio
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']

        # Add `returns` to portfolio
        portfolio['returns'] = portfolio['total'].pct_change()

        # Print the last lines of `portfolio`
        print 'Total money returned by the LFT backtest is ',portfolio.iloc[-1]['total'] 
        print portfolio
        mean_ret = np.mean(portfolio['returns'])
        print 'Mean Return is ', mean_ret

        #calcualte CAGR
        cagr = ((float(portfolio.iloc[-1]['total'])/initial_capital)**0.125)-1 # Calculating CAGR over  8 years period
        print 'CAGR is ', cagr
        # Calculate Sharpe Ratio
        sr =  (mean_ret- 0.01)/np.std(portfolio['returns']) #Assume a risk free rate of 1%
        print 'Sharpe Ratio is', sr
        #Calculate Gain to Pain Ratio
        gpr = abs(np.sum(portfolio['returns'])/sum(n < 0 for n in portfolio['returns']))
        print 'Gain to Pain Ratio is ',gpr
        #Plotting Portfolio Returns and identifying peak periods showing drawdown
        i = np.argmax(np.maximum.accumulate(portfolio['total']) - portfolio['total']) # end of the period
        j = np.argmax(portfolio['total'][:i]) # start of period

        plt.plot(portfolio['total'])
        plt.title('Asset showing Drawdowns')
        plt.ylabel('Amount')
        plt.xlabel('Date')
        plt.plot([i, j], [portfolio['total'][i], portfolio['total'][j]], 'o', color='Red', markersize=10)
    return i

assets = ['AAPL','SPY']
strategy(assets)

df = pd.read_csv("C:\Users\user\Downloads\EURUSD_hours.csv")# importing data from computer path
df.columns = [['date','open','high','low','close','volume']]# setting the columes to open, high,low,close,volumn
df.date = pd.to_datetime(df.date,format='%d.%m.%Y %H:%M:%S.%f')#Resetting the date format
df = df.set_index(df.date)
df = df[['open','high','low','close','volume']]
data = df.drop_duplicates(keep=False)
data = df.resample('D').bfill()
data['ma_30'] = data.close.rolling(window=30).mean()
data['ma_10'] = data.close.rolling(window=10).mean()
data['signal'] = np.where((data['ma_10']>data['ma_30']) , 1, 0)
data['signal2'] = np.where((data['ma_10']<data['ma_30']) , -1,data['signal'] )
data['Return'] = data['close'].pct_change()
print '                                           '
print '...........................................'
print 'The Result of EURUSD is shown below:'
print '..........................................'


data = data.dropna()
y = data['signal2']

X=data[['ma_30','ma_10','close','volume','high','low']]
# Split data into Training and Test sets
split_percentage = 0.75
split = int(split_percentage*len(data))
# Train data set
X_train = X[:split]
y_train = y[:split]
# Test data set
X_test = X[split:]
y_test = y[split:]
clf_random_f = RandomForestClassifier(n_jobs=2, random_state=0)
clf_random_f.fit(X_train, y_train)

accuracy_train_rf = accuracy_score(y_train, clf_random_f.predict(X_train))
accuracy_test_rf = accuracy_score(y_test, clf_random_f.predict(X_test))
print('\nTrain Accuracy:{: .2f}%'.format(accuracy_train_rf*100))
print('Test Accuracy:{: .2f}%'.format(accuracy_test_rf*100))
data['Pred_Sig_Random_F'] = clf_random_f.predict(X)
data['Strategy_Return_RF'] = data.Return * data.Pred_Sig_Random_F

        
        
#Backtesting
# Set the initial capital
initial_capital= float(100000.0)

# Create a DataFrame `positions`
positions = pd.DataFrame(index=data.index).fillna(0.0)

# Buy a 100 units of EURUSD
positions['Stock'] = 100*data['signal2']   
# Initialize the portfolio with value owned   
portfolio = positions.multiply(data['close'], axis=0)

# Store the difference in shares owned 
pos_diff = positions.diff()

# Add `holdings` to portfolio
portfolio['holdings'] = (positions.multiply(data['close'], axis=0)).sum(axis=1)

# Add `cash` to portfolio
portfolio['cash'] = initial_capital - (pos_diff.multiply(data['close'], axis=0)).sum(axis=1).cumsum() - 0.03*(pos_diff.multiply(data['close'], axis=0)).sum(axis=1).cumsum()  

# Add `total` to portfolio
portfolio['total'] = portfolio['cash'] + portfolio['holdings']

# Add `returns` to portfolio
portfolio['returns'] = portfolio['total'].pct_change()

# Print the last lines of `portfolio`
print 'Total money returned by the LFT backtest is ',portfolio.iloc[-1]['total'] 
print portfolio
mean_ret = np.mean(portfolio['returns'])
print 'Mean Return is ', mean_ret

#calcualte CAGR
cagr = ((float(portfolio.iloc[-1]['total'])/initial_capital)**0.2) -1 #Calculating the CAGR over 5 years period
print 'CAGR is ', cagr
# Calculate Sharpe Ratio
sr =  (mean_ret- 0.01)/np.std(portfolio['returns']) #Assume a risk free rate of 1%
print 'Sharpe Ratio is', sr
#Calculate Gain to Pain Ratio
gpr = abs(np.sum(portfolio['returns'])/sum(n < 0 for n in portfolio['returns']))
print 'Gain to Pain Ratio is ',gpr
# Plotting Portfolio Returns and identifying peak periods showing drawdown
i = np.argmax(np.maximum.accumulate(portfolio['total']) - portfolio['total']) # end of the period
j = np.argmax(portfolio['total'][:i]) # start of period

plt.plot(portfolio['total'])
plt.title('EURUSD showing Drawdowns')
plt.ylabel('Amount')
plt.xlabel('Date')
plt.plot([i, j], [portfolio['total'][i], portfolio['total'][j]], 'o', color='Red', markersize=10)

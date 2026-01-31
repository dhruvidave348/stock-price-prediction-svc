#Machine Learning
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#Data manipulation
import pandas as pd
import numpy as np

#to plot
import matplotlib.pyplot as plt

#to ignore warnings
import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("All modules imported successfully")


#read data
#method of pandas

df=pd.read_csv('RELIANCE.csv')   #df is Data Frame its like an excel sheet inside pandas
logging.info("Data read successfully")

#choose column as index

df.index=pd.to_datetime(df['Date'])

#drop the original date column
df=df.drop(['Date'],axis='columns')
logging.info("Date column set as index successfully")

#creating predictive features
df['Open-Close']=df.Open-df.Close
df['High-Low']=df.High-df.Low

X=df[['Open-Close','High-Low']]
logging.info("Predictive features created successfully")    

#defining target feature
# will store +1 for a buy signal and 0 for a no position in y
y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# drop NaN values created by shift and pct_change
df = df.dropna()
X = X.loc[df.index]
y = y[:len(df)]

# split into train and test
split_percentage=0.8
split=int(split_percentage*len(df))

#train data
X_train=X[:split]
y_train=y[:split]

#test data
X_test=X[split:]
y_test=y[split:]

#support vector classifier using fit() method on training dataset

cls=SVC().fit(X_train,y_train)
logging.info("Model trained successfully")

#classifier accuracy

#calc training accuracy
train_acc=accuracy_score(y_train,cls.predict(X_train))
logging.info(f"Training Accuracy: {train_acc*100:.2f}%")

#calc testing accuracy
test_acc=accuracy_score(y_test,cls.predict(X_test))
logging.info(f"Testing Accuracy: {test_acc*100:.2f}%")

df['Predicted_Signal'] = cls.predict(X)

# Calculate daily returns
df['Return'] = df.Close.pct_change()

# Calculate strategy returns
df['Strategy_Return'] = df.Return * df.Predicted_Signal.shift(1)

# Calculate Cumulutive returns
df['Cum_Ret'] = df['Return'].cumsum()

# Plot Strategy Cumulative returns 
df['Cum_Strategy'] = df['Strategy_Return'].cumsum()

plt.plot(df['Cum_Ret'],color='red',label='Market Returns')
plt.plot(df['Cum_Strategy'],color='blue',label='Strategy Returns')
plt.legend()
plt.show()
logging.info("Cumulative returns plotted successfully")
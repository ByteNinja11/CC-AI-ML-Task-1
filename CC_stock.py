import pandas as pd
import yfinance as yf
import sklearn

# Getting Tata Consultancy Service (TCS) stock data from BSE
tcs = yf.Ticker("TCS.BO")
tcs = tcs.history(period="max")
del tcs["Dividends"]
del tcs["Stock Splits"]

# Setting a target
tcs["Tomorrow"] = tcs["Close"].shift(-1)
tcs["Target"] = (tcs["Tomorrow"] > tcs["Close"]).astype(int)

# Training the model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=190, min_samples_split=40, random_state=1)

# Setting train and test set
train = tcs.iloc[:-100]
test = tcs.iloc[-100:]

predictors = ["Close", "Volume", "Open", "High", "Low"]

model.fit(train[predictors], train["Target"])

# Measuring model accuracy
from sklearn.metrics import precision_score

preds = model.predict(test[predictors])

# Converting preds array to series
preds = pd.Series(preds, index=test.index)

print(precision_score(test["Target"], preds))

# Improving
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >=0.6] = 1
    preds[preds < 0.6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


# Backtest function (Each year has 250 trading days)
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i: (i + step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)


predictions = backtest(tcs, model, predictors)
print(predictions["Predictions"].value_counts())

# Accuracy of model in predicting an uptrend in the market
print(100*(float(precision_score(predictions["Target"], predictions["Predictions"]))))

# When closing price actually increased
print(predictions["Target"].value_counts() / predictions.shape[0])







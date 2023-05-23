# Import libraries
import yfinance as yf
import ta
from datetime import timedelta, date
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def add_all_TA(stock_df):
    stock_df = ta.utils.dropna(stock_df)
    stock_df = ta.add_all_ta_features(stock_df, "Open", "High", "Low", "Close", "Volume", fillna=True)
    
    return stock_df


def get_clean_stock_data(ticker, start_date="2020-01-01", end_date = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')):
    return yf.download(ticker, start=start_date, end=end_date, progress=False)


def get_full_stock_data(ticker, start_date="2020-01-01", end_date = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')):
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return add_all_TA(df)


def check_model_for_new_ticker(model, ticker):

    full_stock_data = get_full_stock_data(ticker=ticker, start_date="2020-01-01", end_date="2022-12-31")

    # Remove the direct price data from the dataset we will train on.
    X = full_stock_data.drop(['Open', 'High', 'Low', 'Close', 'Adj Close', 'others_dr', 'others_dlr','others_cr'], axis=1)

    # Set the close price as our "goal" dataset. E.g, use X to try and predict y
    y = full_stock_data['Close']

    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=True)

    fig, ax = plt.subplots(figsize=(12,6))

    ax.plot(y.index, y.values, label='Actual', c='#1f77b4', zorder=-5)
    ax.plot(y.index, y_pred, label='Predicted', ls='--', c='#ff7f0e')

    ax.legend()
    ax.set_title("{}".format(ticker))


    ax.text(0.05, 1.05, r'R$^2$' + ': {:,.2f}, RMSE: {:,.2f}'.format(r2, rmse), transform=ax.transAxes, horizontalalignment='center', verticalalignment='top')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.grid()
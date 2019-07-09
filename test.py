import pandas as pd, numpy as np
import datetime as dt

minute = pd.read_csv('data/eurusdhist/eurusd_minute.csv',
                     usecols=['Date','Minute',
                              'BidOpen','BidHigh','BidLow','BidClose','BidChange',
                              'AskOpen','AskHigh','AskLow','AskClose','AskChange'],
                     nrows=100000)
hour = pd.read_csv('data/eurusdhist/eurusd_hour.csv',
                   usecols=['Date','Hour',
                            'BidOpen','BidHigh','BidLow','BidClose','BidChange',
                            'AskOpen','AskHigh','AskLow','AskClose','AskChange'],
                   nrows=10000)


hour.Hour=hour.Hour.apply(lambda x: dt.timedelta(hours=int(x[:2])))

minute.insert(1,column='Hour',value=minute.Minute)
minute.Hour=minute.Hour.apply(lambda x: dt.timedelta(minutes=x))

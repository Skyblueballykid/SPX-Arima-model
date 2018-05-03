import numpy as np
import pandas as pd

SPX = pd.read_csv("SPX Weekly 2000-2017.csv")

print(SPX.head(n=5))

model = ARIMA(SPX, order=(0,1,0))
model_fit = model.fit()

outcome = model_fit.forecast()[0]

print(model.head(n=5))

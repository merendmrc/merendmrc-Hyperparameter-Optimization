import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from sklearn.ensemble import RandomForestRegressor

data_bank = pd.read_csv("banka/bank.csv")



normalize = MinMaxScaler(feature_range=(0,1))
norm_data = pd.DataFrame(normalize.fit_transform(data_bank),columns=data_bank.columns)

X = norm_data.values[:,:8]
Y = norm_data.values[:,8:]

xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2,shuffle=True,stratify=Y,random_state=2)

params = ((50,150,200,250),('absolute_error','friedman_mse')) #Select params to
results = []
for i in params[0]:
    for j in params[1]:
        model = RandomForestRegressor(n_estimators=i, criterion=j)
        model.fit(xtrain,ytrain)

        predictions = model.predict(xtest)

        results.append(f"""
n_estimators = {i}
criterion = {j}
MAE= {round(mean_absolute_error(ytest, predictions),4)}
MSE= {round(mean_squared_error(ytest, predictions),4)}
RMSE= {round(sqrt(mean_squared_error(ytest, predictions)),4)}
""")



for i in results:
    print(i)
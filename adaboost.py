import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
data_bank = pd.read_csv("banka/bankakredi.csv")



normalize = MinMaxScaler(feature_range=(0,1))
norm_data = pd.DataFrame(normalize.fit_transform(data_bank),columns=data_bank.columns)

X = norm_data.values[:,:8]
Y = norm_data.values[:,8:]

xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2,shuffle=True,stratify=Y,random_state=2)

ytrain = ytrain.ravel()
ytest = ytest.ravel()


model = AdaBoostRegressor(base_estimator=SVR())
model.fit(xtrain,ytrain)
predictions = model.predict(xtest)

print(f"""
MAE= {round(mean_absolute_error(ytest, predictions),4)}
MSE= {round(mean_squared_error(ytest, predictions),4)}
RMSE= {round(sqrt(mean_squared_error(ytest, predictions)),4)}
""")

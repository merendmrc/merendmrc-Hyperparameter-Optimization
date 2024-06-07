from sklearn.ensemble import VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data_bank = pd.read_csv("banka/bank.csv")

normalize = MinMaxScaler(feature_range=(0,1))
norm_data = pd.DataFrame(normalize.fit_transform(data_bank),columns=data_bank.columns)

X = norm_data.values[:,:8]
Y = norm_data.values[:,8:]

xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2,shuffle=True,stratify=Y,random_state=2)

knn = KNeighborsRegressor()
dt = DecisionTreeRegressor()
svr = SVR()

model = VotingRegressor([('knn', knn), ('dt', dt), ('svr', svr)])

model.fit(xtrain, ytrain)

predictions = model.predict(xtest)

print(f"""
MAE= {round(mean_absolute_error(ytest, predictions),4)}
MSE= {round(mean_squared_error(ytest, predictions),4)}
RMSE= {round(sqrt(mean_squared_error(ytest, predictions)),4)}
""")
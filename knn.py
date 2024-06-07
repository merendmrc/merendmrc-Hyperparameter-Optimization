import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from sklearn.neighbors import KNeighborsRegressor


data_bank = pd.read_csv("banka/bank.csv")



normalize = MinMaxScaler(feature_range=(0,1))
norm_data = pd.DataFrame(normalize.fit_transform(data_bank),columns=data_bank.columns)

X = norm_data.values[:,:8]
Y = norm_data.values[:,8:]

xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2,shuffle=True,stratify=Y,random_state=2,)

params = ((1,),('ball_tree','kd_tree','brute')) # Select parameters to test
results = []
for i in params[0]:
    for j in params[1]:
        model = KNeighborsRegressor(weights='distance' ,n_neighbors=40,algorithm=j)
        model.fit(xtrain,ytrain)

        predictions = model.predict(xtest)

        results.append(f"""
Weight = {i}
n_neighbors = {j}
MAE= {round(mean_absolute_error(ytest, predictions),4)}
MSE= {round(mean_squared_error(ytest, predictions),4)}
RMSE= {round(sqrt(mean_squared_error(ytest, predictions)),4)}
""")


for i in results:
    print(i)
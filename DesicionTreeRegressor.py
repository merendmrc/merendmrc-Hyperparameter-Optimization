import pandas as pd
from sklearn.tree import DecisionTreeRegressor #model import
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error #import loss functions 
from math import sqrt


data_bank = pd.read_csv("banka/bank.csv")

normalize = MinMaxScaler(feature_range=(0,1))
norm_data = pd.DataFrame(normalize.fit_transform(data_bank),columns=data_bank.columns)

X = norm_data.values[:,:8]
Y = norm_data.values[:,8:]

xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2,shuffle=True,stratify=Y,random_state=2,)

params = ((50,100,150),(3,)) #Select parameters to test (max leaf nodes, max depth)
results = []

for i in params[0]:
    for j in params[1]:
        model = DecisionTreeRegressor(max_leaf_nodes= i,max_depth=j)
        model.fit(xtrain,ytrain)

        predicts = model.predict(xtest)

        results.append(f"""
Max leaf nodes = {i}
Max depth = {j}
MAE= {round(mean_absolute_error(ytest, predicts),4)}
MSE= {round(mean_squared_error(ytest, predicts),4)}
RMSE= {round(sqrt(mean_squared_error(ytest, predicts)),4)}
""")

for i in results:
    print(i)
import numpy as np
import pandas as pd

RANDOM_STATE = 17
np.random.seed(RANDOM_STATE)

def split(df):

    from sklearn.model_selection import train_test_split
    x_train , x_val , y_train , y_val = train_test_split(df.drop('Target',axis=1),df['Target'],test_size=0.5,random_state=RANDOM_STATE,shuffle=False)

    return x_train , x_val , y_train , y_val

def preprocessing(data, option = 1):

    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    if option == 1:
        processor = MinMaxScaler()
    elif option == 2:
        processor = StandardScaler()
    else:
        return data, None

    return processor.fit_transform(data)

def LinearRegression(x,y,intercept):

    from sklearn.linear_model import LinearRegression
    model = LinearRegression(fit_intercept=intercept)
    model.fit(x,y)

    return model

def model_prediction(model,x):
    pred = model.predict(x)
    return pred

def evaluate(x,y):

    from sklearn.metrics import mean_squared_error
    MSE = mean_squared_error(x,y)
    RMSE = np.sqrt(MSE)

    return RMSE

if __name__ == '__main__':

    df = pd.read_csv("C:\\Users\\MH\\PycharmProjects\\task 1\data2_200x30.csv")

    x_train, x_val, y_train, y_val = split(df)

    # data = df.to_numpy()
    # x = data[:, :-1]
    # y = data[:, -1]
    #
    # x_train, x_val = x[:100], x[100:200]
    # y_train, y_val = y[:100], y[100:200]

    x_train = preprocessing(x_train, 1)
    x_val = preprocessing(x_val, 1)

    model = LinearRegression(x_train, y_train, True)

    y_train_predict = model_prediction(model, x_train)
    RMSE_train = evaluate(y_train, y_train_predict)

    y_val_predict = model_prediction(model, x_val)
    RMSE_val = evaluate(y_val, y_val_predict)

    avg_weight = abs(model.coef_).mean()

    print(f'intercept : {model.intercept_}')
    print(f'avg weight : {avg_weight}')
    print(f'train error: {RMSE_train}')
    print(f'val error : {RMSE_val}')






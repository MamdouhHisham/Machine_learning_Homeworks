import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def evaluate(x, y):
    from sklearn.metrics import mean_squared_error
    MSE = mean_squared_error(x, y)
    RMSE = np.sqrt(MSE)
    return RMSE


def try_polynomial(x_train, y_train, x_val, y_val, degrees):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    train_errors, val_errors = [], []

    for degree in degrees:

        poly = PolynomialFeatures(degree=degree,include_bias=False,interaction_only=False)

        x_poly = poly.fit_transform(x_train)
        x_val_poly = poly.fit_transform(x_val)
        x_poly = scaler.fit_transform(x_poly)
        x_val_poly = scaler.transform(x_val_poly)

        reg = LinearRegression()
        reg.fit(x_poly, y_train)

        y_train_predict = reg.predict(x_poly)
        train_rmse = evaluate(y_train,y_train_predict)
        train_errors.append(train_rmse)

        y_val_predict = reg.predict(x_val_poly)
        val_rmse = evaluate(y_val,y_val_predict)
        val_errors.append(val_rmse)

        print(f'Degree : {degree} --> ,Train Error : {train_rmse} , Val Error : {val_rmse} ,  Intercept : {reg.intercept_}')

    return train_errors, val_errors


if __name__ == '__main__':
    df = pd.read_csv("C:\\Users\\MH\\PycharmProjects\\task2\\data2_200x30.csv")

    data = df.to_numpy()
    x = data[:, :-1]
    y = data[:, -1]

    x_train, x_val = x[:100], x[100:201]
    y_train, y_val = y[:100], y[100:201]

    degrees = [1, 2, 3, 4]
    train_errors, val_errors = try_polynomial(x_train, y_train, x_val, y_val, degrees)

    plt.plot(degrees, train_errors, label='Train Error')
    plt.plot(degrees, val_errors, label='Validation Error')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('RMSE')
    plt.xticks(degrees)
    plt.legend()
    plt.show()




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def evaluate(x, y):
    from sklearn.metrics import mean_squared_error

    MSE = mean_squared_error(x, y)
    RMSE = np.sqrt(MSE)

    return RMSE

def try_polynomial(x,y,degrees,feature_ids):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    results = {}

    for degree in degrees:
        degree_results = {}

        for idx in feature_ids:
            x_feature = x[:,idx].reshape(-1,1)

            x_train, x_val, y_train, y_val = train_test_split(x_feature, y, test_size=0.5, random_state=17,shuffle=False)

            poly = PolynomialFeatures(degree=degree)
            x_train_poly = poly.fit_transform(x_train)
            x_val_poly = poly.transform(x_val)

            reg = LinearRegression()
            reg.fit(x_train_poly,y_train)

            y_train_pred = reg.predict(x_train_poly)
            y_val_pred = reg.predict(x_val_poly)

            train_error = evaluate(y_train,y_train_pred)
            val_error = evaluate(y_val,y_val_pred)

            degree_results[idx] = (train_error,val_error)

        results[degree] = degree_results

    return results

def plot_errors(results,feature_ids):

    degrees = results.keys()
    for degree in degrees:

        train_errors = [results[degree][idx][0] for idx in feature_ids]
        val_errors = [results[degree][idx][1] for idx in feature_ids]

        idx = np.arange(len(feature_ids))

        plt.figure(figsize=(12, 6))
        plt.bar(idx, train_errors, .2 ,label='Train Error')
        plt.bar(idx + .2, val_errors,.2 ,label='Validation Error')

        plt.xlabel('Feature Index')
        plt.ylabel('Error')
        plt.title(f'Polynomial Degree {degree} Errors')
        plt.xticks(idx , feature_ids)
        plt.legend()
        plt.show()


if __name__ == '__main__':

    df = pd.read_csv("C:\\Users\\MH\\PycharmProjects\\task2\\data2_200x30.csv")

    data = df.to_numpy()
    x = data[:, :-1]
    y = data[:, -1]

    features_ids = [0, 3, 6]
    degrees = [1, 2, 3]

    res = try_polynomial(x,y,degrees,features_ids)
    # print(res)
    plot_errors(res,features_ids)




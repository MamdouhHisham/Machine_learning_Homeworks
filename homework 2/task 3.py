import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def monomials_poly_features(X, degree):

    examples = []
    for example in X:
        example_features = []
        for feature in example:
            cur = 1
            feats = []
            for deg in range(degree):
                cur *= feature
                feats.append(cur)
            example_features.extend(feats)
        examples.append(np.array(example_features))

    return np.vstack(examples)

if __name__ == '__main__':
    df = pd.read_csv("C:\\Users\\MH\\PycharmProjects\\task2\\data2_200x30.csv")

    data = df.to_numpy()
    x = data[:, :-1]
    y = data[:, -1]

    x_train, x_val = x[:100], x[100:201]
    y_train, y_val = y[:100], y[100:201]

    features = monomials_poly_features(X=x_train , degree=3)
    print(features.size)



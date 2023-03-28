import numpy as np

from dataset import get_data

X_train, X_test, Y_train, Y_test = get_data("./dataset/forestfires.csv")

# X = np.array([[1, 2, 3], [2, 2, 3], [5, 3, 2]])

# print(np.ones((4, 3)))
# print(np.sum(X**2, axis = 1))
# print((np.sum(X**2, axis = 1) * np.ones((4,3))))

# beta = np.array([1, -1, -2, 2])
# print(np.sign(beta))

# print(max(0, -1))

x = np.array([1, 2, 3, 4])
x.sort()
print(x)
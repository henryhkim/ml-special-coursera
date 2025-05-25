import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1,2])
print(f"x_train = {x_train}")
y_train = np.array([300, 500])
print(f"y_train = {y_train}")

m = x_train.shape[0]
print(f"number of training samples = {m}")
i = 0
print(f"x_train[0] = {x_train[i]}")
print(f"y_train[1] = {y_train[i]}")

# plot the data
plt.scatter(x_train, y_train, marker='x', color='red')
plt.title("housing prices")
plt.ylabel("price in $1000s")
plt.xlabel("size 1000 sq ft")
plt.show()

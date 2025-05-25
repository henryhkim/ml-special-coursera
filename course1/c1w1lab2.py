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
# plt.show()

w = 200
b = 100
print(f"w = {w}, b = {b}")

def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb

f_wb = compute_model_output(x_train, w, b)
print(f"f_wb = {f_wb}")
# plot the model output
plt.plot(x_train, f_wb, color='blue', label='our prediction')
plt.scatter(x_train, y_train, marker='x', color='red', label='actual values (training data)')
plt.title("housing prices")
plt.ylabel("price in $1000s")
plt.xlabel("size 1000 sq ft")
plt.legend()
# plt.show()

x_i = 1.2
cost_1200sqft = w * x_i + b
print(f"cost of 1200 sqft house = {cost_1200sqft:0f} thousand dollars")
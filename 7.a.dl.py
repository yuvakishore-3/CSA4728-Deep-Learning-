import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  
y = 4 + 3 * X + np.random.randn(100, 1)  
plt.scatter(X, y, color='blue', marker='o')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Synthetic Data')
plt.show()
def predict(X, theta):
    return X.dot(theta)
def compute_cost(X, y, theta):
    m = len(y)
    predictions = predict(X, theta)
    return (1 / (2 * m)) * np.sum(np.square(predictions - y))
def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    cost_history = np.zeros(num_iterations)   
    for i in range(num_iterations):
        gradients = (1 / m) * X.T.dot(predict(X, theta) - y)
        theta = theta - learning_rate * gradients
        cost_history[i] = compute_cost(X, y, theta)    
    return theta, cost_history
X_b = np.c_[np.ones((100, 1)), X]
theta_initial = np.random.randn(2, 1)
learning_rate = 0.1
num_iterations = 1000
theta_optimal, cost_history = gradient_descent(X_b, y, theta_initial, learning_rate, num_iterations)
print(f"Optimal parameters: {theta_optimal.ravel()}")
plt.plot(range(num_iterations), cost_history, 'r')
plt.xlabel('Number of iterations')
plt.ylabel('Cost')
plt.title('Cost History')
plt.show()
plt.scatter(X, y, color='blue', marker='o', label='Data')
X_fit = np.linspace(0, 2, 100).reshape(100, 1)
X_fit_b = np.c_[np.ones((100, 1)), X_fit]
y_fit = predict(X_fit_b, theta_optimal)
plt.plot(X_fit, y_fit, 'r-', label='Linear fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Fit')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)  
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1) 
def compute_cost(X, y, m, b):
    m_pred = X * m + b
    cost = np.mean((m_pred - y) ** 2) / 2
    return cost
def compute_gradients(X, y, m, b):
    m_pred = X * m + b
    dm = np.mean((m_pred - y) * X)
    db = np.mean(m_pred - y)
    return dm, db
def gradient_descent(X, y, m_init, b_init, learning_rate, num_iterations):
    m = m_init
    b = b_init
    m_history = [m]
    b_history = [b]
    cost_history = []
    for _ in range(num_iterations):
        dm, db = compute_gradients(X, y, m, b)
        m -= learning_rate * dm
        b -= learning_rate * db
        m_history.append(m)
        b_history.append(b)
        cost_history.append(compute_cost(X, y, m, b))
    return m, b, m_history, b_history, cost_history
m_init = 0
b_init = 0
learning_rate = 0.1
num_iterations = 1000

m_final, b_final, m_history, b_history, cost_history = gradient_descent(
    X, y, m_init, b_init, learning_rate, num_iterations
)
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, m_final * X + b_final, color='red', label='Fitted Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Data Points and Fitted Line')
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(cost_history, color='green')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function History')
plt.grid(True)
plt.tight_layout()
plt.show()

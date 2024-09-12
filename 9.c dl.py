import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
np.random.seed(0)
X = np.linspace(0, 10, 100)
y = 3 * X + np.random.normal(0, 1, X.shape[0]) 
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)
def create_model():
    model = Sequential([
        Dense(10, activation='relu', input_shape=(1,)),  
        Dense(10, activation='relu'),                    
        Dense(1, activation='linear')                    
    ])

    model.compile(optimizer='adam',
                  loss='mean_squared_error')
    return model
model = create_model()
history = model.fit(X, y, epochs=100, batch_size=10, validation_split=0.2, verbose=0)
y_pred = model.predict(X)
mse = np.mean((y - y_pred) ** 2)
print(f'Mean Squared Error (MSE): {mse:.4f}')
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='True Data', color='blue')
plt.plot(X, y_pred, label='Predicted Line', color='red', linewidth=2)
plt.title('Linear Regression with Neural Network')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

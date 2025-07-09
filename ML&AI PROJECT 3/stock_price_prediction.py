"""
Project 6: Stock Price Prediction using LSTM

📌 Problem Statement:
Predicting stock trends can help users make smarter investments.

🎯 Objective:
Use LSTM (Long Short-Term Memory) neural networks to forecast stock prices.

✅ Requirements:
- Historical stock data from Yahoo Finance
- Normalize and reshape data for LSTM input
- Build and train LSTM model
- Plot actual vs predicted prices

📈 Expected Outcome:
Time-series prediction graph with future trend estimation — a real-life example of AI in finance.
"""

# Import required libraries
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Step 1: Download historical stock data
print("Downloading Apple (AAPL) stock data...")
df = yf.download('AAPL', start='2015-01-01', end='2021-01-01', auto_adjust=True)
data = df[['Close']].values

# Step 2: Normalize data to range [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Step 3: Create sequences for LSTM input
def create_sequences(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data)

# Step 4: Reshape input to (samples, time_steps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Step 5: Split into training and testing sets
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Step 6: Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))

# Step 7: Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
print("Training model...")
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# Step 8: Predict on test data
predictions = model.predict(X_test)

# Step 9: Inverse transform predictions and actual prices
predicted_price = scaler.inverse_transform(predictions)
real_price = scaler.inverse_transform(y_test.reshape(-1, 1))

# Step 10: Plot actual vs predicted prices
plt.figure(figsize=(12, 6))
plt.plot(real_price, label='Actual Price')
plt.plot(predicted_price, label='Predicted Price')
plt.title('AAPL Stock Price Prediction using LSTM')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 11: Evaluation Metrics
mse = mean_squared_error(real_price, predicted_price)
mae = mean_absolute_error(real_price, predicted_price)
print("📊 Model Evaluation:")
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)

# Optional: Pause to view plot if running from command line
input("Press Enter to exit...")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv('orders(2021-2023).csv')

# Mengubah kolom 'order_month_year' menjadi datetime
data['order_month_year'] = pd.to_datetime(data['order_month_year'])

# Membuat kolom bulan dan tahun
data['bulan'] = data['order_month_year'].dt.month
data['tahun'] = data['order_month_year'].dt.year

# Total revenue per month untuk tahun 2021 dan 2023
total_revenue_per_month = data.groupby(['tahun', 'bulan'])['total_revenue_per_product'].sum().reset_index()
total_revenue_per_month = total_revenue_per_month.rename(columns={'total_revenue_per_product': 'total_sales'})

# Normalisasi data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(total_revenue_per_month[['total_sales']])

# Membagi data menjadi train dan test (gunakan 2021 dan 2023 sebagai train)
train_size = int(len(data_scaled) * 0.8)
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]

# Fungsi untuk membuat dataset dalam bentuk yang bisa digunakan oleh LSTM
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Mengatur time_step
time_step = 1
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input untuk LSTM [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Membangun model LSTM
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(50, return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

# Kompilasi model
model.compile(optimizer='adam', loss='mean_squared_error')

# Melatih model
history = model.fit(X_train, y_train, epochs=200, batch_size=16, validation_split=0.1, verbose=1)

# Prediksi
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)  # Kembalikan ke skala asli
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))  # Kembalikan ke skala asli

# Menghitung MSE dan RMSE
mse = mean_squared_error(y_test_inv, y_pred)
rmse = np.sqrt(mse)

# Menampilkan hasil di terminal
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')

# Menampilkan prediksi total sales per month di tahun 2023 di terminal
predicted_sales = pd.DataFrame({'Bulan': np.arange(1, len(y_pred) + 1), 'Predicted Sales': y_pred.flatten()})
print("\nPredicted Total Sales per Month in 2024:")
print(predicted_sales)

# Visualisasi hanya prediksi total sales per month di tahun 2023
plt.figure(figsize=(12, 6))
months = np.arange(1, len(y_pred) + 1)  # Menghasilkan label bulan untuk prediksi
plt.plot(months, y_pred, label='Predicted Sales', color='orange', marker='x')
plt.title('Predicted Total Sales per Month in 2024')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.xticks(months)  # Label bulan
plt.xlim(1, 12)  # Batas sumbu x
plt.legend()
plt.grid()
plt.show()
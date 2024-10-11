import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load data
data = pd.read_csv('orders(2021-2023).csv')

# Mengubah kolom 'order_month_year' menjadi datetime
data['order_month_year'] = pd.to_datetime(data['order_month_year'])

# Membuat kolom bulan dan tahun
data['bulan'] = data['order_month_year'].dt.month
data['tahun'] = data['order_month_year'].dt.year

# Total revenue per month untuk tahun 2021 sampai 2023
total_revenue_per_month = data.groupby(['tahun', 'bulan'])['total_revenue_per_product'].sum().reset_index()
total_revenue_per_month = total_revenue_per_month.rename(columns={'total_revenue_per_product': 'total_sales'})

# Set the time series index for ARIMA by assigning day 1 for all dates
total_revenue_per_month['date'] = pd.to_datetime(total_revenue_per_month['tahun'].astype(str) + '-' + total_revenue_per_month['bulan'].astype(str) + '-01')

# Set the index to the 'date' column
total_revenue_per_month.set_index('date', inplace=True)

# Use total sales as the time series data
ts_data = total_revenue_per_month['total_sales']

# Split data into train and test sets (80% train, 20% test)
train_size = int(len(ts_data) * 0.7)
train_data, test_data = ts_data[:train_size], ts_data[train_size:]

# Fit the ARIMA model
# The (p, d, q) values are chosen based on trial and error or using AIC/BIC criteria
# (p: autoregressive part, d: differencing part, q: moving average part)
model = ARIMA(train_data, order=(2, 0, 1))  # Try different (p, d, q) values for best results
arima_model = model.fit()

# Make predictions
start = len(train_data)
end = len(train_data) + len(test_data) - 1
predictions = arima_model.predict(start=start, end=end, typ='levels')

# Evaluate the model
mse = mean_squared_error(test_data, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_data, predictions)

print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'Mean Absolute Error (MAE): {mae:.2f}')

# Plot only the predicted sales
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, predictions, label='Predicted Sales', color='orange', marker='x')
plt.title('ARIMA Model - Predicted Total Sales per Month')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.legend()
plt.grid()
plt.show()

# Forecast future sales for 5 months starting from January 2024
forecast_steps = 5
forecast = arima_model.forecast(steps=forecast_steps)

# Generate correct forecast months starting from January 2024
forecast_start_date = pd.Timestamp('2024-01-01')
forecast_months = pd.date_range(start=forecast_start_date, periods=forecast_steps, freq='MS')
forecast_df = pd.DataFrame({'Month': forecast_months, 'Forecasted Sales': forecast})
print("\nForecasted Total Sales for the Next 5 Months (2024):")
print(forecast_df)
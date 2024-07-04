import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset
data = pd.read_csv('E:/prodigy_intern/mlt_01/test.csv')

# Display the first few rows of the dataset
print(data.head())

# Add a synthetic 'price' column (you can adjust the formula as needed)
np.random.seed(42)
data['price'] = (data['square_footage'] * 150) + (data['bedrooms'] * 10000) + (data['bathrooms'] * 5000) + np.random.normal(0, 25000, size=len(data))

# Display the first few rows with the new 'price' column
print(data.head())

# Prepare the features and target variable
X = data[['square_footage', 'bedrooms', 'bathrooms']]
y = data['price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Make predictions on new data
new_data = pd.DataFrame({
    'square_footage': [2000, 1500],
    'bedrooms': [3, 2],
    'bathrooms': [2, 1]
})

predictions = model.predict(new_data)
print(f'Predictions: {predictions}')

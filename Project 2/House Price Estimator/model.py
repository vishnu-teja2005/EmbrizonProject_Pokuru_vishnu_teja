

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
data = {
    'Area': [1000, 1200, 1300, 1500, 1800, 2000, 2200],
    'Bedrooms': [2, 2, 3, 3, 4, 4, 5],
    'Price': [200000, 240000, 260000, 300000, 360000, 400000, 450000]
}

df = pd.DataFrame(data)

sns.pairplot(df)
plt.savefig("house_data_visualization.png")
plt.show()

X = df[['Area', 'Bedrooms']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predicted Prices:", y_pred)
print("Actual Prices:", list(y_test))
print("\nMean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

new_house = [[1600, 3]]
predicted_price = model.predict(new_house)
print(f"\nPredicted price for 1600 sq.ft and 3 bedrooms:₹{int(predicted_price[0])}")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('cleaned_customer_data.csv')

X = df.drop(['Total Spend', 'Is_High_Spender', 'Customer ID'], axis=1, errors='ignore')
y = df['Total Spend']


X = pd.get_dummies(X, drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print(f"R-Squared Score: {r2_score(y_test, y_pred):.4f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")

#Feature Importance
importance = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print("Feature Importance (Impact on Spending)")
print(importance.sort_values(by='Coefficient', ascending=False))
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

data = pd.read_csv("/workspaces/Projects/Classification/churn.csv")

print("First 5 rows of the dataset:")
print(data.head())
print("\nLast 5 rows of the dataset:")
print(data.tail())
print("\nDataset shape:", data.shape)
print("\nRandom sample of the dataset:")
print(data.sample(5))
print("\nDataset info:")
print(data.info())
print("\nSummary statistics:")
print(data.describe())
print("\nMissing values:")
print(data.isnull().sum())

if "customerID" in data.columns:
    data = data.drop("customerID", axis=1)

data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors='coerce')
data["TotalChargesPerTenure"] = data["TotalCharges"] / (data["tenure"] + 1e-6)

print("\nDataset with new feature:")
print(data[["tenure", "TotalCharges", "TotalChargesPerTenure", "Churn"]].head())

figure = px.scatter(data_frame=data, x="MonthlyCharges", y="TotalChargesPerTenure", 
                    size="tenure", color="Churn", 
                    title="MonthlyCharges vs TotalChargesPerTenure by Churn")
figure.show()

fig = px.box(data, x="Contract", y="MonthlyCharges", color="Churn", 
             title="MonthlyCharges Distribution by Contract Type and Churn")
fig.show()

fig = px.box(data, x="InternetService", y="tenure", color="Churn", 
             title="Tenure Distribution by InternetService and Churn")
fig.show()

correlation = data.select_dtypes(include=[np.number]).corr()
print("\nCorrelation with Churn:")
print(correlation["Churn"].sort_values(ascending=False))

numeric_features = ["tenure", "MonthlyCharges", "TotalChargesPerTenure"]
categorical_features = ["Contract", "InternetService", "PaymentMethod"]

data_encoded = pd.get_dummies(data[categorical_features], drop_first=True)
X = pd.concat([data[numeric_features], data_encoded], axis=1)
y = data["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential()
model.add(Dense(16, activation='relu', input_dim=X_train_scaled.shape[1]))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_scaled, y_train, epochs=100, verbose=0)

y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)

conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", report)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["No Churn", "Churn"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Churn Prediction")
plt.show()

print("\nChurn Prediction for a New Customer")
tenure = float(input("Tenure (months): "))
monthly_charges = float(input("Monthly Charges ($): "))
total_charges = float(input("Total Charges ($): "))
contract = input("Contract Type (Month-to-month, One year, Two year): ")
internet_service = input("Internet Service (DSL, Fiber optic, No): ")
payment_method = input("Payment Method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)): ")

total_charges_per_tenure = total_charges / (tenure + 1e-6)
new_data = pd.DataFrame({
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalChargesPerTenure": [total_charges_per_tenure]
})
new_data_encoded = pd.get_dummies(pd.DataFrame({
    "Contract": [contract],
    "InternetService": [internet_service],
    "PaymentMethod": [payment_method]
}), drop_first=True)

for col in data_encoded.columns:
    if col not in new_data_encoded.columns:
        new_data_encoded[col] = 0

new_data_encoded = new_data_encoded[X.columns[len(numeric_features):]]
new_features = pd.concat([new_data, new_data_encoded], axis=1)
new_features_scaled = scaler.transform(new_features)

prediction = model.predict(new_features_scaled)
churn_prob = prediction[0][0]
print(f"Predicted Churn Probability: {churn_prob:.2f}")
print("Predicted Outcome: ", "Churn" if churn_prob > 0.5 else "No Churn")
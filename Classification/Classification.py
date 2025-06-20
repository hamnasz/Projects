import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

churn_data = pd.read_csv("/workspaces/Projects/Classification/churn.csv")
print(churn_data.head())
print(churn_data.shape)
print(churn_data.ndim)
print(churn_data['Churn'].value_counts().rename('count'))
print(churn_data['Churn'].value_counts(normalize=True).rename('%').mul(100))
sns.countplot(data=churn_data, x='Churn')
plt.title('Number of Customers')
plt.show()
print(churn_data.info())
print(churn_data.sample(20))
print(churn_data.describe())
print(churn_data.columns)
print('Missing data sum:')
print(churn_data.isnull().sum())
print('\nMissing data percentage (%):')
print(churn_data.isnull().sum() / churn_data.count() * 100)
numerical_cols = churn_data.select_dtypes(include=[np.number]).columns
corr = churn_data[numerical_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(data=corr, annot=True, cmap='Spectral').set(title="Correlation Matrix")
plt.show()
corr_matrix = corr.round(2)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix, center=0, vmin=-1, vmax=1, mask=mask, annot=True, cmap='BrBG')
plt.show()
cat_features = [col for col in churn_data.columns if churn_data[col].dtypes == 'object' and col != 'customerID']
print('Number of categorical variables:', len(cat_features))
print('*'*80)
print('Categorical variables column name:', cat_features)
numerical_features = [col for col in churn_data.columns if churn_data[col].dtypes != 'object' and col not in ['customerID']]
print('Number of numerical variables:', len(numerical_features))
print('*'*80)
print('Numerical Variables Column:', numerical_features)
print('Duplicates:', churn_data.duplicated().sum())
for col in cat_features:
    print(f"Unique values in {col}:", churn_data[col].unique())
for col in numerical_features:
    print(f"Number of unique values in {col}:", churn_data[col].nunique())
for col in cat_features:
    plt.figure(figsize=(6, 3), dpi=100)
    sns.countplot(data=churn_data, x=col, hue='Churn', palette='gist_rainbow_r')
    plt.legend(loc=(1.05, 0.5))
    plt.xticks(rotation=45)
    plt.show()
for col in numerical_features:
    plt.figure(figsize=(6, 3), dpi=100)
    sns.barplot(data=churn_data, x='Churn', y=col, palette='gist_rainbow_r')
    plt.show()
churn_data['TotalCharges'] = pd.to_numeric(churn_data['TotalCharges'], errors='coerce')
churn_data['TotalCharges'] = churn_data['TotalCharges'].fillna(churn_data['TotalCharges'].mean())
print(churn_data.isnull().sum())
train = churn_data.drop(['customerID'], axis=1)
train_data_cat = train.select_dtypes("object")
train_data_num = train.select_dtypes(include=[np.number])
train_data_cat_encoded = pd.get_dummies(train_data_cat, columns=train_data_cat.columns.to_list())
data = pd.concat([train_data_cat_encoded, train_data_num], axis=1)
y = data['Churn_Yes']
x = data.drop(['Churn_Yes', 'Churn_No'], axis=1)
sc = StandardScaler()
x = sc.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=7)
print(X_train.shape, X_test.shape)
accuracy = {}
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
y_pred1 = lr.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred1))
accuracy[str(lr)] = accuracy_score(y_test, y_pred1) * 100
cm = confusion_matrix(y_test, y_pred1)
conf_matrix = pd.DataFrame(data=cm, columns=['Predicted:0', 'Predicted:1'], index=['Actual:0', 'Actual:1'])
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu")
plt.show()
print(classification_report(y_test, y_pred1))
dtc = DecisionTreeClassifier(max_depth=3)
dtc.fit(X_train, y_train)
y_pred2 = dtc.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred2))
accuracy[str(dtc)] = accuracy_score(y_test, y_pred2) * 100
cm = confusion_matrix(y_test, y_pred2)
conf_matrix = pd.DataFrame(data=cm, columns=['Predicted:0', 'Predicted:1'], index=['Actual:0', 'Actual:1'])
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu")
plt.show()
print(classification_report(y_test, y_pred2))
rfc = RandomForestClassifier(max_depth=5)
rfc.fit(X_train, y_train)
y_pred3 = rfc.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred3))
accuracy[str(rfc)] = accuracy_score(y_test, y_pred3) * 100
cm = confusion_matrix(y_test, y_pred3)
conf_matrix = pd.DataFrame(data=cm, columns=['Predicted:0', 'Predicted:1'], index=['Actual:0', 'Actual:1'])
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu")
plt.show()
print(classification_report(y_test, y_pred3))
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
gbc.fit(X_train, y_train)
y_pred4 = gbc.predict(X_test)
print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred4))
accuracy[str(gbc)] = accuracy_score(y_test, y_pred4) * 100
cm = confusion_matrix(y_test, y_pred4)
conf_matrix = pd.DataFrame(data=cm, columns=['Predicted:0', 'Predicted:1'], index=['Actual:0', 'Actual:1'])
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu")
plt.show()
print(classification_report(y_test, y_pred4))
svc = SVC()
svc.fit(X_train, y_train)
y_pred5 = svc.predict(X_test)
print("SVC Accuracy:", accuracy_score(y_test, y_pred5))
accuracy[str(svc)] = accuracy_score(y_test, y_pred5) * 100
cm = confusion_matrix(y_test, y_pred5)
conf_matrix = pd.DataFrame(data=cm, columns=['Predicted:0', 'Predicted:1'], index=['Actual:0', 'Actual:1'])
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu")
plt.show()
print(classification_report(y_test, y_pred5))
print("Accuracy scores:", accuracy)
smote = SMOTE()
x1, y1 = smote.fit_resample(x, y)
X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, shuffle=True, random_state=3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
y_pred1 = lr.predict(X_test)
print("Logistic Regression (SMOTE) Accuracy:", accuracy_score(y_test, y_pred1))
accuracy[str(lr)] = accuracy_score(y_test, y_pred1) * 100
cm = confusion_matrix(y_test, y_pred1)
conf_matrix = pd.DataFrame(data=cm, columns=['Predicted:0', 'Predicted:1'], index=['Actual:0', 'Actual:1'])
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu")
plt.show()
print(classification_report(y_test, y_pred1))
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
knn_predict = knn_model.predict(X_test)
print("KNN (SMOTE) Accuracy:", accuracy_score(y_test, knn_predict))
accuracy[str(knn_model)] = accuracy_score(y_test, knn_predict) * 100
cm = confusion_matrix(y_test, knn_predict)
conf_matrix = pd.DataFrame(data=cm, columns=['Predicted:0', 'Predicted:1'], index=['Actual:0', 'Actual:1'])
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu")
plt.show()
print(classification_report(y_test, knn_predict))
model = keras.Sequential([
    keras.layers.Dense(4800, input_shape=(x.shape[1],), activation='relu'),
    keras.layers.Dense(2000, activation='relu'),
    keras.layers.Dense(1000, activation='relu'),
    keras.layers.Dense(1000, activation='relu'),
    keras.layers.Dense(1, activation="sigmoid")
])
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=100, verbose=0)
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("Neural Network 1 Accuracy:", acc)
y_pred = model.predict(X_test).flatten()
y_pred = np.round(y_pred)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
conf_matrix = pd.DataFrame(data=cm, columns=['Predicted:0', 'Predicted:1'], index=['Actual:0', 'Actual:1'])
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu")
plt.show()
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(x.shape[1],)))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
cb = EarlyStopping(monitor='accuracy', min_delta=0.001, patience=100, mode='auto')
model.fit(X_train, y_train, epochs=50, batch_size=100, validation_split=0.30, callbacks=[cb], verbose=0)
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("Neural Network 2 Accuracy:", acc)
y_pred = model.predict(X_test).flatten()
y_pred = np.round(y_pred)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
conf_matrix = pd.DataFrame(data=cm, columns=['Predicted:0', 'Predicted:1'], index=['Actual:0', 'Actual:1'])
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu")
plt.show()
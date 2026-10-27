import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('StudentPerformanceFactors.csv')

print(df.isnull().sum())

# Select INPUT & OUTPUT variables

X = df[["Hours_Studied"]]
y = df["Exam_Score"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model 
model = LinearRegression()
model.fit(X_train, y_train)

#Predict 
y_pred = model.predict(X_test)

# Evaluate 
print("Mean Squared Error :", mean_squared_error(y_test, y_pred))
print("R2 Score :", r2_score(y_test, y_pred))

# Plotting the results

plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, color="red")
plt.title("Linear Regression")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.show()
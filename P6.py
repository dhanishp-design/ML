
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Select features
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Scale data
X_scaled = StandardScaler().fit_transform(X)

# Apply K-Means
model = KMeans(n_clusters=5, random_state=42, n_init=10)
labels = model.fit_predict(X_scaled)

# Evaluate clustering
print("Silhouette Score:", silhouette_score(X_scaled, labels))

# Plot clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
plt.xlabel("Annual Income (Scaled)")
plt.ylabel("Spending Score (Scaled)")
plt.title("K-Means Clustering")
plt.show()


import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load dataset
df = pd.read_csv("Market_Basket_Optimisation.csv", header=None)

# Convert transactions into one-hot encoded format
transactions = []
for row in df.values:
    transactions.append(set(str(x) for x in row if pd.notna(x)))

items = sorted(set(item for t in transactions for item in t))
encoded = pd.DataFrame([
    [item in t for item in items] for t in transactions
], columns=items)

# Find frequent itemsets
freq = apriori(encoded, min_support=0.01, use_colnames=True)
print("Frequent Itemsets:\n", freq.head())

# Generate association rules
rules = association_rules(freq, metric="confidence",
                          min_threshold=0.2)
print("Association Rules:\n", rules.head())

# Identify outliers using transaction size (IQR)
sizes = pd.Series([len(t) for t in transactions])
q1, q3 = sizes.quantile([0.25, 0.75])
iqr = q3 - q1
outliers = sizes[(sizes < q1 - 1.5 * iqr) |
                 (sizes > q3 + 1.5 * iqr)]

print("Outlier transaction indices:", outliers.index.tolist())

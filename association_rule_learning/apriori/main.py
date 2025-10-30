import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

# -----------------------------
# 1️⃣  Load and prepare dataset
# -----------------------------

# Get current file directory and construct dataset path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, '..', 'knowledge', 'Market_Basket_Optimisation.csv')

# Read dataset: each row is a customer's transaction (20 possible items per basket)
dataset = pd.read_csv(dataset_path, header=None)

# Convert dataset into a list of transactions (each transaction = list of items)
transactions = []
for i in range(0, dataset.shape[0]):  # dynamically get number of rows
    transaction = [str(dataset.values[i, j]) for j in range(0, 20) if str(dataset.values[i, j]) != 'nan']
    transactions.append(transaction)

# -----------------------------
# 2️⃣  Apply Apriori algorithm
# -----------------------------
# Parameters explained:
# - min_support: item combinations appearing in at least 0.3% of transactions
# - min_confidence: at least 20% of the time A leads to B
# - min_lift: lift >= 3 (strong rule)
# - min_length / max_length: look for pairs of items only
rules = apriori(
    transactions=transactions,
    min_support=0.003,
    min_confidence=0.2,
    min_lift=3,
    min_length=2,
    max_length=2
)

# Convert generator object to list for further analysis
results = list(rules)

# -----------------------------
# 3️⃣  Function to extract rule components
# -----------------------------
def inspect(results):
    """
    Convert raw 'apyori' rule output into a more readable structured list.
    Each entry: (Left Hand Side, Right Hand Side, Support, Confidence, Lift)
    """
    lhs = []
    rhs = []
    supports = []
    confidences = []
    lifts = []

    for result in results:
        # Each 'result' contains: (items, support, ordered_statistics)
        for stat in result.ordered_statistics:
            lhs.append(tuple(stat.items_base)[0])
            rhs.append(tuple(stat.items_add)[0])
            supports.append(result.support)
            confidences.append(stat.confidence)
            lifts.append(stat.lift)
    
    return list(zip(lhs, rhs, supports, confidences, lifts))

# Get readable results
results_data = inspect(results)

# -----------------------------
# 4️⃣  Convert to DataFrame and display top 10 rules
# -----------------------------
results_df = pd.DataFrame(results_data, columns=['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

# Sort by Lift to find strongest associations
top10 = results_df.nlargest(n=10, columns='Lift')

print("\n🧾 Top 10 Association Rules (by Lift):\n")
print(top10)

# -----------------------------
# 5️⃣  Optional: visualize lift distribution
# -----------------------------
plt.figure(figsize=(8, 5))
plt.hist(results_df['Lift'], bins=30, edgecolor='k')
plt.title('Distribution of Lift Values in Discovered Rules')
plt.xlabel('Lift')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
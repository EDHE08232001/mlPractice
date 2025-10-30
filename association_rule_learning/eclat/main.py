# ==============================================================
# Market Basket Optimization – Enhanced Implementation
# ==============================================================

import os
import pandas as pd
from apyori import apriori

# --------------------------------------------------------------
# Configurable Parameters (Dashboard Vibes)
# --------------------------------------------------------------
MIN_SUPPORT = 0.003
MIN_CONFIDENCE = 0.2
MIN_LIFT = 3
MIN_LENGTH = 2
MAX_LENGTH = 2
TOP_N = 10

# --------------------------------------------------------------
# Dynamically resolve dataset path
# --------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, '..', 'knowledge', 'Market_Basket_Optimisation.csv')

# --------------------------------------------------------------
# Load Dataset
# --------------------------------------------------------------
dataset = pd.read_csv(dataset_path, header=None)

# --------------------------------------------------------------
# Preprocessing: Build Transaction List (Cleaner!)
# Remove 'nan' and strip whitespace.
# --------------------------------------------------------------
transactions = dataset.apply(lambda row: [str(item).strip() 
                                          for item in row.dropna()], axis=1).tolist()

# --------------------------------------------------------------
# Train Association Algorithm
# (Apriori approximates Eclat when max_length=2 + lift filter)
# --------------------------------------------------------------
rules = apriori(
    transactions=transactions,
    min_support=MIN_SUPPORT,
    min_confidence=MIN_CONFIDENCE,
    min_lift=MIN_LIFT,
    min_length=MIN_LENGTH,
    max_length=MAX_LENGTH
)

results = list(rules)

# --------------------------------------------------------------
# Helper: Format results cleanly
# Each rule object contains association metrics buried deep.
# --------------------------------------------------------------
def extract_rule_summary(results):
    product_1 = []
    product_2 = []
    support = []
    confidence = []
    lift = []

    for rule in results:
        for stat in rule.ordered_statistics:
            product_1.append(list(stat.items_base)[0])
            product_2.append(list(stat.items_add)[0])
            support.append(rule.support)
            confidence.append(stat.confidence)
            lift.append(stat.lift)

    return pd.DataFrame({
        'Product 1': product_1,
        'Product 2': product_2,
        'Support': support,
        'Confidence': confidence,
        'Lift': lift
})

# --------------------------------------------------------------
# Build DataFrame of summarised rules
# --------------------------------------------------------------
df_rules = extract_rule_summary(results)

# --------------------------------------------------------------
# Display results, ordered by Support, Confidence, Lift
# --------------------------------------------------------------
print("\n=== TOP ASSOCIATIONS (by Support) ===\n")
print(df_rules.nlargest(TOP_N, 'Support'))

print("\n=== TOP ASSOCIATIONS (by Confidence) ===\n")
print(df_rules.nlargest(TOP_N, 'Confidence'))

print("\n=== TOP ASSOCIATIONS (by Lift) ===\n")
print(df_rules.nlargest(TOP_N, 'Lift'))
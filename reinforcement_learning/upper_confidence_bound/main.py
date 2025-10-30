"""
Upper Confidence Bound (UCB) Algorithm Implementation
-----------------------------------------------------
Author: Edward He
Purpose: Simulate Ad Click Optimization using UCB (a reinforcement learning approach)
Dataset: Ads_CTR_Optimisation.csv
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1️⃣ Load Dataset
# ----------------------------------------------------------------------

# Dynamically resolve the path to the dataset (relative to this script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, '..', 'knowledge', 'Ads_CTR_Optimisation.csv')

# Load the dataset into a pandas DataFrame
dataset = pd.read_csv(dataset_path)

# ----------------------------------------------------------------------
# 2️⃣ Initialize Parameters for UCB
# ----------------------------------------------------------------------

N = 10000  # Total number of rounds (simulated user interactions)
d = 10     # Number of different ads (arms of the bandit)

ads_selected = []                # Record which ad was selected each round
numbers_of_selections = [0] * d  # Count how many times each ad has been selected
sums_of_rewards = [0] * d        # Total reward accumulated by each ad
total_reward = 0                 # Overall reward across all rounds

# ----------------------------------------------------------------------
# 3️⃣ Implement the Upper Confidence Bound Algorithm
# ----------------------------------------------------------------------

for n in range(N):
    ad = 0
    max_upper_bound = 0

    # Loop through all ads to compute the upper confidence bound for each
    for i in range(d):
        if numbers_of_selections[i] > 0:
            # Calculate the average reward (empirical mean) of this ad
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]

            # Calculate the confidence interval (exploration term)
            delta_i = math.sqrt((3/2) * math.log(n + 1) / numbers_of_selections[i])

            # Compute the UCB score
            upper_bound = average_reward + delta_i
        else:
            # Ensure every ad is selected at least once initially
            upper_bound = float('inf')

        # Select the ad with the highest upper bound
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i

    # Update tracking lists after selecting the ad
    ads_selected.append(ad)
    numbers_of_selections[ad] += 1

    # Extract reward (0 or 1) for the chosen ad in this round
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] += reward
    total_reward += reward

# ----------------------------------------------------------------------
# 4️⃣ Visualize Results
# ----------------------------------------------------------------------

plt.figure(figsize=(10, 6))
plt.hist(ads_selected, bins=np.arange(d + 1) - 0.5, rwidth=0.8, color='skyblue', edgecolor='black')
plt.title('Histogram of Ad Selections (UCB Results)', fontsize=14)
plt.xlabel('Ad Index', fontsize=12)
plt.ylabel('Number of Times Each Ad Was Selected', fontsize=12)
plt.xticks(range(d))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# ----------------------------------------------------------------------
# ✅ Summary Output
# ----------------------------------------------------------------------

print(f"Total Reward: {total_reward}")
best_ad = np.argmax(sums_of_rewards)
print(f"Most Selected Ad: Ad #{best_ad} with {sums_of_rewards[best_ad]} total rewards.")
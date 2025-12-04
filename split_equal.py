import pandas as pd

# Load CSV
df = pd.read_csv("Spain_RealEstate_Merged.csv")

# Shuffle randomly
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 50/50 split
half = len(df_shuffled) // 2

train = df_shuffled.iloc[:half]
test = df_shuffled.iloc[half:]

# Save output files
train.to_csv("train_part.csv", index=False)
test.to_csv("test_part.csv", index=False)

print("Rows total:", len(df))
print("Train part:", len(train))
print("Test part:", len(test))
print("\nSaved as: train_part.csv and test_part.csv")

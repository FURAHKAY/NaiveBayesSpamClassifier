import pandas as pd
from NaiveBayesClassifier import NaiveBayesSpamClassifier

# Load SMS spam data
df = pd.read_csv("SMSSpamCollection", sep='\t', header=None, names=["label", "message"])

# Prepare (message, label) pairs
train_data = [(row["message"], row["label"]) for _, row in df.iterrows()]

# Train model
model = NaiveBayesSpamClassifier()
model.train(train_data)
# Load Enron dataset
enron_df = pd.read_csv("emails.csv")  # No sep='\t'
#
# # Predict for each Enron email message
# for idx, row in enron_df.iterrows():
#     message = row["message"]
#     prediction = model.predict(message)
#     print(f"Email {idx}: Predicted as {prediction}")
print("First few Enron emails:")
print(enron_df.head())

print("\nPredictions on Enron emails:")

for idx, row in enron_df.iterrows():
    email_text = row["message"]
    prediction = model.predict(email_text)
    print(f"Email {idx} predicted as: {prediction}")
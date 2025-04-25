# Naive Bayes Spam Classifier

A simple but powerful spam classifier built from scratch using Python and Naive Bayes.  
Trained on the SMS Spam Collection dataset and tested on real-world emails (Enron dataset).

---

## Features

- Built with pure Python (no scikit-learn!)
- Preprocessing: tokenization, lowercasing, Laplace smoothing
- Train on labeled SMS spam data
- Predict spam/ham on real email datasets (Enron emails)
- Log-probabilities to handle underflow
- Ready for extension (TF-IDF, stopword removal, bigrams)

---

## Project Structure
- SpamClassifier
    - NaiveBayesClassifier.py   # Your classifier class 
    - main.py                   # The script above 
    - SMSSpamCollection         # Downloaded labeled dataset (for training)
    - emails.csv                # Your Enron email data (for testing)
    - README.md                  # (optional) Project description


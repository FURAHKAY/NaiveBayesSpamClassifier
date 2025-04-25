import re
from collections import defaultdict
import math

import re
from collections import defaultdict
import math

def tokenize(text):
    text = text.lower()
    words = re.findall(r'\b[a-z]{2,}\b', text)  # remove punctuation, keep words only
    return words

class NaiveBayesSpamClassifier:
    def __init__(self):
        self.spam_counts = defaultdict(int)
        self.ham_counts = defaultdict(int)
        self.spam_total = 0
        self.ham_total = 0
        self.vocab = set()
        self.num_spam = 0
        self.num_ham = 0

    def train(self, data):
        for message, label in data:
            words = tokenize(message)
            self.vocab.update(words)

            if label == "spam":
                self.num_spam += 1
                for word in words:
                    self.spam_counts[word] += 1
                    self.spam_total += 1
            else:
                self.num_ham += 1
                for word in words:
                    self.ham_counts[word] += 1
                    self.ham_total += 1

        self.total_messages = self.num_spam + self.num_ham

    def predict(self, message):
        words = tokenize(message)

        # log priors
        log_prob_spam = math.log(self.num_spam / self.total_messages)
        log_prob_ham = math.log(self.num_ham / self.total_messages)

        for word in words:
            # Laplace smoothing
            prob_word_given_spam = (self.spam_counts[word] + 1) / (self.spam_total + len(self.vocab))
            prob_word_given_ham = (self.ham_counts[word] + 1) / (self.ham_total + len(self.vocab))

            log_prob_spam += math.log(prob_word_given_spam)
            log_prob_ham += math.log(prob_word_given_ham)

        return "spam" if log_prob_spam > log_prob_ham else "ham"

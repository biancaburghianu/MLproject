
import os
import re
import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

def read_messages(directory):
    texts = []
    labels = []
    spam_regex = re.compile(r'^spmsg')

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8', errors='ignore') as file:
                texts.append(file.read())
                labels.append(1 if spam_regex.match(filename) else 0)
    return texts, labels

main_dir = "/Users/bianca.burghianu/Documents/GitHub/MLproject/lingspam_public/stop"

train_texts, train_labels = [], []
for i in range(1, 10):
    part_dir = os.path.join(main_dir, f'part{i}')
    texts, labels = read_messages(part_dir)
    train_texts.extend(texts)
    train_labels.extend(labels)

test_texts, test_labels = read_messages(os.path.join(main_dir, 'part10'))

start_time = time.time()
train_texts, train_labels = [], []
test_texts, test_labels = [], []

for i in range(1, 10):
    texts, labels = read_messages(os.path.join(main_dir, f'part{i}'))
    train_texts.extend(texts)
    train_labels.extend(labels)

texts, labels = read_messages(os.path.join(main_dir, 'part10'))
test_texts.extend(texts)
test_labels.extend(labels)

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(train_texts)
X_test_tfidf = tfidf_vectorizer.transform(test_texts)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(train_labels)
y_test_encoded = label_encoder.transform(test_labels)

ada_classifier = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(criterion='entropy'),
    n_estimators=50,
    random_state=0
)
ada_classifier.fit(X_train_tfidf.toarray(), train_labels)

y_pred_ada = ada_classifier.predict(X_test_tfidf.toarray())

accuracy_ada = accuracy_score(test_labels, y_pred_ada)
report_ada = classification_report(test_labels, y_pred_ada, target_names=['ham', 'spam'], zero_division=0)

end_time = time.time()

duration = end_time-start_time

print(f'AdaBoost Accuracy: {accuracy_ada}')
print(report_ada)
print(duration)

loo = LeaveOneOut()
loo_accuracies = []


subset_size = 550
X_train_subset = X_train_tfidf[:subset_size]
y_train_subset = train_labels[:subset_size]

for train_index, test_index in tqdm(loo.split(X_train_subset), total=subset_size):
    X_train, X_test = X_train_subset[train_index], X_train_subset[test_index]
    y_train, y_test = np.array(y_train_subset)[train_index], np.array(y_train_subset)[test_index]

    ada_classifier = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(criterion='entropy'),
        n_estimators=50,
        random_state=0
    )
    ada_classifier.fit(X_train.toarray(), y_train)
    y_pred = ada_classifier.predict(X_test.toarray())

    loo_accuracies.append(accuracy_score(y_test, y_pred))

loo_mean_accuracy = np.mean(loo_accuracies)

plt.figure(figsize=(10, 6))
plt.plot(loo_accuracies)
plt.axhline(y=loo_mean_accuracy, color='r', linestyle='--')
plt.title('Rezultatele Cross-Validation Leave-One-Out cu AdaBoost')
plt.xlabel('Iteratie')
plt.ylabel('Acuratete')
plt.show()


plt.figure(figsize=(10, 6))
plt.bar(['Acuratețe Testare'], [accuracy_ada])
plt.title('Acuratețea pe Setul de Date de Testare cu AdaBoost')
plt.ylabel('Acuratețe')
plt.show()
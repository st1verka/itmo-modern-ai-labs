"""Пункт 4. Код Gaussian Naive Bayes из методички."""
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utilities import visualize_classifier
input_file = ROOT / "data" / "data_multivar_nb.txt"
data = np.loadtxt(input_file, delimiter=",")
X, y = data[:, :-1], data[:, -1]
classifier = GaussianNB()
classifier.fit(X, y)
y_pred = classifier.predict(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=3)
classifier_new = GaussianNB()
classifier_new.fit(X_train, y_train)
y_test_pred = classifier_new.predict(X_test)
metrics = {name: cross_val_score(GaussianNB(), X, y, scoring=score, cv=3).mean() for name, score in {"Accuracy":"accuracy", "Precision":"precision_weighted", "Recall":"recall_weighted", "F1":"f1_weighted"}.items()}
visualize_classifier(classifier, X, y, ROOT / "screenshots" / "04_naive_bayes_full.png", "Gaussian Naive Bayes: все данные")
visualize_classifier(classifier_new, X_test, y_test, ROOT / "screenshots" / "04_naive_bayes_test.png", "Gaussian Naive Bayes: тестовая выборка")
text = "ПУНКТ 4. НАИВНЫЙ БАЙЕСОВСКИЙ КЛАССИФИКАТОР\n\n"
text += f"Accuracy of Naive Bayes classifier = {accuracy_score(y, y_pred) * 100:.2f}%\n"
text += f"Accuracy of the new classifier = {accuracy_score(y_test, y_test_pred) * 100:.2f}%\n"
text += "\n".join(f"{name} (CV=3): {value:.2%}" for name, value in metrics.items()) + "\n"
(ROOT / "results" / "04_naive_bayes.txt").write_text(text, encoding="utf-8")
print(text)

"""Пункт 3. Код логистического классификатора из методички."""
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import numpy as np
from sklearn import linear_model
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utilities import visualize_classifier
X = np.array([[3.1,7.2],[4,6.7],[2.9,8],[5.1,4.5],[6,5],[5.6,5],[3.3,.4],[3.9,.9],[2.8,1],[.5,3.4],[1,4],[.6,4.9]])
y = np.array([0,0,0,1,1,1,2,2,2,3,3,3])
rows = ["ПУНКТ 3. ЛОГИСТИЧЕСКИЙ КЛАССИФИКАТОР", ""]
for c in (1, 100):
    # В методичке: linear_model.LogisticRegression(solver='liblinear', C=c).
    # Обёртка OneVsRest нужна только потому, что современный sklearn больше не
    # включает многоклассовый режим liblinear автоматически.
    classifier = OneVsRestClassifier(linear_model.LogisticRegression(solver="liblinear", C=c))
    classifier.fit(X, y)
    rows.append(f"C={c}: accuracy = {accuracy_score(y, classifier.predict(X)):.2%}")
    visualize_classifier(classifier, X, y, ROOT / "screenshots" / f"03_logistic_C_{c}.png", f"Логистическая регрессия, C={c}")
rows.append("При C=100 регуляризация слабее, поэтому границы сильнее подстраиваются под данные.")
text = "\n".join(rows) + "\n"
(ROOT / "results" / "03_logistic_regression.txt").write_text(text, encoding="utf-8")
print(text)

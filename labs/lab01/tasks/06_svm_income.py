"""Пункт 6. Код SVM-классификации дохода из методички."""
from pathlib import Path
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier

ROOT = Path(__file__).resolve().parents[1]
input_file = ROOT / "data" / "income_data.txt"
X, y = [], []
count_class1 = count_class2 = 0
max_datapoints = 25000
with open(input_file, "r", encoding="utf-8") as file:
    for line in file.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints: break
        if "?" in line: continue
        data = line[:-1].split(", ")
        if data[-1] == "<=50K" and count_class1 < max_datapoints:
            X.append(data); count_class1 += 1
        if data[-1] == ">50K" and count_class2 < max_datapoints:
            X.append(data); count_class2 += 1
X = np.array(X)
label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit(): X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])
X, y = X_encoded[:, :-1].astype(int), X_encoded[:, -1].astype(int)
classifier = OneVsOneClassifier(LinearSVC(random_state=0))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=5)
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)
f1 = cross_val_score(classifier, X, y, scoring="f1_weighted", cv=3)
samples = [["37","Private","215646","HS-grad","9","Never-married","Handlers-cleaners","Not-in-family","White","Male","0","0","40","United-States"], ["52","Self-emp-not-inc","209642","Bachelors","13","Married-civ-spouse","Exec-managerial","Husband","White","Male","0","0","50","United-States"], ["28","Private","338409","Bachelors","13","Never-married","Prof-specialty","Not-in-family","Asian-Pac-Islander","Female","0","0","45","United-States"]]
encoded_samples = []
for input_data in samples:
    input_data_encoded = [-1] * len(input_data); count = 0
    for i, item in enumerate(input_data):
        if item.isdigit(): input_data_encoded[i] = int(item)
        else:
            input_data_encoded[i] = label_encoder[count].transform([item])[0]; count += 1
    encoded_samples.append(input_data_encoded)
sample_predictions = label_encoder[-1].inverse_transform(classifier.predict(np.array(encoded_samples)))
text = f"""ПУНКТ 6. SVM-КЛАССИФИКАЦИЯ УРОВНЯ ДОХОДА

Количество очищенных наблюдений: {len(X)}
F1 score: {100 * f1.mean():.2f}%

Прогнозы для трёх людей:
Точка 1: {sample_predictions[0]}
Точка 2: {sample_predictions[1]}
Точка 3: {sample_predictions[2]}
"""
(ROOT / "results" / "06_svm_income.txt").write_text(text, encoding="utf-8")
print(text)

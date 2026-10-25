"""Пункт 7. Одномерная и многомерная регрессия."""
from pathlib import Path
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.preprocessing import PolynomialFeatures

ROOT = Path(__file__).resolve().parents[1]
input_file = ROOT / "data" / "data_singlevar_regr.txt"
data = np.loadtxt(input_file, delimiter=","); X, y = data[:, :-1], data[:, -1]
num_training = int(.8 * len(X)); num_test = len(X) - num_training
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]
model = LinearRegression(); model.fit(X_train, y_train); predicted = model.predict(X_test)
with open(ROOT / "results" / "singlevar_linear_model.pkl", "wb") as file: pickle.dump(model, file)
order = np.argsort(X_test[:, 0]); fig, ax = plt.subplots(figsize=(8, 5.5)); ax.scatter(X_test, y_test, color="forestgreen", label="Тестовые точки"); ax.plot(X_test[order], predicted[order], color="black", linewidth=2.5, label="Линейная регрессия"); ax.set(xlabel="Признак", ylabel="Целевое значение", title="Одномерная линейная регрессия"); ax.legend(); fig.tight_layout(); fig.savefig(ROOT / "screenshots" / "07_singlevar_regression.png", dpi=180); plt.close(fig)
input_file = ROOT / "data" / "data_multivar_regr.txt"
data = np.loadtxt(input_file, delimiter=","); Xm, ym = data[:, :-1], data[:, -1]
num_training = int(.8 * len(Xm)); num_test = len(Xm) - num_training
Xm_train, ym_train = Xm[:num_training], ym[:num_training]
Xm_test, ym_test = Xm[num_training:], ym[num_training:]
linear = LinearRegression(); linear.fit(Xm_train, ym_train)
polynomial_features = PolynomialFeatures(degree=10)
Xm_train_transformed = polynomial_features.fit_transform(Xm_train)
polynomial = LinearRegression(); polynomial.fit(Xm_train_transformed, ym_train)
point = np.array([[7.75, 6.35, 5.56]])
poly_point = polynomial_features.transform(point)
text = f"""ПУНКТ 7. РЕГРЕССИЯ

7.1 Одномерный регрессор
Mean absolute error = {mean_absolute_error(y_test, predicted):.2f}
Mean squared error = {mean_squared_error(y_test, predicted):.2f}
Median absolute error = {median_absolute_error(y_test, predicted):.2f}
Explained variance score = {explained_variance_score(y_test, predicted):.2f}
R2 score = {model.score(X_test, y_test):.2f}

7.2 Многомерный регрессор
Линейная регрессия, прогноз для {point[0].tolist()}: {linear.predict(point)[0]:.2f}
Полиномиальная регрессия (степень 10), прогноз: {polynomial.predict(poly_point)[0]:.2f}
R2 линейной модели на тесте: {linear.score(Xm_test, ym_test):.2f}
R2 полиномиальной модели на тесте: {polynomial.score(polynomial_features.transform(Xm_test), ym_test):.2f}
"""
(ROOT / "results" / "07_regression.txt").write_text(text, encoding="utf-8")
print(text)

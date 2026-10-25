"""Пункт 8. Оценка стоимости жилья с помощью регрессора SVR."""
from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import explained_variance_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.utils import shuffle

ROOT = Path(__file__).resolve().parents[1]
housing = fetch_california_housing(data_home=ROOT / "data" / "sklearn_cache")
X, y = shuffle(housing.data, housing.target, random_state=7)
X, y = X[:3000], y[:3000]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=7)
rows = ["ПУНКТ 8. ОЦЕНКА СТОИМОСТИ ЖИЛЬЯ С ПОМОЩЬЮ SVR", "", "Датасет: California Housing (современная замена удалённого Boston Housing)."]
for c, epsilon in [(1.0, .1), (10.0, .1), (10.0, .3)]:
    model = make_pipeline(StandardScaler(), SVR(kernel="linear", C=c, epsilon=epsilon)).fit(X_train, y_train)
    predicted = model.predict(X_test); forecasts = model.predict(X_test[:3])
    rows += ["", f"C={c}, epsilon={epsilon}", f"Mean squared error = {mean_squared_error(y_test, predicted):.3f}", f"Explained variance score = {explained_variance_score(y_test, predicted):.3f}"]
    rows += [f"Прогноз для тестовой точки {i}: {value:.3f} (100 тыс. USD)" for i, value in enumerate(forecasts, 1)]
text = "\n".join(rows) + "\n"
(ROOT / "results" / "08_housing_svr.txt").write_text(text, encoding="utf-8")
print(text)

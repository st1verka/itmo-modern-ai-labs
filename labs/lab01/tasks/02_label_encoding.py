"""Пункт 2. Кодирование меток."""
from pathlib import Path
from sklearn import preprocessing

ROOT = Path(__file__).resolve().parents[1]
labels = ["red", "black", "red", "green", "black", "yellow", "white"]
encoder = preprocessing.LabelEncoder().fit(labels)
test_labels = ["green", "red", "black"]
mapping = "\n".join(f"{item} --> {i}" for i, item in enumerate(encoder.classes_))
text = f"""ПУНКТ 2. КОДИРОВАНИЕ МЕТОК

Соответствие меток:
{mapping}

Метки: {test_labels}
Коды: {list(encoder.transform(test_labels))}
Коды для декодирования: [3, 0, 4, 1]
Декодированные метки: {list(encoder.inverse_transform([3, 0, 4, 1]))}
"""
(ROOT / "results" / "02_label_encoding.txt").write_text(text, encoding="utf-8")
print(text)

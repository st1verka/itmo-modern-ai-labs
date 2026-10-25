"""Пункт 1. Предварительная обработка данных."""
from pathlib import Path
import numpy as np
from sklearn import preprocessing

ROOT = Path(__file__).resolve().parents[1]
input_data = np.array([[5.1,-2.9,3.3],[-1.2,7.8,-6.1],[3.9,.4,2.1],[7.3,-9.9,-4.5]])
binarized = preprocessing.Binarizer(threshold=2.1).transform(input_data)
scaled = preprocessing.scale(input_data)
minmax = preprocessing.MinMaxScaler().fit_transform(input_data)
l1 = preprocessing.normalize(input_data, norm="l1")
l2 = preprocessing.normalize(input_data, norm="l2")
text = f"""ПУНКТ 1. ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА ДАННЫХ

Исходные данные:
{input_data}

Бинаризация (порог 2.1):
{binarized}

Среднее до: {input_data.mean(axis=0)}
Стандартное отклонение до: {input_data.std(axis=0)}
Среднее после стандартизации: {scaled.mean(axis=0)}
Стандартное отклонение после: {scaled.std(axis=0)}

MinMax [0, 1]:
{minmax}

L1-нормализация:
{l1}

L2-нормализация:
{l2}
"""
(ROOT / "results" / "01_preprocessing.txt").write_text(text, encoding="utf-8")
print(text)

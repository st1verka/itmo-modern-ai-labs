"""Пункт 5. Матрица неточностей и отчёт классификации."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
true_labels = [2,0,0,2,4,4,1,0,3,3,3]
pred_labels = [2,1,0,2,4,3,1,0,1,3,3]
cm = confusion_matrix(true_labels, pred_labels, labels=range(5))
fig, ax = plt.subplots(figsize=(7, 5.5)); image = ax.imshow(cm, cmap="Blues"); fig.colorbar(image, ax=ax)
ax.set(xticks=range(5), yticks=range(5), xlabel="Предсказанный класс", ylabel="Истинный класс", title="Матрица неточностей")
for i in range(5):
    for j in range(5): ax.text(j, i, cm[i, j], ha="center", va="center")
fig.tight_layout(); fig.savefig(ROOT / "screenshots" / "05_confusion_matrix.png", dpi=180); plt.close(fig)
report = classification_report(true_labels, pred_labels, target_names=[f"Class-{i}" for i in range(5)], zero_division=0)
text = f"ПУНКТ 5. МАТРИЦА НЕТОЧНОСТЕЙ\n\nМатрица:\n{cm}\n\nОтчёт классификации:\n{report}"
(ROOT / "results" / "05_confusion_matrix.txt").write_text(text, encoding="utf-8")
print(text)

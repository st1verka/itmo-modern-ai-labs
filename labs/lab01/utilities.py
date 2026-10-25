"""Вспомогательная функция построения границ классификации из методички."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def visualize_classifier(classifier, X, y, output_file: Path, title: str) -> None:
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, .02), np.arange(min_y, max_y, .02))
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()]).reshape(x_vals.shape)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.Pastel2, shading="auto")
    ax.scatter(X[:, 0], X[:, 1], c=y, s=70, edgecolors="black", linewidth=.8, cmap=plt.cm.Set1)
    ax.set(title=title, xlabel="Признак 1", ylabel="Признак 2")
    fig.tight_layout(); fig.savefig(output_file, dpi=180); plt.close(fig)

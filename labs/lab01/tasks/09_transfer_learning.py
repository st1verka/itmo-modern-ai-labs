"""Пункт 9. Классификация изображений с Feature Extraction и Fine-tuning."""
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
SHOT_DIR = ROOT / "screenshots"
CLASSES = ["0", "1", "2"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(4)


class DigitDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images, self.labels, self.transform = images, labels, transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = Image.fromarray((self.images[index] / 16 * 255).astype(np.uint8)).convert("RGB")
        return self.transform(image), int(self.labels[index])


def make_model(fine_tuning=False):
    # The MobileNetV2 ImageNet weights are available locally in the PyTorch cache.
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for parameter in model.parameters():
        parameter.requires_grad = False
    if fine_tuning:
        for parameter in model.features[-1].parameters():
            parameter.requires_grad = True
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASSES))
    return model.to(DEVICE)


def run_training(model, train_loader, test_loader, epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    history = []
    for epoch in range(epochs):
        model.train(); correct = total = 0; loss_sum = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(); output = model(images); loss = criterion(output, labels)
            loss.backward(); optimizer.step()
            loss_sum += loss.item() * labels.size(0); correct += (output.argmax(1) == labels).sum().item(); total += labels.size(0)
        history.append((loss_sum / total, correct / total))
    return history, evaluate(model, test_loader)


def evaluate(model, loader):
    model.eval(); true, predicted = [], []
    with torch.no_grad():
        for images, labels in loader:
            output = model(images.to(DEVICE)).argmax(1).cpu().numpy()
            predicted.extend(output.tolist()); true.extend(labels.numpy().tolist())
    accuracy = float(np.mean(np.array(true) == np.array(predicted)))
    precision, recall, _, _ = precision_recall_fscore_support(true, predicted, average="weighted", zero_division=0)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "true": true, "predicted": predicted}


def save_confusion(result, filename, title):
    matrix = confusion_matrix(result["true"], result["predicted"], labels=range(3))
    fig, ax = plt.subplots(figsize=(6.2, 5.2)); image = ax.imshow(matrix, cmap="Blues"); fig.colorbar(image, ax=ax)
    ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASSES, yticklabels=CLASSES, xlabel="Предсказанный класс", ylabel="Истинный класс", title=title)
    for i in range(3):
        for j in range(3): ax.text(j, i, matrix[i, j], ha="center", va="center")
    fig.tight_layout(); fig.savefig(SHOT_DIR / filename, dpi=180); plt.close(fig)


def main():
    normalize = transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    train_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.RandomAffine(10, translate=(.08, .08)), transforms.ToTensor(), normalize])
    test_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), normalize])
    digits = load_digits()
    mask = np.isin(digits.target, [0, 1, 2])
    images, labels = digits.images[mask], digits.target[mask]
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=.25, random_state=SEED, stratify=labels)
    train_loader = DataLoader(DigitDataset(train_images, train_labels, train_transform), batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(DigitDataset(test_images, test_labels, test_transform), batch_size=64, shuffle=False, num_workers=0)

    feature_model = make_model(False)
    feature_history, feature_result = run_training(feature_model, train_loader, test_loader, epochs=2, learning_rate=1e-3)
    # Fine-tuning is a continuation of Feature Extraction: retain the learned
    # classifier head, unfreeze the final MobileNetV2 block, and reduce LR.
    fine_model = feature_model
    for parameter in fine_model.features[-1].parameters():
        parameter.requires_grad = True
    fine_history, fine_result = run_training(fine_model, train_loader, test_loader, epochs=5, learning_rate=1e-4)
    save_confusion(feature_result, "09_feature_extraction_confusion.png", "Feature Extraction: матрица ошибок")
    save_confusion(fine_result, "09_fine_tuning_confusion.png", "Fine-tuning: матрица ошибок")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4)); feature_epochs = [1, 2]; fine_epochs = [3, 4, 5, 6, 7]
    axes[0].plot(feature_epochs, [x[0] for x in feature_history], marker="o", label="Feature Extraction")
    axes[0].plot(fine_epochs, [x[0] for x in fine_history], marker="o", label="Fine-tuning")
    axes[0].set(title="Функция потерь", xlabel="Эпоха", ylabel="Loss"); axes[0].legend()
    axes[1].plot(feature_epochs, [x[1] for x in feature_history], marker="o", label="Feature Extraction")
    axes[1].plot(fine_epochs, [x[1] for x in fine_history], marker="o", label="Fine-tuning")
    axes[1].set(title="Точность на обучении", xlabel="Эпоха", ylabel="Accuracy"); axes[1].legend()
    fig.tight_layout(); fig.savefig(SHOT_DIR / "09_training_comparison.png", dpi=180); plt.close(fig)

    lines = [
        "ПУНКТ 9. КЛАССИФИКАЦИЯ ИЗОБРАЖЕНИЙ С ПРИМЕНЕНИЕМ TRANSFER LEARNING", "",
        "Датасет: sklearn Digits; классы: 0, 1, 2.",
        f"Обучение: {len(train_labels)} изображений; тест: {len(test_labels)} изображений.",
        "Архитектура: MobileNetV2 с весами ImageNet.", "",
        "Feature Extraction (заморожена базовая сеть, обучается fc):",
    ]
    for epoch, (loss, acc) in enumerate(feature_history, 1): lines.append(f"Эпоха {epoch}: loss={loss:.4f}, train accuracy={acc:.2%}")
    lines += [f"Test Accuracy={feature_result['accuracy']:.2%}", f"Test Precision={feature_result['precision']:.2%}", f"Test Recall={feature_result['recall']:.2%}", "", "Fine-tuning (разморожен layer4 и fc; уменьшен learning rate):"]
    for epoch, (loss, acc) in enumerate(fine_history, 3): lines.append(f"Эпоха {epoch}: loss={loss:.4f}, train accuracy={acc:.2%}")
    lines += [f"Test Accuracy={fine_result['accuracy']:.2%}", f"Test Precision={fine_result['precision']:.2%}", f"Test Recall={fine_result['recall']:.2%}", f"Изменение Accuracy: {(fine_result['accuracy'] - feature_result['accuracy']):+.2%}"]
    text = "\n".join(lines) + "\n"
    (RESULT_DIR / "09_transfer_learning.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

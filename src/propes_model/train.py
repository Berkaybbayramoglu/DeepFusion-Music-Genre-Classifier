import os
import sys
import argparse
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from propes_model.data import (
    load_features,
    preprocess_all_features,
    create_data_loaders,
)
from propes_model.models import MultiModalNet
from propes_model.utils import log_results_to_csv, ensure_dir
from collections import Counter
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate_model(model: nn.Module, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy


def plot_training_history(history, save_path):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs: int,
    model_save_dir: str,
    patience: int,
    results_csv_path: str,
):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0

    ensure_dir(model_save_dir)

    if os.path.exists(results_csv_path):
        os.remove(results_csv_path)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training")):
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation"):
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                labels = labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        scheduler.step(epoch_val_loss)

        log_data = {
            'epoch': epoch + 1,
            'train_loss': f"{epoch_train_loss:.4f}",
            'train_accuracy': f"{epoch_train_acc:.4f}",
            'val_loss': f"{epoch_val_loss:.4f}",
            'val_accuracy': f"{epoch_val_acc:.4f}",
            'learning_rate': f"{optimizer.param_groups[0]['lr']:.6f}",
        }
        log_results_to_csv(results_csv_path, log_data, mode='a', header=(epoch == 0))

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(model_save_dir, 'best_multi_modal_gtzan_model.pth'))
            print(f"Best model saved: Epoch {epoch+1}, Val Loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Early Stopping: Validation loss did not improve for {patience} epochs. Training stopped.")
                break

    return history


def parse_args():
    parser = argparse.ArgumentParser(description="Train multi-modal music genre classifier")
    parser.add_argument("--features", type=str, required=True, help="Path to features pickle file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--model-dir", type=str, default="./models", help="Directory to save best model and artifacts")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of classes")
    return parser.parse_args()


def main(argv: List[str] = sys.argv[1:]):
    args = parse_args()
    set_seed(42)

    all_features, all_labels, _ = load_features(args.features)
    (
        mel_orig_specs,
        mel_harm_specs,
        mel_perc_specs,
        mfccs,
        chromas,
        tempograms,
        labels,
    ) = preprocess_all_features(all_features, all_labels)

    train_loader, val_loader, y_train_full_labels = create_data_loaders(
        mel_orig_specs,
        mel_harm_specs,
        mel_perc_specs,
        mfccs,
        chromas,
        tempograms,
        labels,
        batch_size=args.batch_size,
        validation_split_ratio=args.val_split,
        random_seed=42,
    )

    class_counts = Counter(y_train_full_labels.tolist())
    total_samples = sum(class_counts.values())
    class_weights = torch.tensor([total_samples / (args.num_classes * class_counts[i]) for i in range(args.num_classes)], dtype=torch.float32).to(DEVICE)

    model = MultiModalNet(args.num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=max(1, args.patience - 2), verbose=True)

    history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs=args.epochs,
        model_save_dir=args.model_dir,
        patience=args.patience,
        results_csv_path="training_results_single_split.csv",
    )

    best_model_path = os.path.join(args.model_dir, 'best_multi_modal_gtzan_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)
        print(f"Final Validation Loss: {val_loss:.4f}")
        print(f"Final Validation Accuracy: {val_accuracy:.4f}")

    plot_training_history(history, save_path=os.path.join(args.model_dir, "training_history.png"))


if __name__ == "__main__":
    main()

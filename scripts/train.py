import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import sys

sys.path.append('/workspace/ML_Spring25') # add project dir to sys path for module imports

from datasets.fddb_dataset import FDDBFaceDataset
from models.face_classifier import FaceClassifier

# --- Config/Hyperparams ---
csv_path = 'data/fddb_crops/fddb_labels.csv'
image_dir = 'data/fddb_crops'
batch_size = 64
epochs = 30
learning_rate = 1e-3

# --- Load Dataset ---
dataset = FDDBFaceDataset(csv_file=csv_path, img_dir=image_dir)
num_classes = len(set(dataset.annotations['label']))  # dynamically count classes

# Train Val Split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

# --- Benchmark CPU vs GPU --- trains model twice (once on cpu, once on gpu)
training_times = {}
for device_type in ['cuda', 'cpu']:
    if device_type == 'cuda' and not torch.cuda.is_available():
        continue
    device = torch.device(device_type)
    print(f"\nTraining on: {device}")

    # model, loss, optimizer
    model = FaceClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    train_losses, val_losses, epoch_times = [], [], []
    start_time = time.time()

    # Training Loop:
    # - trains on each dataset
    # - validates on validation set
    # - measures epoch duration
    # - tracks --> train loss, validation loss, validation accuracy, epoch time
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_acc = correct / total

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2%}, Time: {epoch_time:.2f}s")

        # Saves best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'best_model_{device_type}.pth')

    end_time = time.time()
    total_train_time = end_time - start_time
    training_times[device_type] = total_train_time

    # Save Loss and Time Plots -- per device
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training vs Validation Loss ({device_type})')
    plt.legend()
    plt.savefig(f'loss_curve_{device_type}.png')
    plt.close()

    plt.figure()
    plt.plot(epoch_times, marker='o', linestyle='-', label='Epoch Time')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.title(f'Training Time per Epoch ({device_type})')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.title(f'Training Time per Epoch ({device_type})')
    plt.savefig(f'epoch_times_{device_type}.png')
    plt.close()


# --- Inference Time per Image ---
inference_times = {}
for device_type in ['cuda', 'cpu']:
    if device_type == 'cuda' and not torch.cuda.is_available():
        continue
    device = torch.device(device_type)
    model = FaceClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(f'best_model_{device_type}.pth', map_location=device))
    model.eval()

    sample_batch = next(iter(val_loader))[0].to(device)
    with torch.no_grad():
        start = time.time()
        _ = model(sample_batch)
        end = time.time()
    avg_infer_time = (end - start) / len(sample_batch)
    inference_times[device_type] = avg_infer_time

# --- Bar Chart: Inference Time per Image Comparison ---
plt.figure()
bars = plt.bar(inference_times.keys(), inference_times.values())
plt.ylabel('Time (s)')
plt.title('Inference Time per Image: GPU vs CPU')
for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height:.6f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
plt.savefig('inference_time_comparison.png')
plt.close()
plt.figure()
plt.bar(inference_times.keys(), inference_times.values())
plt.ylabel('Time (s)')
plt.title('Inference Time per Image: GPU vs CPU')
plt.savefig('inference_time_comparison.png')
plt.close()


# ---Bar Chart: Total Training Time Comparison ---
plt.figure()
bars = plt.bar(training_times.keys(), training_times.values())
plt.ylabel('Total Time (s)')
plt.title('Total Training Time: GPU vs CPU')
for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
plt.savefig('training_time_comparison.png')
plt.close()

plt.figure()
plt.bar(training_times.keys(), training_times.values())
plt.ylabel('Total Time (s)')
plt.title('Total Training Time: GPU vs CPU')
plt.savefig('training_time_comparison.png')
plt.close()

# Save training and inference times to CSV
with open('performance_metrics.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Device', 'Training Time (s)', 'Inference Time per Image (s)'])
    for device in training_times:
        writer.writerow([device, training_times[device], inference_times.get(device, 'N/A')])

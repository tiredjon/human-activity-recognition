import os
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# ===== CONFIG =====
train_csv = 'data/Training_set.csv'
test_csv = 'data/Testing_set.csv'
train_dir = 'data/train'
test_dir = 'data/test'
num_epochs = 5
batch_size = 32
img_size = 224
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# ===== CLASSES =====
class_names = ['calling', 'clapping', 'cycling', 'dancing', 'drinking',
               'eating', 'fighting', 'hugging', 'laughing', 'listening_to_music',
               'running', 'sitting', 'sleeping', 'texting', 'using_laptop']
label_encoder = LabelEncoder().fit(class_names)

# ===== TRANSFORM =====
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===== DATASET =====
class ActivityDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform, is_test=False):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        if not is_test:
            self.labels = label_encoder.transform(self.df['label'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['filename']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        if self.is_test:
            return image
        else:
            label = self.labels[idx]
            return image, label

# ===== MODEL =====
def get_model(num_classes):
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ===== TRAIN =====
def train(model, loader, criterion, optimizer):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {running_loss:.4f} | Accuracy: {acc:.2f}%")

# ===== PREDICT =====
def predict(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
    return [class_names[i] for i in preds]

# ===== MAIN =====
def main():
    train_set = ActivityDataset(train_csv, train_dir, transform, is_test=False)
    test_set = ActivityDataset(test_csv, test_dir, transform, is_test=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = get_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training started...")
    train(model, train_loader, criterion, optimizer)

    torch.save(model.state_dict(), "activity_model.pth")
    print("✅ Model saved as 'activity_model.pth'")

    print("Predicting test images...")
    predictions = predict(model, test_loader)

    for i, label in enumerate(predictions[:10]):
        print(f"Image {i+1}: {label}")

if __name__ == "__main__":
    main()
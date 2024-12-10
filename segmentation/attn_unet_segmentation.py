import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

dataset_path = 'anomaly_detection_dataset'
good_path = os.path.join(dataset_path, 'good')
bad_path = os.path.join(dataset_path, 'bad')
masks_path = os.path.join(dataset_path, 'masks')

image_size = (128, 128)
batch_size = 16
test_size = 0.2
val_size = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AnomalyDataset(Dataset):
    def __init__(self, images, masks, labels, transform=None):
        self.images = images
        self.masks = masks
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img).float()
            mask = self.transform(mask).float()
        
        return img, mask, label

def load_images_and_masks():
    images = []
    masks = []
    labels = []

    for img_file in os.listdir(good_path):
        img = Image.open(os.path.join(good_path, img_file)).resize(image_size)
        images.append(np.array(img))
        masks.append(np.zeros((*image_size, 1)))  # Black mask
        labels.append(0)

    for img_file in os.listdir(bad_path):
        img = Image.open(os.path.join(bad_path, img_file)).resize(image_size)
        mask = Image.open(os.path.join(masks_path, img_file)).resize(image_size).convert("L")
        images.append(np.array(img))
        masks.append(np.array(mask).reshape(*image_size, 1))
        labels.append(1)

    return np.array(images), np.array(masks), np.array(labels)

images, masks, labels = load_images_and_masks()
images = images / 255.0
masks = masks / 255.0

X_train, X_temp, y_train, y_temp, m_train, m_temp = train_test_split(images, labels, masks, test_size=(test_size + val_size), stratify=labels, random_state=42)
X_val, X_test, y_val, y_test, m_val, m_test = train_test_split(X_temp, y_temp, m_temp, test_size=test_size / (test_size + val_size), stratify=y_temp, random_state=42)

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = AnomalyDataset(X_train, m_train, y_train, transform=transform)
val_dataset = AnomalyDataset(X_val, m_val, y_val, transform=transform)
test_dataset = AnomalyDataset(X_test, m_test, y_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UNetWithAttention(nn.Module):
    def __init__(self):
        super(UNetWithAttention, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.attention = AttentionBlock(F_g=128, F_l=64, F_int=32)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.mask_output = nn.Conv2d(64, 1, kernel_size=1)
        self.class_output = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        enc = self.encoder(x)
        bottleneck = self.bottleneck(enc)
        attn = self.attention(bottleneck, enc)  
        dec = self.decoder(attn)               
        mask = torch.sigmoid(self.mask_output(dec))
        classification = self.class_output(bottleneck)
        return mask, classification

model = UNetWithAttention().to(device)
criterion_mask = nn.BCELoss()
criterion_class = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss, mask_loss, class_loss = 0, 0, 0
    for images, masks, labels in train_loader:
        images, masks, labels = images.to(device).float(), masks.to(device).float(), labels.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        pred_masks, pred_classes = model(images)
        loss_mask = criterion_mask(pred_masks, masks)
        loss_class = criterion_class(pred_classes, labels.unsqueeze(1))
        loss = loss_mask + loss_class

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        mask_loss += loss_mask.item()
        class_loss += loss_class.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Mask Loss: {mask_loss:.4f}, Class Loss: {class_loss:.4f}")

model.eval()
y_true, y_pred_class, y_pred_mask = [], [], []
with torch.no_grad():
    for images, masks, labels in test_loader:
        images, masks, labels = images.to(device).float(), masks.to(device).float(), labels.to(device, dtype=torch.float32)
        pred_masks, pred_classes = model(images)
        y_true.extend(labels.cpu().numpy())
        y_pred_class.extend((pred_classes.cpu().numpy() > 0.5).astype(int).flatten())
        y_pred_mask.append(pred_masks.cpu().numpy())

model_save_path = r"models\atten_unet_segmentation.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved at {model_save_path}")

print("Classification Report:")
print(classification_report(y_true, y_pred_class, target_names=["Good", "Bad"]))
class_cm = confusion_matrix(y_true, y_pred_class)
sns.heatmap(class_cm, annot=True, fmt="d", cmap="Blues")
plt.title("Classification Confusion Matrix")
plt.show()

def show_predictions(model, dataloader, num_samples=5):
    model.eval()
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, num_samples * 4))
    with torch.no_grad():
        for idx, (images, masks, labels) in enumerate(dataloader):
            images, masks, labels = images.to(device).float(), masks.to(device).float(), labels.to(device, dtype=torch.float32)
            pred_masks, pred_classes = model(images)
            
            for i in range(min(num_samples, len(images))):
                input_img = images[i].cpu().permute(1, 2, 0).numpy()
                actual_mask = masks[i].cpu().squeeze().numpy()
                predicted_mask = pred_masks[i].cpu().squeeze().numpy()
                actual_label = "Bad" if labels[i].item() == 1 else "Good"
                predicted_label = "Bad" if pred_classes[i].item() > 0.5 else "Good"

                axes[i, 0].imshow(input_img)
                axes[i, 0].set_title("Input Image")
                axes[i, 0].axis("off")

                axes[i, 1].imshow(actual_mask, cmap="gray")
                axes[i, 1].set_title("Actual Mask")
                axes[i, 1].axis("off")

                axes[i, 2].imshow(predicted_mask, cmap="gray")
                axes[i, 2].set_title("Predicted Mask")
                axes[i, 2].axis("off")

                axes[i, 3].text(0.5, 0.5, actual_label, fontsize=12, ha='center', va='center', color="black")
                axes[i, 3].set_title("Actual Label")
                axes[i, 3].axis("off")

                axes[i, 4].text(0.5, 0.5, predicted_label, fontsize=12, ha='center', va='center', color="black")
                axes[i, 4].set_title("Predicted Label")
                axes[i, 4].axis("off")

            if idx + 1 >= num_samples:
                break

    plt.tight_layout()
    plt.show()

show_predictions(model, test_loader, num_samples=5)
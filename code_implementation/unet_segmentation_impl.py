import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

model_path = r"models\unet_segmentation.pth"
image_size = (128, 128)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
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
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
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
        dec = self.decoder(bottleneck)
        mask = torch.sigmoid(self.mask_output(dec))
        classification = self.class_output(bottleneck)
        return mask, classification

model = UNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_mask, pred_class = model(input_tensor)
    
    pred_mask = pred_mask.squeeze().cpu().numpy()
    pred_class = "Bad" if pred_class.item() > 0.5 else "Good"

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.text(0.5, 0.5, pred_class, fontsize=18, ha='center', va='center')
    plt.title("Predicted Class")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

test_image_path = r"path\to\test\image.jpg"  
predict_image(test_image_path)
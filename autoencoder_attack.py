import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import matplotlib.pyplot as plt


class AttackAutoencoder(nn.Module):
    def __init__(self):
        super(AttackAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_attacker(model, watermarked_img_path, original_img_path, epochs=100):
    wm_img = cv2.imread(watermarked_img_path, 0).astype(np.float32) / 255.0
    orig_img = cv2.imread(original_img_path, 0).astype(np.float32) / 255.0
    
    wm_img = cv2.resize(wm_img, (512, 512))
    orig_img = cv2.resize(orig_img, (512, 512))

    input_tensor = torch.tensor(wm_img).unsqueeze(0).unsqueeze(0)
    target_tensor = torch.tensor(orig_img).unsqueeze(0).unsqueeze(0)

    # Eğitim Ayarları
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(input_tensor)
        
        loss = criterion(output, target_tensor)
        
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    
    # Saldırı Sonucu (Temizlenmiş Resim)
    model.eval()
    with torch.no_grad():
        cleaned_tensor = model(input_tensor)
        cleaned_img = cleaned_tensor.squeeze().numpy() * 255.0
        cleaned_img = np.clip(cleaned_img, 0, 255).astype(np.uint8)
        
    return cleaned_img
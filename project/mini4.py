import torch
import transformers
from torch import nn
from torch.utils.data import DataLoader
import os
import pandas as pd
from torchvision import transforms

# 영상 파일이 있는 디렉토리
video_directory = 'D:/minipro//'

# 디렉토리 내 파일 목록 가져오기
file_list = os.listdir(video_directory)

# text 값 가져오기
text = pd.read_csv('KETI-2017-SL-Annotation-v2_1.csv')

# y 라벨 값 때기
text = text['한국어']


# Define Vision Transformer (ViT) model
class VisionTransformer(nn.Module):
    def __init__(self, num_classes, dim, num_heads, num_layers, hidden_dim, dropout):
        super(VisionTransformer, self).__init__()
        self.patch_embed = nn.Conv3d(3, dim, kernel_size= 16 , stride= 2 )
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout
            ),
            num_layers=num_layers
        )
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)  # (B, dim, H // 16, W // 16)
        x = x.flatten(2).permute(2, 0, 1)  # (H*W/(16*16), B, dim)
        x = self.transformer_encoder(x)  # (seq_len, B, dim)
        x = x.mean(0)  # (B, dim)
        x = self.fc(x)  # (B, num_classes)
        return x

# Hyperparameters
num_classes = 10
dim = 256
num_heads = 8
num_layers = 6
hidden_dim = 512
dropout = 0.1
batch_size = 64
epochs = 10

transform = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# train_dataset = file_list(root='D:/minipro//', train=True, download=True, transform=transform)
train_loader = DataLoader(file_list, batch_size=batch_size, shuffle=True)

# Initialize ViT model
model = VisionTransformer(num_classes, dim, num_heads, num_layers, hidden_dim, dropout)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')
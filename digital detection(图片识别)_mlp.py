import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 320)
        x = self.fc(x)
        return x

# 加载模型
model=Net()

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 读取和处理图像
image= Image.open('5.jpg')
image = transform(image)
image = image.unsqueeze(0)


with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output.data, 1)
print('The predicted number is:', predicted.item())
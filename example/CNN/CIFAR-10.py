import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from dwmgrad.dwmg import DynamicWinMoAdaGrad

# 定义CNN网络结构（适用于CIFAR-10）
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 初始化网络
# model = CIFAR10CNN()
model = CIFAR10CNN().to(device)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 数据加载和预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 初始化自定义优化器
optimizer = DynamicWinMoAdaGrad(lr=0.0001, max_window_size=100)

# 用于记录损失的列表
losses = []

# 初始化跟踪最佳准确率的变量
best_accuracy = 0
best_epoch = 0

# 训练模型
with open('cifar-our3-drop.txt', 'w') as f:
    for epoch in range(100):  # 假设训练10个epoch
        model.train()  # 设置模型为训练模式
        epoch_loss = 0
        last_loss = 0

        for data, target in tqdm(train_loader):
            data, target = data.to(device), target.to(device)  # 将数据和目标转移到GPU

            # 清除之前的梯度
            model.zero_grad()

            # 前向传播
            output = model(data)
            loss = criterion(output, target)

            # 反向传播
            loss.backward()

            # 获取梯度，并将它们从 CUDA 设备移动到 CPU
            grads = [param.grad.cpu().numpy() for param in model.parameters() if param.grad is not None]

            # 获取当前参数
            current_params = [param.data.cpu().numpy() for param in model.parameters()]

            # 更新优化器
            optimizer.update(current_params, grads, loss.item() - last_loss)

            # 将当前损失值保存为下一次迭代的 last_loss
            last_loss = loss.item()

            # 将更新后的参数复制回模型
            with torch.no_grad():
                for param, updated_param in zip(model.parameters(), current_params):
                    param.copy_(torch.from_numpy(updated_param).to(device))

            # 更新损失
            epoch_loss += loss.item()
            last_loss = loss.item()
            # optimizer.print_window_sizes()

        # 计算平均损失
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        # 模型评估
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for data, target in tqdm(test_loader):
                data, target = data.to(device), target.to(device)  # 将数据和目标转移到GPU

                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        f.write(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy}%\n')
        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy}%')

# 绘制损失曲线
plt.plot(losses, label='Training Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 保存模型
torch.save(model.state_dict(), 'cifar10_cnn_model.pth')

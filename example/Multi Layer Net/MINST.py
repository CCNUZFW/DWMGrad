import torch
import torch.nn as nn
from torch.nn.functional import relu
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dwmgrad.dwmg import DynamicWinMoAdaGrad


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        # self.dropout1 = nn.Dropout(0.5)  # Dropout层
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = relu(self.fc1(x))
        # x = self.dropout1(x)
        x = self.fc2(x)
        return x


# 初始化网络
model = SimpleNet()

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 数据加载和预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 初始化自定义优化器
optimizer = DynamicWinMoAdaGrad(lr=0.001, max_window_size=1000)

# 用于记录损失的列表
losses = []

# 初始化跟踪最佳准确率的变量
best_accuracy = 0
best_epoch = 0

# 训练模型
with open('our-drop1.txt', 'w') as f:
    for epoch in range(100):  # 训练10个epoch
        model.train()
        epoch_loss = 0
        last_loss = 0
        for data, target in train_loader:
            # 前向传播
            output = model(data)
            loss = criterion(output, target)

            # 反向传播和优化
            model.zero_grad()
            loss.backward()

            # 获取梯度
            grads = [param.grad.numpy() for param in model.parameters() if param.grad is not None]

            # 获取当前参数
            current_params = [param.data.numpy() for param in model.parameters()]

            # 更新参数
            optimizer.update(current_params, grads, loss.item() - last_loss)

            # 将更新后的参数复制回模型
            with torch.no_grad():
                for param, updated_param in zip(model.parameters(), current_params):
                    param.copy_(torch.from_numpy(updated_param))

            epoch_loss += loss.item()
            last_loss = loss.item()

        # 计算平均损失
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        # 模型评估
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        # f.write(f'Epoch {epoch + 1}, Loss: {avg_loss:.8f}, Accuracy: {accuracy}%\n')
        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.8f}, Accuracy: {accuracy}%')

        # 更新最佳准确率
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch + 1

    # 输出最佳准确率和对应的epoch
    # f.write(f'Best Accuracy: {best_accuracy}% achieved at Epoch {best_epoch}\n')
    print(f'Best Accuracy: {best_accuracy}% achieved at Epoch {best_epoch}')

# 绘制损失曲线
plt.plot(losses, label='Training Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 保存模型
torch.save(model.state_dict(), 'simple_mnist_model.pth')

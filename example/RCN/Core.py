import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
from dwmgrad.dwmg import DynamicWinMoAdaGradv3
import torch.optim as optim


# 计算准确率的函数
def compute_accuracy(model, data):
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    return correct / int(data.test_mask.sum())


# 定义图神经网络模型
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# 加载Cora数据集
dataset = Planetoid(root=r'', name='Cora')

# 实例化模型和优化器
model_dynamic_adagrad = GCN()
optimizer_dynamic_adagrad = DynamicWinMoAdaGradv3(model_dynamic_adagrad.parameters(), lr=0.001)

model_adam = GCN()
optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=0.001)

# 训练模型并收集损失和准确率
losses_dynamic_adagrad = []
accuracies_dynamic_adagrad = []
losses_adam = []
accuracies_adam = []

for epoch in range(100):
    # DynamicAdaGrad
    model_dynamic_adagrad.train()
    optimizer_dynamic_adagrad.zero_grad()
    out_dynamic_adagrad = model_dynamic_adagrad(dataset[0])
    loss_dynamic_adagrad = F.nll_loss(out_dynamic_adagrad[dataset[0].train_mask], dataset[0].y[dataset[0].train_mask])
    loss_dynamic_adagrad.backward()
    optimizer_dynamic_adagrad.step(lambda: loss_dynamic_adagrad)
    losses_dynamic_adagrad.append(loss_dynamic_adagrad.item())
    train_accuracy_dynamic_adagrad = compute_accuracy(model_dynamic_adagrad, dataset[0])
    accuracies_dynamic_adagrad.append(train_accuracy_dynamic_adagrad)

    # Adam
    model_adam.train()
    optimizer_adam.zero_grad()
    out_adam = model_adam(dataset[0])
    loss_adam = F.nll_loss(out_adam[dataset[0].train_mask], dataset[0].y[dataset[0].train_mask])
    loss_adam.backward()
    optimizer_adam.step()
    losses_adam.append(loss_adam.item())
    train_accuracy_adam = compute_accuracy(model_adam, dataset[0])
    accuracies_adam.append(train_accuracy_adam)

    if epoch % 10 == 0:
        print(
            f'Epoch {epoch}: Loss DynamicAdaGrad: {loss_dynamic_adagrad.item()}, Training Accuracy: {train_accuracy_dynamic_adagrad:.8f}')
        print(f'Epoch {epoch}: Loss Adam: {loss_adam.item()}, Training Accuracy: {train_accuracy_adam:.8f}')

# 绘制损失曲线
plt.figure(figsize=(6, 5))
# plt.subplot(1, 2, 1)
plt.plot(losses_dynamic_adagrad, label='DynamicAdaGrad Loss')
plt.plot(losses_adam, label='Adam Loss', linestyle='--')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.jpg', dpi=600)

# 绘制准确率曲线
plt.figure(figsize=(6, 5))
# plt.subplot(1, 2, 2)
plt.plot(accuracies_dynamic_adagrad, label='DynamicAdaGrad Accuracy')
plt.plot(accuracies_adam, label='Adam Accuracy', linestyle='--')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('acc.jpg', dpi=600)

plt.show()

# 测试模型
test_accuracy_dynamic_adagrad = compute_accuracy(model_dynamic_adagrad, dataset[0])
test_accuracy_adam = compute_accuracy(model_adam, dataset[0])
print(f'Test Accuracy DynamicAdaGrad: {test_accuracy_dynamic_adagrad:.4f}')
print(f'Test Accuracy Adam: {test_accuracy_adam:.4f}')

# 保存模型
torch.save(model_dynamic_adagrad.state_dict(), 'cora_gnn_model_dynamic_adagrad.pth')
torch.save(model_adam.state_dict(), 'cora_gnn_model_adam.pth')

print("模型已保存至 '/mnt/data/cora_gnn_model_dynamic_adagrad.pth'")
print("模型已保存至 '/mnt/data/cora_gnn_model_adam.pth'")

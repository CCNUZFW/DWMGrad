import numpy as np
import torch.optim as optim
import torch

class DynamicWinMoAdaGrad:
    def __init__(self, lr=0.001, base_momentum=0.9, epsilon=1e-8, initial_window_size=5, max_window_size=1000,
                 loss_threshold=0.1, oscillation_threshold=2):
        self.lr = lr
        self.base_momentum = base_momentum
        self.epsilon = epsilon
        self.initial_window_size = initial_window_size
        self.max_window_size = max_window_size
        self.grad_squares = {}
        self.window_sizes = {}
        self.velocity = {}
        self.loss_diff_accumulator = {}
        self.loss_history = []  # 跟踪损失历史
        self.loss_threshold = loss_threshold  # 损失变化阈值
        self.oscillation_threshold = oscillation_threshold  # 震荡阈值

    def update(self, params, grads, loss):
        # 更新损失历史
        self.loss_history.append(loss)
        if len(self.loss_history) > self.oscillation_threshold:
            self.loss_history.pop(0)

        # 检测震荡
        if self._detect_oscillation():
            self.lr *= 0.5  # 减少学习率以抑制震荡

        for i, param in enumerate(params):
            if i not in self.grad_squares:
                self.grad_squares[i] = np.zeros_like(param)
                self.window_sizes[i] = self.initial_window_size
                self.velocity[i] = np.zeros_like(param)
                self.loss_diff_accumulator[i] = 0  # 初始化累积器

            # 累积损失差
            self.loss_diff_accumulator[i] += loss

            # 根据累积的损失差调整窗口大小
            if self.loss_diff_accumulator[i] > 0:
                self.window_sizes[i] = min(self.window_sizes[i] + 1, self.max_window_size)
                self.loss_diff_accumulator[i] = 0  # 重置累积器
            elif self.loss_diff_accumulator[i] <= 0:
                self.window_sizes[i] = max(self.window_sizes[i] - 1, 1)
                self.loss_diff_accumulator[i] = 0  # 重置累积器

            # 使用滑动窗口方法更新梯度平方
            self.grad_squares[i] = self.grad_squares[i] * (self.window_sizes[i] - 1) / self.window_sizes[i]
            self.grad_squares[i] += grads[i] ** 2 / self.window_sizes[i]

            # 根据窗口大小调整动量
            adjusted_momentum = self.base_momentum * self.window_sizes[i] / self.max_window_size

            # 使用调整后的动量更新速度
            self.velocity[i] = adjusted_momentum * self.velocity[i] + self.lr * grads[i] / (
                        np.sqrt(self.grad_squares[i]) + self.epsilon)

            # 使用速度更新参数
            params[i] -= self.velocity[i]

    def _detect_oscillation(self):
        # 检测损失震荡
        oscillations = 0
        for i in range(1, len(self.loss_history)):
            if abs(self.loss_history[i] - self.loss_history[i - 1]) > self.loss_threshold:
                oscillations += 1

        return oscillations >= self.oscillation_threshold

    def print_window_sizes(self):
        # 打印当前的窗口大小
        for i, window_size in self.window_sizes.items():
            print(f"Param {i}: Window Size = {window_size}")

class DynamicWinMoAdaGradv2:
    def __init__(self, lr=0.01, base_momentum=0.9, epsilon=1e-8, initial_window_size=5, max_window_size=15):
        self.lr = lr
        self.base_momentum = base_momentum
        self.epsilon = epsilon
        self.initial_window_size = initial_window_size
        self.max_window_size = max_window_size
        self.grad_squares = {}
        self.window_sizes = {}
        self.velocity = {}

    def update(self, params, grads, loss_diff):
        for i, param in enumerate(params):
            if i not in self.grad_squares:
                self.grad_squares[i] = np.zeros_like(param)
                self.window_sizes[i] = self.initial_window_size
                self.velocity[i] = np.zeros_like(param)

            # Update window size based on loss difference
            if loss_diff > 0:
                self.window_sizes[i] = min(self.window_sizes[i] + 1, self.max_window_size)
            else:
                self.window_sizes[i] = max(self.window_sizes[i] - 1, 1)

            # Update grad_squares with sliding window approach
            self.grad_squares[i] = self.grad_squares[i] * (self.window_sizes[i] - 1) / self.window_sizes[i]
            self.grad_squares[i] += grads[i] ** 2 / self.window_sizes[i]

            # Adjust momentum based on window size
            adjusted_momentum = self.base_momentum * self.window_sizes[i] / self.max_window_size

            # Update velocity with adjusted momentum
            self.velocity[i] = adjusted_momentum * self.velocity[i] + self.lr * grads[i] / (np.sqrt(self.grad_squares[i]) + self.epsilon)

            # Update parameters using velocity
            params[i] -= self.velocity[i]

class DynamicWinMoAdaGradv3(optim.Optimizer):
    def __init__(self, params, lr=0.001, base_momentum=0.9, epsilon=1e-8,
                 initial_window_size=5, max_window_size=10, loss_threshold=0.1,
                 oscillation_threshold=2):
        defaults = dict(lr=lr, base_momentum=base_momentum, epsilon=epsilon,
                        initial_window_size=initial_window_size, max_window_size=max_window_size,
                        loss_threshold=loss_threshold, oscillation_threshold=oscillation_threshold)
        super(DynamicWinMoAdaGradv3, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['grad_square'] = torch.zeros_like(p.data)
                    state['window_size'] = group['initial_window_size']
                    state['velocity'] = torch.zeros_like(p.data)
                    state['loss_diff_accumulator'] = 0

                state['step'] += 1
                grad_square = state['grad_square']
                window_size = state['window_size']
                velocity = state['velocity']
                loss_diff_accumulator = state['loss_diff_accumulator']

                # Dynamic window size adjustment
                loss_diff_accumulator += loss.item()
                if loss_diff_accumulator > 0:
                    window_size = min(window_size + 1, group['max_window_size'])
                    loss_diff_accumulator = 0
                elif loss_diff_accumulator <= 0:
                    window_size = max(window_size - 1, 1)
                    loss_diff_accumulator = 0
                state['window_size'] = window_size
                state['loss_diff_accumulator'] = loss_diff_accumulator

                # Update grad_square
                grad_square = grad_square * (window_size - 1) / window_size + grad**2 / window_size
                state['grad_square'] = grad_square

                # Adjusted momentum
                adjusted_momentum = group['base_momentum'] * window_size / group['max_window_size']

                # Update velocity
                velocity = adjusted_momentum * velocity + group['lr'] * grad / (grad_square.sqrt() + group['epsilon'])
                state['velocity'] = velocity

                # Update parameters
                p.data.add_(-velocity)

        return loss
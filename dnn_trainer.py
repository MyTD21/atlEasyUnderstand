import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
import numpy as np

from torch.optim.optimizer import Optimizer, required
import math

# DNN模型定义
class DNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, output_size=1):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层到隐藏层
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)  # 隐藏层到隐藏层
        self.fc3 = nn.Linear(hidden_size // 2, output_size)  # 隐藏层到输出层

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # 回归问题，输出层不使用激活函数
        return x


class MyMSELoss(nn.Module): # 自定义均方误差损失类，接口和功能与nn.MSELoss完全一致,支持直接替换代码中的nn.MSELoss
    def __init__(self, reduction: str = "mean") -> None:
        """
        初始化损失函数
        reduction: 损失聚合方式，可选值：mean, sum, none
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Args:
            y_pred: 模型预测值张量，形状需与y_true一致
            y_true: 真实值（ground truth）张量，形状需与y_pred一致

        Returns:
            聚合后的损失值（或原始平方误差张量）

        """
        # 检查输入形状是否匹配
        if y_pred.shape != y_true.shape:
            raise RuntimeError(f"预测值形状 {y_pred.shape} 与真实值形状 {y_true.shape} 不匹配")

        # 计算每个元素的平方误差：(y_pred - y_true)²
        squared_error = (y_pred - y_true) **2

        # 根据初始化时的reduction参数聚合结果
        if self.reduction == "mean":
            return squared_error.mean()
        elif self.reduction == "sum":
            return squared_error.sum()
        else:  # 'none'
            return squared_error

class MyAdam(Optimizer): # 自定义Adam优化器，完全兼容PyTorch原生optim.Adam的接口和行为, 论文：https://arxiv.org/abs/1412.6980v8
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0): # 初始化
        """
        Args:
            params: 待优化的参数迭代器（如model.parameters()）
            lr: 学习率（默认1e-3）
            betas: 一阶矩和二阶矩的指数衰减率（默认(0.9, 0.999)）
            eps: 数值稳定性常数，默认1e-8,分母中加入一个很小的数，避免除零
            weight_decay: 权重衰减（L2正则化）系数（默认0）
        """
        # 构造超参数字典
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(MyAdam, self).__init__(params, defaults)

    def zero_grad1(self, set_to_none: bool = False) -> None: # 显式实现梯度清零：清除所有参数的梯度
        # 参数 set_to_none: 若为True，将param.grad设为None（更高效，不占用内存）; 若为False，将param.grad清零（保持张量结构）
        # import pdb; pdb.set_trace()
        for group in self.param_groups: # 遍历所有参数组
            for p in group['params']: # 遍历组内每个参数
                if p.grad is not None:  # 仅处理有梯度的参数
                    p.grad.zero_()

    def step1(self, closure=None):
        # import pdb; pdb.set_trace()
        # 遍历所有参数组（支持多参数组配置）
        for group in self.param_groups:
            # 获取当前组的超参数
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for p in group['params']: # 遍历组内所有参数
                if p.grad is None:
                    continue  # 无梯度的参数跳过更新
                grad = p.grad.data  # 获取梯度数据

                # 初始化参数状态（一阶矩、二阶矩、时间步）
                state = self.state[p]
                if len(state) == 0: # 初始状态，step=0，一阶矩和二阶矩都是0;
                    state['step'] = 0  
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1  # 时间步+1
                t = state['step']

                # 权重衰减（L2正则化）：等价于参数先乘以(1 - lr*weight_decay)
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1) # 更新一阶矩：exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2) # 更新二阶矩：exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2

                # 偏差修正：由于初始时刻矩的估计值偏向0，需要修正
                # 修正后的一阶矩：exp_avg / (1 - beta1^t)
                # 修正后的二阶矩：exp_avg_sq / (1 - beta2^t)
                bias_correction1 = 1 - beta1 **t
                bias_correction2 = 1 - beta2 **t
                step_size = lr / bias_correction1  # 修正后的学习率步长

                # 计算参数更新：param = param - step_size * exp_avg / (sqrt(exp_avg_sq) + eps)
                denom = exp_avg_sq.sqrt().add_(eps)  # 分母：sqrt(二阶矩修正值) + eps
                p.data.addcdiv_(exp_avg, denom, value=-step_size)


class MySGD(Optimizer):
    """
    自定义SGD优化器，支持动量和权重衰减，与torch.optim.SGD接口和功能完全一致

    参数:
        params (iterable): 待优化的参数迭代器（如model.parameters()）
        lr (float): 学习率（必填，无默认值）
        momentum (float, 可选): 动量因子，范围[0, 1)，默认0（不使用动量）
        weight_decay (float, 可选): 权重衰减系数（L2正则化），默认0
    """
    def __init__(self, params, lr=required, momentum=0, weight_decay=0):
        # 检查超参数合法性,已省略
        # 包装超参数（与PyTorch原生SGD保持一致的参数结构）
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(MySGD, self).__init__(params, defaults)

    def zero_grad1(self):
        super(MySGD, self).zero_grad()

    def step1(self, closure=None):
        # 遍历所有参数组（支持不同参数设置不同学习率/动量等）
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']

            for p in group['params']: # 遍历组内每个参数
                if p.grad is None:
                    continue  # 无梯度的参数跳过更新
 
                grad = p.grad.data.detach() # 获取参数梯度（分离计算图，避免修改原梯度）
                param = p.data  # 参数值（不含梯度信息）

                if weight_decay != 0: # 应用权重衰减（L2正则化）：梯度 += weight_decay * 参数值
                    grad.add_(param, alpha=weight_decay)

                # 应用动量：v = momentum * v_prev + grad（v为动量缓存）
                if momentum != 0:
                    # 初始化动量缓存（存储在self.state中，以参数p为键）
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        # 首次迭代：动量缓存 = 梯度（无历史值）
                        buf = state['momentum_buffer'] = torch.clone(grad).detach()
                    else:
                        # 非首次迭代：动量 = 动量因子 * 历史缓存 + 当前梯度
                        buf = state['momentum_buffer']
                        buf.mul_(momentum).add_(grad)  # buf = momentum * buf + grad

                    grad = buf # 用动量缓存替代原始梯度进行更新

                # 核心更新公式：param = param - lr * grad
                param.add_(grad, alpha=-lr)

class DnnTrainer():
    def __init__(self, model, epochs, ):
        self.model = model

        # 训练配置
        #self.criterion = nn.MSELoss()  # 系统自带mean squared error，均方误差损失，适用于回归问题
        self.criterion = MyMSELoss()  # 均方误差损失，适用于回归问题

        #self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.optimizer = MyAdam(model.parameters(), lr=0.001)
        #self.optimizer = MySGD(model.parameters(), lr=0.0001, momentum=0.0,)

        self.epochs = epochs

    def show_backward_effect(self):
        x = torch.tensor([2.0], requires_grad=False)  # 输入（不需要梯度）
        w = torch.tensor([3.0], requires_grad=True)   # 权重（需要梯度）
        b = torch.tensor([1.0], requires_grad=True)   # 偏置（需要梯度）

        y_pred = w * x + b # 前向传播
        loss = (y_pred - 10) **2  #计算loss 此时loss = (3*2 + 1 - 10)^2 = (7-10)^2 = 9
        import pdb; pdb.set_trace()

        print(f"w.grad : {w.grad}, b.grad : {b.grad}")

        # 正向传播时候，计算图会记录输出传输关系，x → (×w) → ( +b ) → y_pred → (MSE) → loss
        # 调用 loss.backward() 时, PyTorch 会从损失值 loss 出发, 沿着计算图反向遍历, 根据链式法则, 依次计算损失对每个中间变量(如 y_pred、w*x)和参数(w,b)的梯度。
        # 计算得到的梯度，会被存储于变量的 grad 属性中；
        loss.backward()

        print(f"w.grad : {w.grad}, b.grad : {b.grad}")

        print(loss)

    def train(self, train_loader, test_loader):
        train_losses = []
        test_losses = []

        # 训练循环
        for epoch in range(self.epochs):
            self.model.train()
            total_train_loss = 0
    
            # 训练阶段
            for batch_x, batch_y in train_loader:
                self.optimizer.zero_grad1()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                #self.show_backward_effect()
                self.optimizer.step1()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # 测试阶段
            self.model.eval()
            total_test_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                    total_test_loss += loss.item() # loss.item()提取loss的数值

            avg_test_loss = total_test_loss / len(test_loader)
            test_losses.append(avg_test_loss)

            if epoch % 50 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], 训练损失: {avg_train_loss:.4f}, 测试损失: {avg_test_loss:.4f}')

    def predict(self, x_values): # 使用训练好的模型进行预测
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor([x_values], dtype=torch.float32).view(-1, 1)
            predictions = self.model(x_tensor)
            return predictions.numpy().flatten()


def generate_data(num_samples=1000, x_range=(-10, 10), noise_std=0.5): # 生成模拟数据：y = 3x² + 5 + 噪声
    x = np.random.uniform(x_range[0], x_range[1], num_samples)
    y = 3 * x**2 + 5 + np.random.normal(0, noise_std, num_samples)
    return x, y

def get_data_loader():
    # 生成训练数据
    x_data, y_data = generate_data(num_samples=1000)
    x_tensor = torch.tensor(x_data, dtype=torch.float32).view(-1, 1)  # 转换为列向量
    y_tensor = torch.tensor(y_data, dtype=torch.float32).view(-1, 1)

    # 分割训练集和测试集
    train_size = int(0.8 * len(x_tensor))
    test_size = len(x_tensor) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        list(zip(x_tensor, y_tensor)), [train_size, test_size]
    )

    # 创建数据加载器
    batch_size = 32
    #batch_size = 1 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader

if __name__ == '__main__':
    
    # 构造训练数据
    train_loader, test_loader = get_data_loader()

    # 初始化模型
    dnn_model = DNN(input_size=1, hidden_size=128, output_size=1)

    # 初始化训练器
    trainer = DnnTrainer(model = dnn_model, epochs = 200)

    print("开始训练DNN模型...")
    trainer.train(train_loader = train_loader, test_loader = test_loader)
    print("训练完成!")

    # 测试
    print("\n=== 预测测试 ===")
    #test_points = [-5, -2, 0, 2, 5]
    test_points = np.random.uniform(-10, 10, 5)
    print("x值\tgroundtruth\t\tpredict\t\tgap")
    for x in test_points:
        true_y = 3 * x**2 + 5
        pred_y = trainer.predict(x)[0]
        print(f"{x:.2f}\t{true_y:.2f}\t\t\t{pred_y:.2f}\t\t{true_y-pred_y:.2f}")


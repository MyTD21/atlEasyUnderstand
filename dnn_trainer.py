import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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

class DnnTrainer():
    def __init__(self, model, epochs, ):
        self.model = model

        # 训练配置
        self.criterion = nn.MSELoss()  # 均方误差损失，适用于回归问题
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.epochs = epochs

    def train(self):
        train_losses = []
        test_losses = []

        # 训练循环
        for epoch in range(self.epochs):
            model.train()
            total_train_loss = 0
    
            # 训练阶段
            for batch_x, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # 测试阶段
            model.eval()
            total_test_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    outputs = model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                    total_test_loss += loss.item()

            avg_test_loss = total_test_loss / len(test_loader)
            test_losses.append(avg_test_loss)

            if epoch % 50 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], 训练损失: {avg_train_loss:.4f}, 测试损失: {avg_test_loss:.4f}')

    def predict(self, x_values): # 使用训练好的模型进行预测
        model.eval()
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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader

if __name__ == '__main__':
    
    # 构造训练数据
    train_loader, test_loader = get_data_loader()

    # 初始化模型
    model = DNN(input_size=1, hidden_size=128, output_size=1)

    # 初始化训练器
    trainer = DnnTrainer(model = model, epochs = 200)

    print("开始训练DNN模型...")
    trainer.train()
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


import torch

x = torch.tensor([2.0], requires_grad=False)  # 输入（不需要梯度）
w = torch.tensor([3.0], requires_grad=True)   # 权重（需要梯度）
b = torch.tensor([1.0], requires_grad=True)   # 偏置（需要梯度）

# 前向传播：y_pred = w*x + b
y_pred = w * x + b  
# 计算损失（假设真实值y_true=10）
loss = (y_pred - 10) **2  # 此时loss = (3*2 + 1 - 10)^2 = (7-10)^2 = 9
#import pdb; pdb.set_trace()

print("==== a case show grad")

print(f"w.grad : {w.grad}, b.grad : {b.grad}")

# 正向传播时候，计算图会记录输出传输关系，x → (×w) → ( +b ) → y_pred → (MSE) → loss
# 调用 loss.backward() 时, PyTorch 会从损失值 loss 出发, 沿着计算图反向遍历, 根据链式法则, 依次计算损失对每个中间变量(如 y_pred、w*x)和参数(w,b)的梯度。
# 计算得到的梯度，会被存储于变量的 grad 属性中；
loss.backward()

print(f"w.grad : {w.grad}, b.grad : {b.grad}")

print(loss)

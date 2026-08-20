import torch


learning_rates = [0.1, 0.5, 1.0, 1.1]

for learning_rate in learning_rates:
    w = torch.tensor(3.0, requires_grad=True)

    print(f"\nLearning rate: {learning_rate}")

    for step in range(6):
        # TODO 1：计算 loss = (w - 1)²
        loss = (w-1)**2

        # TODO 2：反向传播，计算 w.grad
        loss.backward()

        print(
            f"Step {step}: "
            f"w = {w.item():.4f}, "
            f"loss = {loss.item():.4f}, "
            f"gradient = {w.grad.item():.4f}"
        )

        # 更新参数时不需要建立新的计算图
        with torch.no_grad():
            # TODO 3：按照SGD公式更新w
            w -= learning_rate*  w.grad

        # 清除旧梯度
        w.grad.zero_()
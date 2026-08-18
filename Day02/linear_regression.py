import torch
import torch.nn as nn

#training data

x_train = torch.tensor(
    [[1.0],
     [2.0],
     [3.0],
     [4.0]]
)

y_train = torch.tensor(
    [[2.0],
     [4.0],
     [6.0],
     [8.0]]
)

#model

model = nn.Linear(
    1,
    1
)

#loss function

loss_function = nn.MSELoss()

#optimizer

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

#training
for epoch in range(1000):
    preediction = model(x_train)

    loss = loss_function(
        preediction,
        y_train
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 100 == 0:
        print(
            epoch,
            loss.item()
        )

print("weight:")
print(model.weight)

print("bias:")
print(model.bias)
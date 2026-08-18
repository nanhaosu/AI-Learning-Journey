import torch

#data

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

#parameters

w = torch.tensor(
    [[0.5]],
    requires_grad=True
)

b = torch.tensor(
    [0.0],
    requires_grad=True
)

#training

learning_rate = 0.01

for epoch in range(1000):

    #forward

    y_pred = x_train @ w + b

    #loss

    loss = ((y_pred-y_train)**2).mean()

    #backward

    loss.backward()

    #update

    with torch.no_grad():
        w -= learning_rate*w.grad

        b -= learning_rate*b.grad

    #clear gradient

    w.grad.zero_()

    b.grad.zero_()

    if epoch % 100 ==0:

        print(epoch,loss.item())

print(w)

print(b)

import torch
from matplotlib import pyplot as plt
import torch.optim as optim


def model(t_u, w, b):
    return w*t_u + b


def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()


def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    w, b = params
    w.requires_grad = True
    b.requires_grad = True
    opt = optim.SGD([w,b], lr=learning_rate)

    losses = []
    for epoch in range(1, n_epochs + 1):
        opt.zero_grad()
        t_p = model(t_u, w, b)
        loss = loss_fn(t_p, t_c)
        losses.append(loss.item())
        loss.backward()
        opt.step()
        print("Epoch %d, Loss %f" % (epoch, float(loss)))
        print("Params:{}".format(params))
        print("w grad:{}, b grad{}\n".format(w.grad,b.grad))
    return params, losses


t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0])
t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])
t_un = (t_u - t_u.mean())/t_u.std()
param, train_loss = training_loop(1000, 1e-2, torch.tensor([[1.0], [0.0]]), t_un, t_c)

plt.plot(train_loss, color="blue", label='Training Loss')
plt.legend()
plt.show()


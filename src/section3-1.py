
import torch
from matplotlib import pyplot as plt

def model(t_u, w, b):
    return w*t_u + b
def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()

w = torch.ones(1)
b = torch.zeros(1)
t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0])
t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])
t_p = model(t_u, w, b)
loss = loss_fn(t_p, t_c)

def dloss_fn(t_p, t_c):
    dsq_diffs = (t_p - t_c)*2
    return dsq_diffs
def dmodel_dw(t_u, w, b):
    return t_u
def dmodel_db(t_u, w, b):
    return 1.0
def grad_fn(t_u, t_c, t_p, w, b):
    dloss_dw = dloss_fn(t_p, t_c)*dmodel_dw(t_u, w, b)
    dloss_db = dloss_fn(t_p, t_c)*dmodel_db(t_u, w, b)
    return torch.stack([dloss_dw.mean(), dloss_db.mean()])

def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        w, b = params
        t_p = model(t_u, w, b)
        loss = loss_fn(t_p, t_c)
        grad = grad_fn(t_u, t_c, t_p, w, b)
        params = params - learning_rate * grad
        print("Epoch %d, Loss %f" % (epoch, float(loss)))
        print("Params:{}".format(params))
        print("Grad:{}\n".format(grad))
    return params

#training_loop(11, 1e-2, torch.tensor([1.0, 0.0]), t_u, t_c)
t_un = (t_u - t_u.mean())/t_u.std()
param = training_loop(1000, 1e-2, torch.tensor([1.0, 0.0]), t_un, t_c)

t_p = model(t_un, *param)
fig = plt.figure(dpi=100)
fig.patch.set_facecolor('lightgray')
plt.xlabel("Unknown")
plt.ylabel("Celsius")

plt.plot(t_u.numpy(), t_p.detach().numpy())
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
plt.show()

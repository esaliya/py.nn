import torch

dtype = torch.float
device = torch.device('cpu')
# device = torch.device('cuda:0') # uncomment this to run on GPU

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

W1 = torch.randn(D_in, H, device=device, dtype=dtype)
W2 = torch.randn(H, D_out, device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(500):
    z2 = x.mm(W1)
    a2 = z2.clamp(min=0)
    z3 = a2.mm(W2)
    yHat = z3

    J = (y - yHat).pow(2).sum().item()
    print(t, J)

    del3 = -2.0*(y - yHat)
    dJdW2 = a2.t().mm(del3)

    del2 = del3.mm(W2.t())
    del2[z2<0] = 0
    dJdW1 = x.t().mm(del2)

    W1 -= learning_rate * dJdW1
    W2 -= learning_rate * dJdW2




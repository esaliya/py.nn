import torch

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
)

loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4
for t in range(500):
    yHat = model(x)

    J = loss_fn(yHat, y)
    print(t, J.item())

    model.zero_grad()
    J.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
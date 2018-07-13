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
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500):
    yHat = model(x)

    J = loss_fn(yHat, y)
    print(t, J.item())

    # Note, now it's optimizer.zero_grad() not model.zero_grad()
    optimizer.zero_grad()
    J.backward()

    # Update weights using the optimizer
    optimizer.step()
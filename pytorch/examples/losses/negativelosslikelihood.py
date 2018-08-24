import torch
import torch.nn.functional as F

# out = torch.ones((3, 2)).float()
# print(F.log_softmax(out, dim=1))
# target = torch.ones((3,)).long()
# print(out)
# print(target)



out = torch.tensor([[0.9, 0.1],
                   [0.8, 0.2],
                   [0.4, 0.6]])
target = torch.tensor([0, 0, 1])

print('out\n', out)
print('target\n', target)

loss = F.nll_loss(out, target)
print('nll reductin: unspecified', loss)

loss = F.nll_loss(out, target, reduction='none')
print('nll reduction: none', loss)

loss = F.nll_loss(out, target, reduction='sum')
print('nll reduction: sum', loss)

import math
print(math.log(0.9))

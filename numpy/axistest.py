import numpy as np

a = np.arange(6).reshape(2, 3)
print(a)

print("a - max along axis 0")
print(np.max(a, axis=0))


b = np.arange(12).reshape(2, 3, 2)
print(b)

print("b - max along axis 0")
print(np.max(b, axis=0))

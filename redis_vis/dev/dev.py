import numpy as np

x = np.array([1.234, 2.33, 3.321]).tobytes()
print(x)
y = np.frombuffer(x, dtype=np.dtype(float))
print(y)

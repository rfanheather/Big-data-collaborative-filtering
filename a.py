import numpy as np

a = np.array([
            [1, 3, 4, 5, 6],
            [2, 1, 2, 3, 4]
          ])
a = a.tolist()

a = a[a != 2]
print(a)

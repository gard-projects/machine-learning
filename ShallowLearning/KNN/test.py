import numpy as np

a1 = np.array([[1,1,1]])
a2 = np.array([[3,3,3], [4,4,4]])

result = np.concatenate((a1, a2),axis=0)
print(result)
print(result.shape)
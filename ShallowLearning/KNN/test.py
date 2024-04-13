import numpy as np

q = np.array([[1,2,2], [2,3,3]])
p  = np.array([5,5,6])

t = np.dot(p,q.T)
t2 = np.linalg.norm(q, axis=1)
print(t2)
print(t)
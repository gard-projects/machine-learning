import numpy as np

# Example arrays
array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array([5, 2, 3, 8, 5])
test = np.empty((50,1), dtype=str)
print(test.shape)
test[0] = "abc"
print(test)

array1[0] = 50
print(array1)
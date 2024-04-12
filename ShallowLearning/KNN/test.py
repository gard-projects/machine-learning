import numpy as np
data = np.array([10, 20, 30, 40, 50, 70, 10, 20, 40, 80, 90, 100, 200, 20])
test = {val: np.where(data == val) for val in np.unique(data)}

import numpy as np
import pandas as pd

data = [[1,2,3],[4,5,6]]

np1 = pd.DataFrame(data)

print(np1)

pd1 = np1.to_numpy()

print(pd1)
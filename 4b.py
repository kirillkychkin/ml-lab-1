import numpy as np

def normalize(v):
    return v / np.linalg.norm(v) # делим на длину

v = np.array([[3], 
              [4]])

print(normalize(v))
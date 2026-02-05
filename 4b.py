import numpy as np

def normalize(v):
    return v / np.linalg.norm(v) # делим на длину
if __name__ == '__main__':
    v = np.array([[3], 
                [4]])

    print(normalize(v))
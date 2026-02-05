import numpy as np

def calc(matrix):
    n = matrix.shape[0]
    ones = np.ones((1, n))
    sums = ones.dot(matrix)
    
    return sums / n

if __name__ == '__main__':
    people_data = np.array([[170, 65],
                            [174, 80],
                            [179, 90],
                            [166, 62]])
    print(calc(people_data))
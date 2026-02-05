import numpy as np

if __name__ == '__main__':
    # task 1
    print("task 1:\n")
    a = np.array([[1],[5],[-3],[2]])
    b = np.array([[8],[2],[4],[7]])

    print("b - a:\n", b - a)
    print("2a^t + 3b^t:\n", 2 * a.T + 3 * b.T)

    # task 2
    print("\ntask 2:\n")
    M = np.array([[1, 2, 4], [-2, 5, -1]])
    v = np.array([[3], [0], [-5]])
    w = np.array([[2], [-3]])

    def calculate_dot_product(a, b):
        if a.shape[1] != b.shape[0]:
            print("incorrect shape, can't multiply\n")
            return
        print(a.dot(b),"\n")

    # a
    print("M * v:\n", calculate_dot_product(M, v))
    print("M* v^t:\n", calculate_dot_product(M, v.T))
    print("v * M:\n", calculate_dot_product(v, M))
    print("v^t * M:\n", calculate_dot_product(v.T, M))
    print("w^t * M:\n", calculate_dot_product(w.T, M))
    print("w * M:\n", calculate_dot_product(w, M))

    # b
    print("v * w:\n", calculate_dot_product(v, w))
    print("v * w^t:\n", calculate_dot_product(v, w.T))
    print("v^t * w:\n", calculate_dot_product(v.T, w))
    print("w * v:\n", calculate_dot_product(w, v))
    print("w * v^t:\n", calculate_dot_product(w, v.T))

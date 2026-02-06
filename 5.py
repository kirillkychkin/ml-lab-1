import numpy as np


def signed_dist(x, th, th0):
    numerator = np.dot(th.T, x) + th0
    denominator = np.linalg.norm(th)
    return (numerator / denominator)[0, 0]


def positive(x, th, th0):
    value = (np.dot(th.T, x) + th0)[0, 0]
    return int(np.sign(value))


def score(data, labels, th, th0):
    predictions = np.sign(np.dot(th.T, data) + th0)
    return np.sum(predictions == labels)


def best_separator(data, labels, ths, th0s):
    m = ths.shape[1]
    best_score = -1
    best_index = 0
    
    for i in range(m):
        # : чтобы взять все строки
        # от i до i + 1
        # берем столбец как 2d array
        th_i = ths[:, i:i+1]
        th0_i = th0s[0, i]
        current_score = score(data, labels, th_i, th0_i)
        
        if current_score > best_score:
            best_score = current_score
            best_index = i
    
    return (ths[:, best_index:best_index+1], th0s[0, best_index])


if __name__ == "__main__":
    # Тест signed_dist
    x = np.array([[13], [4], [9]])
    th = np.array([[-7], [3], [-5]])
    th0 = -7
    print("Signed distance:", signed_dist(x, th, th0))
    
    # Тест positive
    print("Positive:", positive(x, th, th0))
    
    # Тест score и best_separator
    data = np.array([[1, 2, 3, 4, 5, 3, 2],
                     [3, 4, 5, 1, 2, 2, 2]])
    labels = np.array([[1, 1, 1, -1, -1, 0, 1]])
    
    th1 = np.array([[1], [1]])
    th0_1 = -5
    th2 = np.array([[-1], [2]])
    th0_2 = -2
    
    print("Score 1:", score(data, labels, th1, th0_1), "of", data.shape[1])
    print("Score 2:", score(data, labels, th2, th0_2), "of", data.shape[1])
    
    ths = np.array([[1, -1], [1, 2]])
    th0s = np.array([[-5, -2]])
    
    best_th, best_th0 = best_separator(data, labels, ths, th0s)
    print("Best th:\n", best_th)
    print("Best th0:", best_th0)
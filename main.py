# import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
from qpsolvers import solve_qp
import scipy.io
# %matplotlib inline

# Separate Versicolor data and Virginica data from Iris data set
def separate(data):
    # 將資料分為前半(first_half)與後半(second_half)
    split_data = np.split(data, 6)
    first_half = [split_data[0], split_data[2], split_data[4]]
    second_half = [split_data[1], split_data[3], split_data[5]]

    # 只取Versicolor(label=2) 與 Virginica(label=3)
    training_data = np.vstack((first_half[1], first_half[2]))
    test_data = np.vstack((second_half[1], second_half[2]))

    # positive(label=2), negative(label=3)
    for i in range(len(test_data)):
        if (test_data[i][4] == 2):
            training_data[i][4] = 1
            test_data[i][4] = 1
        else:
            training_data[i][4] = -1
            test_data[i][4] = -1

    return training_data, test_data

# Evalute linear kernel
def linear_kernel(feature_1, feature_2):
    x_1 = np.array(feature_1)
    x_2 = np.array(feature_2)
    kernel_set = np.zeros((len(x_1), len(x_2)))
    kernel_set = kernel_set.astype(float)

    for i in range(len(x_1)):
        for j in range(len(x_2)):
            kernel_set[i][j] = round(((x_1[i].T).dot(x_2[j])), 6)

    return kernel_set

# Evalute poly kernel
def poly_kernel(feature_1, feature_2, P):
    x_1 = np.array(feature_1)
    x_2 = np.array(feature_2)
    kernel_set = np.zeros((len(x_1), len(x_2)))
    kernel_set = kernel_set.astype(float)
    
    for i in range(len(x_1)):
        for j in range(len(x_2)):
            kernel_set[i][j] = round((((x_1[i].T).dot(x_2[j])) ** P), 6)
    
    return kernel_set

# Evalute RBF kernel
def RBF_kernel(feature_1, feature_2, sigma):
    x_1 = np.array(feature_1)
    x_2 = np.array(feature_2)
    kernel_set = np.zeros((len(x_1), len(x_2)))
    kernel_set = kernel_set.astype(float)

    r = 1/((2* (sigma**2)))
    
    for i in range(len(x_1)):
        for j in range(len(x_2)):
            # 計算歐幾里德距離
            norm = np.sqrt(np.square(x_1[i][0] - x_2[j][0]) + np.square(x_1[i][1] - x_2[j][1]))
            # euclidean_distance = np.linalg.norm(x_1[i] - x_2[j])
            kernel_set[i][j] = np.exp(-r * norm**2)

    return kernel_set


# Evalute Hessian Matrix
def Hessian(label_set, kernel_set):
    y = label_set

    H = np.zeros((len(y), len(y)))
    H = H.astype(float)
    for i in range(len(y)):
        for j in range(len(y)):
            H[i][j] = y[i] * y[j] * kernel_set[i][j]
            
    return H

# Dual problem
def Dual(label_set, Hessian_matrix, C):
    y = label_set
    
    P = Hessian_matrix
    q = -1 * np.ones(len(y))
    A = np.array(y)
    b = np.array([0.0])
    lb = np.zeros(len(y))               # 都是0的一維陣列
    ub = C * np.ones(len(y))            # 都是C的一維陣列

    alpha = solve_qp(P, q, None, None, A, b, lb, ub, solver="clarabel")
    
    eps =   2.2204e-16
    for i in range(alpha.size):
        if alpha[i] >= C - np.sqrt(eps):
            alpha[i] = C
            alpha[i] = np.round(alpha[i],6)
        elif  alpha[i] <= 0 + np.sqrt(eps):
            alpha[i] = 0
            alpha[i] = np.round(alpha[i],6)
        else:
            alpha[i] = np.round(alpha[i],6)
            print(f"support vector: alpha = {alpha[i]}")
            # print(f"alpha = {np.round(alpha[i],4)}")

    print(alpha[:5])
    return alpha


# Kuhn-Tucker condition
def KT_condition(feature_set, label_set, alpha, C, kernel):
    x = feature_set
    y = label_set
    alpha_set = alpha
    K = kernel

    b_set = np.zeros(len(alpha))
    print(len(K[0]))
    count = 0
    for i in range(len(alpha_set)):
        if (alpha_set[i] > 0 and alpha_set[i] < C):
            count += 1
            # print(alpha_set[i])
            w_dot_phi = 0
            for j in range(len(y)):
                w_dot_phi += alpha_set[j]* y[j]* K[i][j]
                # print("w_dot_phi = ", w_dot_phi, alpha_set[j], K[i][j])
            b_set[i] = (1/y[i]) - w_dot_phi

    # print("b_set = ", b_set)
    # 計算最佳化的b(平均的b)
    b = 0
    for i in range(len(b_set)):
        b += b_set[i]
    b = round((b/count), 4)

    return b

# 將資料分割為feature與label
def x_y(data):
    # 取PL 跟PW 作為feature
    x = data[:, 2:4]
    y = data[:, 4]

    return x, y


# 將資料輸出為csv
def output_result(data, filename):
    with open(f"{filename}.csv", "a", newline="") as file:
        for i in range(len(data)):
            file.write(f"{data[i]},")
        file.write("\n")


# Decision Rule
def Decision(training_data, test_data, alpha_set, bias, C, kernel_set):
    # 將資料的feature跟label分離
    x_training, y_training = x_y(training_data)
    x_test, y_test = x_y(test_data)

    # Prediction result
    prediction = []

    # print(kernel_set)
    
    print("y_training = ", y_training)
    
    for i in range(len(x_test)):
        # Evalute <w, phi>
        w_dot_phi = 0
        D = 0
        for j in range(len(x_training)):
            w_dot_phi += alpha_set[j] * y_training[j] * kernel_set[i][j]
            # w_dot_phi += alpha_set[j] * y_training[j] * (x_training[j].T).dot(x_test[i])
            # print(kernel_set[j][i], (x_training[j].T).dot(x_test[i]))
        print("w_dot_phi = ", w_dot_phi)

        # Decision rule
        D = round(w_dot_phi, 6) + bias
        if (D >= 0):
            prediction.append(1)
        else:
            prediction.append(-1)

    prediction = np.array(prediction)
    
    return prediction


# 計算分類率
def classification_rate(test_data, predict):
    # 預測正確的資料總數
    True_prediction = 0

    # 將predict的label與test data的label做比對
    for i in range(len(predict)):
        if predict[i] == test_data[i][4]:
            True_prediction += 1
    
    # 分類率
    # print(True_prediction)
    CR = round(True_prediction / len(test_data), 5) * 100
    return CR


# SVM classifier
def SVM(initial_data, kernel, C, P=None, sigma=None):
    # Split data
    training_data, test_data = separate(initial_data)

    # 將資料的feature跟label分離
    x_training, y_training = x_y(training_data)
    x_test, y_test = x_y(test_data)

    '''training process'''
    # Evalute training data的linear kernel
    if (kernel == "Linear"):
        training_kernel_set = linear_kernel(x_training, x_training)
    elif (kernel == "Polynomial"):
        training_kernel_set = poly_kernel(x_training, x_training, P)
    else:
        training_kernel_set = RBF_kernel(x_training, x_training, sigma)
    # print("training_kernel_set =", training_kernel_set)
    
    # Evalute Hessian Matrix
    H = Hessian(y_training, training_kernel_set)
    print("H =", H)


    # Evalute alpha
    alpha_set = Dual(y_training, H, C)
    num = 0
    for i in range(len(alpha_set)):
        num += alpha_set[i]
        alpha_set[i] = round(alpha_set[i], 4)
    num = round(num, 4)
    print("num = ", num)
    print("alpha = ", alpha_set)

    # 將alpha資料輸出
    if (kernel == "Linear"):
        filename = f"alpha_{kernel}_C{C}_SVM"
    elif (kernel == "Polynomial"):
        filename = f"alpha_{kernel}_poly{P}_SVM"
    else:
        filename = f"alpha_{kernel}_sigma{sigma}_SVM"
    
    output_result(alpha_set, filename)

    # Evalute bias
    b = KT_condition(x_training, y_training, alpha_set, C, training_kernel_set)
    print("bias = ", b)
    
    # 將bias輸出
    with open(f"{filename}.csv", "a", newline="") as file:
        file.write(f"alpha total = {'%.4f'%round(num, 4)}\n")
        file.write(f"bias = {'%.4f'%round(b, 4)}\n")

    '''testing process'''
    # Evalute test data的linear kernel
    if (kernel == "Linear"):
        test_kernel_set = linear_kernel(x_test, x_training)
    elif (kernel == "Polynomial"):
        test_kernel_set = poly_kernel(x_test, x_training, P)
    else:
        test_kernel_set = RBF_kernel(x_test, x_training, sigma)

    # test_kernel_set = np.array(test_kernel_set)
    # print("test kernel set = ", test_kernel_set)
    
    # 預測結果
    prediction = Decision(training_data, test_data, alpha_set, b, C, test_kernel_set)
    print("prediction = ", prediction)
    # print(len(prediction))

    # 分類率
    CR = classification_rate(test_data, prediction)
    print("CR =", CR)

    # 將prediction 與CR輸出為file
    if (kernel == "Linear"):
        filename = f"prediction_{kernel}_C{C}_SVM"
    elif (kernel == "Polynomial"):
        filename = f"prediction_{kernel}_poly{P}_SVM"
    else:
        filename = f"prediction_{kernel}_sigma{sigma}_SVM"
    
    output_result(prediction, filename)
    with open(f"{filename}.csv", "a", newline="") as file:
        file.write(f"CR = {'%.4f'%round(CR, 4)} %\n")

    # 將結果繪圖
    scatter_plot(training_data, test_data, alpha_set, b, C)


# 繪製結果
def scatter_plot(training_data, test_data, alpha_set, b, C):
    # 提取特徵和標籤
    x_training, y_training = x_y(training_data)
    x_test, y_test = x_y(test_data)

    # 計算支持向量
    support_vectors = []
    for i, alpha in enumerate(alpha_set):
        if alpha > 0:
            support_vectors.append(x_training[i])

    support_vectors = np.array(support_vectors)

    # 繪製散點圖
    plt.figure(figsize=(8, 8))
    plt.rcParams['font.size'] = 14
    plt.title(f'SVM Classifier (C={C})')

    plt.scatter(x_training[y_training == 1][:, 0], x_training[y_training == 1][:, 1], c='b', cmap=plt.cm.Paired, marker='o', label='Positive')
    plt.scatter(x_training[y_training == -1][:, 0], x_training[y_training == -1][:, 1], c='r', cmap=plt.cm.Paired, marker='x', label='Negative')
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], c='g', marker='*', s=100, label='Support Vectors')

    # 繪製決策邊界
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))

    # x_min = np.amin(x_test.T[0])
    # x_max = np.amax(x_test.T[0])
    # y_min = np.amin(x_test.T[1])
    # y_max = np.amax(x_test.T[1])
    # xx, yy = np.mgrid[x_min:x_max:50j, y_min:y_max:50j]
    xy = np.vstack([xx.ravel(), yy.ravel()]).T
    W = np.dot(x_training.T, (alpha_set * y_training))
    Z = np.dot(xy, W) + b
    Z = Z.reshape(xx.shape)

    # levels = np.linspace(Z.min(), Z.max(), 1)
    plt.contour(xx, yy, Z, colors=['k', 'k', 'k'], levels=[-1, 0, 1], linestyles=['--', '-', '--'])
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.legend(scatterpoints=1, markerscale=1)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# main function
def main():
    # load data
    raw_data = np.loadtxt("iris.txt", dtype=float)

    # 參數設定
    kernel_type = ["Linear", "Polynomial", "RBF"]
    kernel = kernel_type[0]
    C = 100
    P = 2
    sigma = None

    # 執行SVM classifier
    SVM(raw_data, kernel, C)

    
if __name__ == "__main__":
    main()
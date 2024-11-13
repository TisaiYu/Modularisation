import numpy as np
from numpy import linalg
# np.set_printoptions(precision=4)


class AHP:
    def __init__(self,a):
        self.a = a
        self.CR = 0
    '''算术平均法权重'''
    def arithmetic_mean(self):
        n = len(self.a)
        b = sum(self.a)
        print('b:', b)
        # 归一化处理
        normal_a = self.a / b
        print("算术平均法权重-归一化处理：")
        print(normal_a)
        average_weight = []
        for i in range(n):
            s = sum(normal_a[i])
            print("第{}行求和 ".format(i + 1), s)
            # 平均权重
            average_weight.append(s / n)
        # print(average_weight)
        print('算术平均法权重:')
        print(np.array(average_weight))
        return np.array(average_weight)

    '''几何平均法求权重'''
    def geometric_mean(self):
        n = len(self.a)
        # 1表示按照行相乘，得到一个新的列向量
        b = np.prod(self.a, 1)
        print(b)
        c = pow(b, 1 / n)
        print(c)
        # 归一化处理
        average_weight = c / sum(c)
        print('几何平均法权重:')
        print(average_weight)
        return average_weight

    '''特征值法求权重'''
    def eigenvalue(self):
        w, v = np.linalg.eig(self.a) #得到特征值与特征向量
        for i in range(len(w)):
            print('特征值', self.a[i], '特征向量', v[:, i])
        index = np.argmax(w)
        w_max = np.real(w[index])
        vector = v[:, index]
        vector_final = np.transpose(np.real(vector))
        print('最大特征值', w_max, '对应特征向量', vector_final)
        normalized_weight = vector_final / sum(vector_final)
        print('***归一化处理后:', normalized_weight)
        return w_max, normalized_weight


    '''综合平均权重'''
    def average_Weight(self):
        am = self.arithmetic_mean()
        gm = self.geometric_mean()
        ev = self.eigenvalue()[1]
        aw = np.array([am, gm, ev])
        print(aw)
        final_weight = sum(aw) / 3
        print("final weight：",final_weight)
        return final_weight


    '''判断正互反矩阵'''
    def reciprocal_matrix_judge(self):
        print(self.a)
        n = len(self.a)
        b = 0
        for j in range(n):
            for i in range(n):
                if self.a[:, j][i] * self.a[j, :][i] == 1:
                    b += 1
        if b == n * n:
            print("该矩阵是正互反矩阵！\n")
            return True
        else:
            print("该矩阵不是正互反矩阵！\n")
            return False

    '''CI计算'''
    def CI_calc(self):
        n = len(self.a)
        λ_max = self.eigenvalue()[0]
        print(λ_max)
        if n>1:
            CI = (λ_max-n)/(n-1)
        else:
            CI = 0
        print('CI:', CI)
        return CI

    '''CR计算'''
    def CR_calc(self):
        RI = np.array([0, 0, 0.52, 0.89, 1.12, 1.26, 1.36, 1.41,
                       1.46, 1.49, 1.52, 1.54, 1.56, 1.58, 1.59])
        n = len(self.a)
        CI = self.CI_calc()
        if RI[n - 1]==0:
            self.CR = 0
        else:
            self.CR = CI / RI[n - 1]
        print('CR:', self.CR)
        if self.CR < 0.1:
            print("一致性检验通过！\n")
            return True
        else:
            print("一致性检验失败，请修改！")
            return False

    def calculate(self):
        # a = np.array([[1, 2, 5],
        #               [1 / 2, 1, 2],
        #               [1 / 5, 1 / 2, 1]])

        rmj = self.reciprocal_matrix_judge()
        if rmj:
            CR_0 = self.CR_calc()
            weights = self.average_Weight()
            return weights
        else:
            return False

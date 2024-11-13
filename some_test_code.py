from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import *

def elbow_test():
    D=np.array([[1.0,0.8,0.8,0.8],[0.8,1.0,0.8,0.8],[0.8,0.8,1.0,0.8],[0.8,0.8,0.8,1]])
    dis_mat = pdist(D,
                    'cosine')
    Z_new = hierarchy.linkage(dis_mat, method='ward', metric="euclidean")

    cluster_labels = hierarchy.fcluster(Z_new,Z_new[-3,2]-0.01,"distance")
    print(cluster_labels)
    re  = hierarchy.dendrogram(Z_new, color_threshold=0.2, above_threshold_color='#bcbddc')
    plt.show()

def save_module_test():
    iteration_num = 1
    cluster_labels = np.array([1,1,2,2,3,3])
    module_weight = [1,2 ,3, 4, 5, 6]
    moudule_size = [1,2 ,3, 4, 5, 6]
    SIC = 20
    save_module_iteration(iteration_num,cluster_labels,module_weight,moudule_size,SIC)
def d1_conv():
    a=np.array([[1,1,1]])
    b=a.transpose()
    print(a)
    print(b)
    c = np.convolve(a,b)
    print(c)



import sympy as sp
def solve_exponential_function(f, b):
    # 定义符号
    C = sp.symbols('C')

    # 计算 B
    B = 1 / sp.sqrt(f)

    # 计算 A
    A = 0.2 / sp.exp(1 - C / sp.sqrt(f))

    # 计算 C 的方程
    equation = A * sp.exp(B * (1 / f - C)) - b

    # 解方程
    print(sp.solve(equation, C))
    C_value = sp.solve(equation, C)[0]

    # 更新 A 的值
    A_value = A.subs(C, C_value)

    # 返回最终表达式
    final_expression = A_value * sp.exp(B * (sp.symbols('x') - C_value))

    return final_expression.simplify(), A_value, B, C_value


def calculate_values(f, b):
    # 求解函数表达式和参数
    final_expression, A, B, C = solve_exponential_function(f, b)

    # 生成 x 的值
    x_values = np.linspace(0.5, f - 1, 100)  # 从 0.5 到 f-1 生成 100 个点

    # 将符号表达式转换为可计算的函数
    func = sp.lambdify(sp.symbols('x'), final_expression, 'numpy')

    # 计算函数值
    y_values = func(x_values)

    return x_values, y_values






def plot_piecewise_function():
    lf = 5
    FNC = 0.5
    k=1
    b=1/FNC
    x_values1 = np.linspace(0.5, np.sqrt(lf), 500)
    x_values2 = np.linspace(np.sqrt(lf), lf - 0.8, 500)
    # 定义分段函数
    y1 = (0.2 / (np.sqrt(lf) - 0.5) * (x_values1 - 0.5))
    # return FNC*(0.2 + ((1 / FNC) - 0.2) * np.power((x-np.sqrt(lf))/(lf-1-np.sqrt(lf)),k))
    y2 = 0.2+(b-0.2)*np.power((x_values2-np.sqrt(lf))/(lf-1-np.sqrt(lf)),k)

    # 生成 x 值

    # 绘制图像
    plt.plot(x_values1, y1)
    plt.plot(x_values2, y2)

    plt.title('Piecewise Function')
    plt.show()

def plot_one_funciton():
    lf = 5
    FNC=0.5
    k=0.5
    b=1/FNC
    equal_proportion = 0.8
    for k in np.linspace(0.2,2,5):
        for equal_proportion in np.linspace(0.6,0.9,4):
            equal_proportion=0.8
            x= np.linspace(0.51,lf-1,4000)

            from scipy.optimize import fsolve
            # 定义方程
            def equation(k, value):
                return k * np.log((np.exp(1 / k) - 1) / 0.2*FNC + 1) - value

            # 初始猜测值
            k_guess = 0.5
            # 计算右边的值
            value = ((lf-1) * equal_proportion - 0.5) / (np.sqrt(lf) - 0.5)
            # 使用牛顿迭代法求解
            k_solution = fsolve(equation, k_guess, args=(value))
            print("k =", k_solution[0])
            k = k_solution[0]
            y = b*(np.exp((x-0.5)/k*(np.sqrt(lf)-0.5))-1)/(np.exp(((lf-1)*equal_proportion-0.5)/k*(np.sqrt(lf)-0.5))-1)
            y[np.where(y>b)]=0
            max_va= np.max(y)
            y[np.where(y==0)]=max_va
            curve_label = f'k={k}, p={equal_proportion}'
            plt.plot(x, y,label=curve_label)
            plt.legend()
            plt.title('Piecewise Function')
            plt.show()

if __name__ == "__main__":
    import numpy as np
    import skfuzzy as fuzz
    import skfuzzy.control as ctrl

    elbow_test()
    # plot_one_funciton()
    # plot_piecewise_function()

    # # 定义模糊变量的范围
    # universe = np.linspace(1, 6, 100)
    # # 创建模糊变量
    # variable = ctrl.Antecedent(universe, 'variable')
    # # 使用 automf 函数生成隶属函数
    # variable.automf(3)
    # # 查看生成的隶属函数
    # variable.view()
    # plt.plot()


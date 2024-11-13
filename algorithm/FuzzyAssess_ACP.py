import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['font.family'] = ['Microsoft YaHei']  # 设置字体为微软雅黑
jiange1 = 1.8
jiange2 = 0.2

class FuzzySystemACP:
    def __init__(self):
        # 定义模糊变量
        self.Wcij = np.arange(1, 6.1, 0.1)
        self.Acij = np.arange(1, 6.1, 0.1)
        self.Tcij = np.arange(1, 6.1, 0.1)
        self.ACPij = np.arange(1, 6.1, 0.1)

        # 定义隶属度函数
        self.Wcij_low = fuzz.trapmf(self.Wcij, [0, 0,1.8, 2.5])
        self.Wcij_medium = fuzz.trapmf(self.Wcij, [1.8, 3.5,3.5, 5.2])
        self.Wcij_high = fuzz.trapmf(self.Wcij, [3.8, 5.5,6, 6])

        self.Acij_low = fuzz.trapmf(self.Acij, [0, 0,1.8, 2.5])
        self.Acij_medium = fuzz.trapmf(self.Acij, [1.8, 3.5,3.5, 5.2])
        self.Acij_high = fuzz.trapmf(self.Acij, [3.8, 5.5,6, 6])

        self.Tcij_low = fuzz.trapmf(self.Tcij, [0, 0,1.8, 2.5])
        self.Tcij_medium = fuzz.trapmf(self.Tcij, [1.8, 3.5,3.5, 5.2])
        self.Tcij_high = fuzz.trapmf(self.Tcij, [3.8, 5.5,6, 6])

        self.ACPij_low = fuzz.trapmf(self.ACPij, [0, 0,1.8, 2.5])
        self.ACPij_medium = fuzz.trapmf(self.ACPij, [1.8, 3.5,3.5, 5.2])
        self.ACPij_high = fuzz.trapmf(self.ACPij, [3.8, 5.5,6, 6])

        # 定义规则
        self.rules = [
            ('low', 'low', 'low', self.ACPij_low),
            ('low', 'medium', 'low', self.ACPij_low),
            ('low', 'medium', 'medium', self.ACPij_medium),
            ('low', 'high', 'low', self.ACPij_medium),
            ('low', 'high', 'high', self.ACPij_high),
            ('medium', 'low', 'low', self.ACPij_low),
            ('high', 'low', 'low', self.ACPij_low),
            ('medium', 'medium', 'low', self.ACPij_medium),
            ('medium', 'high', 'low', self.ACPij_high),
            ('medium', 'high', 'high', self.ACPij_high),
            ('high', 'medium', 'low', self.ACPij_medium),
            ('high', 'high', 'low', self.ACPij_high),
            ('high', 'high', 'high', self.ACPij_high),
            ('high', 'high', 'medium', self.ACPij_high),
            ('low', 'low', 'high', self.ACPij_medium),
            ('low', 'low', 'medium', self.ACPij_low),
            ('low', 'high', 'medium', self.ACPij_high),
            ('high', 'low', 'high', self.ACPij_medium),
            ('medium', 'low', 'medium', self.ACPij_low),
            ('medium', 'medium', 'medium', self.ACPij_medium),
            ('high', 'medium', 'high', self.ACPij_high),
            ('high', 'low', 'medium', self.ACPij_medium),
            ('medium', 'medium', 'high', self.ACPij_high),
            ('low', 'medium', 'high', self.ACPij_medium),
            ('medium', 'high', 'medium', self.ACPij_high),
            ('medium', 'low', 'high', self.ACPij_medium),
            ('high', 'medium', 'medium', self.ACPij_medium)
        ]

    def t_norm_min(self, a, b):
        return np.minimum(a, b)

    def t_norm_product(self, a, b):
        return a * b

    def mamdani_aggregation(self, inputs):
        aggregated = np.zeros_like(self.ACPij)
        for rule in self.rules:
            antecedent = self.t_norm_min(self.t_norm_min(inputs['Wcij'][rule[0]], inputs['Acij'][rule[1]]), inputs['Tcij'][rule[2]])
            consequent = rule[-1]
            # aggregated += np.minimum(antecedent, consequent)  # 使用sum操作
            aggregated = np.maximum(aggregated, np.minimum(antecedent, consequent)) # TisaiYu[2024/8/28] 使用max操作
        return aggregated

    def defuzzify_centroid(self, aggregated):
        numerator = np.sum(aggregated * self.ACPij)
        denominator = np.sum(aggregated)
        if denominator == 0:
            return 0
        return numerator / denominator

    def defuzzify_weighted_average(self, aggregated):
        numerator = np.sum(aggregated * self.ACPij)
        denominator = np.sum(aggregated)
        if denominator == 0:
            return 0
        return numerator / denominator

    def defuzzify_bisector(self, aggregated):
        total_area = np.sum(aggregated)
        if total_area == 0:
            return 0
        cumulative_area = np.cumsum(aggregated)
        bisector_index = np.where(cumulative_area >= total_area / 2)[0][0]
        return self.ACPij[bisector_index]

    def defuzzify_mom(self, aggregated):
        max_value = np.max(aggregated)
        max_indices = np.where(aggregated == max_value)[0]
        return np.mean(self.ACPij[max_indices])

    def defuzzify_som(self, aggregated):
        max_value = np.max(aggregated)
        max_indices = np.where(aggregated == max_value)[0]
        return self.ACPij[max_indices[0]]

    def defuzzify_lom(self, aggregated):
        max_value = np.max(aggregated)
        max_indices = np.where(aggregated == max_value)[0]
        return self.ACPij[max_indices[-1]]

    def plot_membership_functions(self):
        plt.figure(figsize=(10, 6))

        # 绘制模糊隶属度函数
        # plt.plot(self.Wcij, self.Wcij_low, label='Wcij Low', color='red')
        # plt.plot(self.Wcij, self.Wcij_medium, label='Wcij Medium', color='orange')
        # plt.plot(self.Wcij, self.Wcij_high, label='Wcij High', color='green')
        #
        # plt.plot(self.Acij, self.Acij_low, label='Acij Low', color='red', linestyle='--')
        # plt.plot(self.Acij, self.Acij_medium, label='Acij Medium', color='orange', linestyle='--')
        # plt.plot(self.Acij, self.Acij_high, label='Acij High', color='green', linestyle='--')
        #
        # plt.plot(self.Tcij, self.Tcij_low, label='Tcij Low', color='red', linestyle=':')
        # plt.plot(self.Tcij, self.Tcij_medium, label='Tcij Medium', color='orange', linestyle=':')
        # plt.plot(self.Tcij, self.Tcij_high, label='Tcij High', color='green', linestyle=':')

        plt.plot(self.ACPij, self.ACPij_low, label='ACPij Low', color='red', linewidth=2)
        plt.plot(self.ACPij, self.ACPij_medium, label='ACPij Medium', color='orange', linewidth=2)
        plt.plot(self.ACPij, self.ACPij_high, label='ACPij High', color='green', linewidth=2)

        # 水平截断线
        plt.axhline(y=0.5, color='black', linestyle='--', label='规则1去模糊重心法面积')
        plt.axhline(y=0.12, color='blue', linestyle='--', label='规则2去模糊重心法面积')

        # 填充低于0.5的区域
        plt.fill_between(self.ACPij, 0, np.minimum(self.ACPij_low, 0.5), color='red',
                         alpha=0.5)

        # 填充低于0.12的区域
        plt.fill_between(self.ACPij, 0, np.minimum(self.ACPij_medium, 0.12),
                         color='orange', alpha=0.5)

        plt.title('去模糊重心法示意图')
        plt.xlabel('ACP')
        plt.ylabel('隶属度')
        plt.legend()
        plt.grid()
        plt.show()

    def calculate_ACPij(self, wcij, acij, tcij):
        if wcij > 6:
            wcij= 6
        if acij > 6:
            acij = 6
        if tcij > 6:
            tcij = 6

        inputs = {
            'Wcij': {
                'low': fuzz.interp_membership(self.Wcij, self.Wcij_low, wcij),
                'medium': fuzz.interp_membership(self.Wcij, self.Wcij_medium, wcij),
                'high': fuzz.interp_membership(self.Wcij, self.Wcij_high, wcij)
            },
            'Acij': {
                'low': fuzz.interp_membership(self.Acij, self.Acij_low, acij),
                'medium': fuzz.interp_membership(self.Acij, self.Acij_medium, acij),
                'high': fuzz.interp_membership(self.Acij, self.Acij_high, acij)
            },
            'Tcij': {
                'low': fuzz.interp_membership(self.Tcij, self.Tcij_low, tcij),
                'medium': fuzz.interp_membership(self.Tcij, self.Tcij_medium, tcij),
                'high': fuzz.interp_membership(self.Tcij, self.Tcij_high, tcij)
            }
        }
        # print(f"输入为Wcij,Acij,Tcij为[{wcij},{acij},{tcij}]")
        # print(f"Wcij隶属度(low,medium,high):{inputs['Wcij']['low'],inputs['Wcij']['medium'],inputs['Wcij']['high']}")
        # print(f"Acij隶属度(low,medium,high):{inputs['Acij']['low'],inputs['Acij']['medium'],inputs['Acij']['high']}")
        # print(f"Tcij隶属度(low,medium,high):{inputs['Tcij']['low'],inputs['Tcij']['medium'],inputs['Tcij']['high']}")
        # print("------------------------------------------------------")
        # print("计算每个规则的隶属度：")
        aggregated = self.mamdani_aggregation(inputs)
        ACPij = self.defuzzify_centroid(aggregated)
        # ACPij = self.defuzzify_weighted_average(aggregated)
        # ACPij = self.defuzzify_bisector(aggregated)
        # ACPij = self.defuzzify_lom(aggregated)
        # ACPij = self.defuzzify_mom(aggregated)
        # ACPij = self.defuzzify_som(aggregated)
        return ACPij

    def view(self):
        # 显示隶属度函数
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        axs[0, 0].plot(self.Wcij, self.Wcij_low, 'b', label='Low')
        axs[0, 0].plot(self.Wcij, self.Wcij_medium, 'g', label='Medium')
        axs[0, 0].plot(self.Wcij, self.Wcij_high, 'r', label='High')
        axs[0, 0].set_title('Wcij Membership Functions')
        axs[0, 0].legend()

        axs[0, 1].plot(self.Acij, self.Acij_low, 'b', label='Low')
        axs[0, 1].plot(self.Acij, self.Acij_medium, 'g', label='Medium')
        axs[0, 1].plot(self.Acij, self.Acij_high, 'r', label='High')
        axs[0, 1].set_title('Acij Membership Functions')
        axs[0, 1].legend()

        axs[1, 0].plot(self.Tcij, self.Tcij_low, 'b', label='低')
        axs[1, 0].plot(self.Tcij, self.Tcij_medium, 'g', label='中')
        axs[1, 0].plot(self.Tcij, self.Tcij_high, 'r', label='高')
        axs[1, 0].set_title('Tcij Membership Functions')
        axs[1, 0].legend()

        axs[1, 1].plot(self.ACPij, self.ACPij_low, 'b', label='低')
        axs[1, 1].plot(self.ACPij, self.ACPij_medium, 'g', label='中')
        axs[1, 1].plot(self.ACPij, self.ACPij_high, 'r', label='高')
        axs[1, 1].set_title('ACP及输入变量的隶属度函数')
        axs[1, 1].legend()

        plt.tight_layout()
        plt.show()

        # 显示规则组合下两两输入对于结果的变化
        input_variables = {"Wcij":self.Wcij,"Acij":self.Acij,"Tcij":self.Tcij}
        input_variables_name = ["Wcij","Acij","Tcij"]

        combination = itertools.combinations(input_variables_name,2)
        for select_show in combination:
            params_dict = {}
            var1 = select_show[0]
            var2 = select_show[1]
            var3 = None
            for var in input_variables_name:
                if var not in [var1,var2]:
                    var3 = var

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            X, Y = np.meshgrid(input_variables[var1], input_variables[var2])
            Z = np.zeros_like(X)
            params_dict[var1] = X
            params_dict[var2] = Y
            params_dict[var3] = np.ones_like(X)*3
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i, j] = self.calculate_ACPij(params_dict['Wcij'][i,j], params_dict['Acij'][i,j], params_dict['Tcij'][i,j])  # 假设L和W为1

            ax.plot_surface(X, Y, Z, cmap='viridis')
            ax.set_xlabel(var1)
            ax.set_ylabel(var2)
            ax.set_zlabel('ACP')
            ax.set_title(f'在输入变量{var1}和{var2}影响下的模糊规则结果')

            plt.show()


# 示例使用
# fuzzy_system = FuzzySystemACP()
# wcij =1
# acij =1
# tcij =1
# ACPij = fuzzy_system.calculate_ACPij(wcij, acij, tcij)
# print(f"计算得到的ACPij: {ACPij}")
# fuzzy_system.view()
# fuzzy_system.plot_membership_functions()



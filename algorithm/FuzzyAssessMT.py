import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import itertools
plt.rcParams['font.family'] = ['Microsoft YaHei']
class FuzzySystemMT:

    def __init__(self,cz=12000,cl=9000,cw=3000,ch=3000):
        self.cz=  cz
        self.cl = cl
        self.cw = cw
        self.ch = ch
        # 定义模糊变量
        self.Z = ctrl.Antecedent(np.arange(1, 15001, 1), 'Z')
        self.H = ctrl.Antecedent(np.arange(1, 3201, 1), 'H')
        self.L = ctrl.Antecedent(np.arange(1, 9001, 1), 'L')
        self.W = ctrl.Antecedent(np.arange(1, 3201, 1), 'W')

        # 定义输出变量
        self.MT = ctrl.Consequent(np.arange(0, 7.001, 0.001), 'MT')

        # 定义隶属度函数
        self.Z['light'] = fuzz.trimf(self.Z.universe, [1, 1000, 1000]) # TisaiYu[2024/9/4] universe返回一个ctrl的输入变量或者输出变量的定义域取值范围数组
        self.Z['medium'] = fuzz.trimf(self.Z.universe, [501, 2500, 2500])
        self.Z['large'] = fuzz.trimf(self.Z.universe, [2001, 15000, 15000])

        self.H['small'] = fuzz.trimf(self.H.universe, [1, 500, 500])
        self.H['medium'] = fuzz.trimf(self.H.universe, [351, 1500, 1500])
        self.H['large'] = fuzz.trimf(self.H.universe, [1251, 3200, 3200])

        self.L['small'] = fuzz.trimf(self.L.universe, [1, 1500, 1500])
        self.L['medium'] = fuzz.trimf(self.L.universe, [1051, 3750, 3750])
        self.L['large'] = fuzz.trimf(self.L.universe, [3001, 9000, 9000])

        self.W['small'] = fuzz.trimf(self.W.universe, [1, 500, 500])
        self.W['medium'] = fuzz.trimf(self.W.universe, [351, 1500, 1500])
        self.W['large'] = fuzz.trimf(self.W.universe, [1251, 3200, 3200])

        self.MT['low'] = fuzz.trimf(self.MT.universe, [0.6, 2, 2.2])
        self.MT['moderate'] = fuzz.trimf(self.MT.universe, [1.2, 4, 4.2])
        self.MT['high'] = fuzz.trimf(self.MT.universe, [3.5, 7, 7])
        self.MT['low_low'] = fuzz.trimf(self.MT.universe, [0, 0.8, 1])

        # 定义规则
        self.rules = [
            ctrl.Rule(self.Z['light'] & self.H['small'] & self.L['small'] & self.W['small'], self.MT['low_low']),  # 1 1 1 1, 1
            ctrl.Rule(self.Z['light'] & self.H['small'] & self.L['small'] & self.W['medium'], self.MT['low']),  # 1 1 1 2, 1
            ctrl.Rule(self.Z['light'] & self.H['small'] & self.L['small'] & self.W['large'], self.MT['low']),  # 1 1 1 3, 1
            ctrl.Rule(self.Z['light'] & self.H['small'] & self.L['medium'] & self.W['small'], self.MT['low']),  # 1 1 2 1, 1
            ctrl.Rule(self.Z['light'] & self.H['small'] & self.L['medium'] & self.W['medium'], self.MT['low']),  # 1 1 2 2, 1
            ctrl.Rule(self.Z['light'] & self.H['small'] & self.L['medium'] & self.W['large'], self.MT['moderate']),  # 1 1 2 3, 2
            ctrl.Rule(self.Z['light'] & self.H['small'] & self.L['large'] & self.W['small'], self.MT['moderate']),  # 1 1 3 1, 2
            ctrl.Rule(self.Z['light'] & self.H['small'] & self.L['large'] & self.W['medium'], self.MT['moderate']),  # 1 1 3 2, 2
            ctrl.Rule(self.Z['light'] & self.H['small'] & self.L['large'] & self.W['large'], self.MT['high']),  # 1 1 3 3, 3
            ctrl.Rule(self.Z['light'] & self.H['medium'] & self.L['small'] & self.W['small'], self.MT['low']),  # 1 2 1 1, 1
            ctrl.Rule(self.Z['light'] & self.H['medium'] & self.L['small'] & self.W['medium'], self.MT['low']),  # 1 2 1 2, 1
            ctrl.Rule(self.Z['light'] & self.H['medium'] & self.L['small'] & self.W['large'], self.MT['moderate']),  # 1 2 1 3, 2
            ctrl.Rule(self.Z['light'] & self.H['medium'] & self.L['medium'] & self.W['small'], self.MT['low']),  # 1 2 2 1, 1
            ctrl.Rule(self.Z['light'] & self.H['medium'] & self.L['medium'] & self.W['medium'], self.MT['moderate']),  # 1 2 2 2, 2
            ctrl.Rule(self.Z['light'] & self.H['medium'] & self.L['medium'] & self.W['large'], self.MT['moderate']),  # 1 2 2 3, 2
            ctrl.Rule(self.Z['light'] & self.H['medium'] & self.L['large'] & self.W['small'], self.MT['moderate']),  # 1 2 3 1, 2
            ctrl.Rule(self.Z['light'] & self.H['medium'] & self.L['large'] & self.W['medium'], self.MT['moderate']),  # 1 2 3 2, 2
            ctrl.Rule(self.Z['light'] & self.H['medium'] & self.L['large'] & self.W['large'], self.MT['high']),  # 1 2 3 3, 3
            ctrl.Rule(self.Z['light'] & self.H['large'] & self.L['small'] & self.W['small'], self.MT['moderate']),  # 1 3 1 1, 2
            ctrl.Rule(self.Z['light'] & self.H['large'] & self.L['small'] & self.W['medium'], self.MT['moderate']),  # 1 3 1 2, 2
            ctrl.Rule(self.Z['light'] & self.H['large'] & self.L['small'] & self.W['large'], self.MT['high']),  # 1 3 1 3, 3
            ctrl.Rule(self.Z['light'] & self.H['large'] & self.L['medium'] & self.W['small'], self.MT['moderate']),  # 1 3 2 1, 2
            ctrl.Rule(self.Z['light'] & self.H['large'] & self.L['medium'] & self.W['medium'], self.MT['high']),  # 1 3 2 2, 3
            ctrl.Rule(self.Z['light'] & self.H['large'] & self.L['medium'] & self.W['large'], self.MT['high']),  # 1 3 2 3, 3
            ctrl.Rule(self.Z['light'] & self.H['large'] & self.L['large'] & self.W['small'], self.MT['high']),  # 1 3 3 1, 3
            ctrl.Rule(self.Z['light'] & self.H['large'] & self.L['large'] & self.W['medium'], self.MT['high']),  # 1 3 3 2, 3
            ctrl.Rule(self.Z['light'] & self.H['large'] & self.L['large'] & self.W['large'], self.MT['high']),  # 1 3 3 3, 3
            ctrl.Rule(self.Z['medium'] & self.H['small'] & self.L['small'] & self.W['small'], self.MT['low']),  # 2 1 1 1, 1
            ctrl.Rule(self.Z['medium'] & self.H['small'] & self.L['small'] & self.W['medium'], self.MT['moderate']),  # 2 1 1 2, 2
            ctrl.Rule(self.Z['medium'] & self.H['small'] & self.L['small'] & self.W['large'], self.MT['moderate']),  # 2 1 1 3, 2
            ctrl.Rule(self.Z['medium'] & self.H['small'] & self.L['medium'] & self.W['small'], self.MT['low']),  # 2 1 2 1, 1
            ctrl.Rule(self.Z['medium'] & self.H['small'] & self.L['medium'] & self.W['medium'], self.MT['moderate']),  # 2 1 2 2, 2
            ctrl.Rule(self.Z['medium'] & self.H['small'] & self.L['medium'] & self.W['large'], self.MT['high']),  # 2 1 2 3, 3
            ctrl.Rule(self.Z['medium'] & self.H['small'] & self.L['large'] & self.W['small'], self.MT['moderate']),  # 2 1 3 1, 2
            ctrl.Rule(self.Z['medium'] & self.H['small'] & self.L['large'] & self.W['medium'], self.MT['high']),  # 2 1 3 2, 3
            ctrl.Rule(self.Z['medium'] & self.H['small'] & self.L['large'] & self.W['large'], self.MT['high']),  # 2 1 3 3, 3
            ctrl.Rule(self.Z['medium'] & self.H['medium'] & self.L['small'] & self.W['small'], self.MT['low']),  # 2 2 1 1, 1
            ctrl.Rule(self.Z['medium'] & self.H['medium'] & self.L['small'] & self.W['medium'], self.MT['low']),  # 2 2 1 2, 1
            ctrl.Rule(self.Z['medium'] & self.H['medium'] & self.L['small'] & self.W['large'], self.MT['moderate']),  # 2 2 1 3, 2
            ctrl.Rule(self.Z['medium'] & self.H['medium'] & self.L['medium'] & self.W['small'], self.MT['low']),  # 2 2 2 1, 1
            ctrl.Rule(self.Z['medium'] & self.H['medium'] & self.L['medium'] & self.W['medium'], self.MT['moderate']),  # 2 2 2 2, 2
            ctrl.Rule(self.Z['medium'] & self.H['medium'] & self.L['medium'] & self.W['large'], self.MT['high']),  # 2 2 2 3, 3
            ctrl.Rule(self.Z['medium'] & self.H['medium'] & self.L['large'] & self.W['small'], self.MT['moderate']),  # 2 2 3 1, 2
            ctrl.Rule(self.Z['medium'] & self.H['medium'] & self.L['large'] & self.W['medium'], self.MT['moderate']),  # 2 2 3 2, 2
            ctrl.Rule(self.Z['medium'] & self.H['medium'] & self.L['large'] & self.W['large'], self.MT['high']),  # 2 2 3 3, 3
            ctrl.Rule(self.Z['medium'] & self.H['large'] & self.L['small'] & self.W['small'], self.MT['moderate']),  # 2 3 1 1, 2
            ctrl.Rule(self.Z['medium'] & self.H['large'] & self.L['small'] & self.W['medium'], self.MT['moderate']),  # 2 3 1 2, 2
            ctrl.Rule(self.Z['medium'] & self.H['large'] & self.L['small'] & self.W['large'], self.MT['high']),  # 2 3 1 3, 3
            ctrl.Rule(self.Z['medium'] & self.H['large'] & self.L['medium'] & self.W['small'], self.MT['high']),  # 2 3 2 1, 3
            ctrl.Rule(self.Z['medium'] & self.H['large'] & self.L['medium'] & self.W['medium'], self.MT['high']),  # 2 3 2 2, 3
            ctrl.Rule(self.Z['medium'] & self.H['large'] & self.L['medium'] & self.W['large'], self.MT['high']),  # 2 3 2 3, 3
            ctrl.Rule(self.Z['medium'] & self.H['large'] & self.L['large'] & self.W['small'], self.MT['high']),  # 2 3 3 1, 3
            ctrl.Rule(self.Z['medium'] & self.H['large'] & self.L['large'] & self.W['medium'], self.MT['high']),  # 2 3 3 2, 3
            ctrl.Rule(self.Z['medium'] & self.H['large'] & self.L['large'] & self.W['large'], self.MT['high']),  # 2 3 3 3, 3
            ctrl.Rule(self.Z['large'] & self.H['small'] & self.L['small'] & self.W['small'], self.MT['low']),  # 3 1 1 1, 1
            ctrl.Rule(self.Z['large'] & self.H['small'] & self.L['small'] & self.W['medium'], self.MT['moderate']),  # 3 1 1 2, 2
            ctrl.Rule(self.Z['large'] & self.H['small'] & self.L['small'] & self.W['large'], self.MT['high']),  # 3 1 1 3, 3
            ctrl.Rule(self.Z['large'] & self.H['small'] & self.L['medium'] & self.W['small'], self.MT['moderate']),  # 3 1 2 1, 2
            ctrl.Rule(self.Z['large'] & self.H['small'] & self.L['medium'] & self.W['medium'], self.MT['high']),  # 3 1 2 2, 3
            ctrl.Rule(self.Z['large'] & self.H['small'] & self.L['medium'] & self.W['large'], self.MT['high']),  # 3 1 2 3, 3
            ctrl.Rule(self.Z['large'] & self.H['small'] & self.L['large'] & self.W['small'], self.MT['high']),  # 3 1 3 1, 3
            ctrl.Rule(self.Z['large'] & self.H['small'] & self.L['large'] & self.W['medium'], self.MT['high']),  # 3 1 3 2, 3
            ctrl.Rule(self.Z['large'] & self.H['small'] & self.L['large'] & self.W['large'], self.MT['high']),  # 3 1 3 3, 3
            ctrl.Rule(self.Z['large'] & self.H['medium'] & self.L['small'] & self.W['small'], self.MT['moderate']),  # 3 2 1 1, 2
            ctrl.Rule(self.Z['large'] & self.H['medium'] & self.L['small'] & self.W['medium'], self.MT['high']),  # 3 2 1 2, 3
            ctrl.Rule(self.Z['large'] & self.H['medium'] & self.L['small'] & self.W['large'], self.MT['high']),  # 3 2 1 3, 3
            ctrl.Rule(self.Z['large'] & self.H['medium'] & self.L['medium'] & self.W['small'], self.MT['high']),  # 3 2 2 1, 3
            ctrl.Rule(self.Z['large'] & self.H['medium'] & self.L['medium'] & self.W['medium'], self.MT['high']),  # 3 2 2 2, 3
            ctrl.Rule(self.Z['large'] & self.H['medium'] & self.L['medium'] & self.W['large'], self.MT['high']),  # 3 2 2 3, 3
            ctrl.Rule(self.Z['large'] & self.H['medium'] & self.L['large'] & self.W['small'], self.MT['high']),  # 3 2 3 1, 3
            ctrl.Rule(self.Z['large'] & self.H['medium'] & self.L['large'] & self.W['medium'], self.MT['high']),  # 3 2 3 2, 3
            ctrl.Rule(self.Z['large'] & self.H['medium'] & self.L['large'] & self.W['large'], self.MT['high']),  # 3 2 3 3, 3
            ctrl.Rule(self.Z['large'] & self.H['large'] & self.L['small'] & self.W['small'], self.MT['high']),  # 3 3 1 1, 3
            ctrl.Rule(self.Z['large'] & self.H['large'] & self.L['small'] & self.W['medium'], self.MT['high']),  # 3 3 1 2, 3
            ctrl.Rule(self.Z['large'] & self.H['large'] & self.L['small'] & self.W['large'], self.MT['high']),  # 3 3 1 3, 3
            ctrl.Rule(self.Z['large'] & self.H['large'] & self.L['medium'] & self.W['small'], self.MT['high']),  # 3 3 2 1, 3
            ctrl.Rule(self.Z['large'] & self.H['large'] & self.L['medium'] & self.W['medium'], self.MT['high']),  # 3 3 2 2, 3
            ctrl.Rule(self.Z['large'] & self.H['large'] & self.L['medium'] & self.W['large'], self.MT['high']),  # 3 3 2 3, 3
            ctrl.Rule(self.Z['large'] & self.H['large'] & self.L['large'] & self.W['small'], self.MT['high']),  # 3 3 3 1, 3
            ctrl.Rule(self.Z['large'] & self.H['large'] & self.L['large'] & self.W['medium'], self.MT['high']),  # 3 3 3 2, 3
            ctrl.Rule(self.Z['large'] & self.H['large'] & self.L['large'] & self.W['large'], self.MT['high']),  # 3 3 3 3, 3
        ]

        # 创建控制系统
        self.cp_control_system = ctrl.ControlSystem(self.rules)
        self.cp_simulation = ctrl.ControlSystemSimulation(self.cp_control_system)

    def t_norm_min(self, a, b):
        return np.minimum(a, b)

    def mamdani_aggregation(self, inputs):
        # 初始化输出隶属度
        aggregated_output = np.zeros_like(self.MT.universe)

        # 遍历所有规则
        for rule in self.rules:
            # 计算当前规则的激活度
            rule_activation = np.min([
                inputs['Z'][rule.antecedent_terms[0].label],
                inputs['H'][rule.antecedent_terms[1].label],
                inputs['L'][rule.antecedent_terms[2].label],
                inputs['W'][rule.antecedent_terms[3].label]
            ])

            # 遍历规则的所有后件模糊集
            for consequent in rule.consequent:
                aggregated_output = np.maximum(aggregated_output, rule_activation * consequent.term.mf)

        return aggregated_output

    def defuzzify_centroid(self, aggregated):
        numerator = np.sum(aggregated * self.MT.universe)
        denominator = np.sum(aggregated)
        if denominator == 0:
            return 0
        return round(numerator / denominator, 3)

    def calculate_MT(self, z, l, w, h):
        penalty = 0
        if z>self.cz:
            penalty += (z-self.cz)/self.cz
            z = self.cz
        if l>self.cl:
            penalty += (l - self.cl) / self.cl
            l = self.cl
        if w>self.cw:
            penalty += (w - self.cw) / self.cw
            w = self.cw
        if h>self.ch:
            penalty += (h - self.ch) / self.ch
            h = self.ch
        inputs = {
            'Z': {
                'light': fuzz.interp_membership(self.Z.universe, self.Z['light'].mf, z), # TisaiYu[2024/9/4] mf返回一个trimf对象的mf函数，是一个数组，表示定义域内的三角形隶属度取值
                'medium': fuzz.interp_membership(self.Z.universe, self.Z['medium'].mf, z),
                'large': fuzz.interp_membership(self.Z.universe, self.Z['large'].mf, z)
            },
            'H': {
                'small': fuzz.interp_membership(self.H.universe, self.H['small'].mf, h),
                'medium': fuzz.interp_membership(self.H.universe, self.H['medium'].mf, h),
                'large': fuzz.interp_membership(self.H.universe, self.H['large'].mf, h)
            },
            'L': {
                'small': fuzz.interp_membership(self.L.universe, self.L['small'].mf, l),
                'medium': fuzz.interp_membership(self.L.universe, self.L['medium'].mf, l),
                'large': fuzz.interp_membership(self.L.universe, self.L['large'].mf, l)
            },
            'W': {
                'small': fuzz.interp_membership(self.W.universe, self.W['small'].mf, w),
                'medium': fuzz.interp_membership(self.W.universe, self.W['medium'].mf, w),
                'large': fuzz.interp_membership(self.W.universe, self.W['large'].mf, w)
            }
        }

        self.cp_simulation.input['Z'] = z
        self.cp_simulation.input['L'] = l
        self.cp_simulation.input['H'] = h
        self.cp_simulation.input['W'] = w

        self.cp_simulation.compute()
        MT = self.cp_simulation.output['MT']
        # aggregated = self.mamdani_aggregation(inputs)
        # MT = self.defuzzify_centroid(aggregated)
        # MT = MT
        return MT

    def view(self):
        # 显示隶属度函数
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))

        axs[0, 0].plot(self.Z.universe, self.Z['light'].mf, 'b', label='低')
        axs[0, 0].plot(self.Z.universe, self.Z['medium'].mf, 'g', label='中')
        axs[0, 0].plot(self.Z.universe, self.Z['large'].mf, 'r', label='高')
        axs[0, 0].set_title('Z Membership Functions')
        axs[0, 0].legend()

        axs[0, 1].plot(self.H.universe, self.H['small'].mf, 'b', label='低')
        axs[0, 1].plot(self.H.universe, self.H['medium'].mf, 'g', label='中')
        axs[0, 1].plot(self.H.universe, self.H['large'].mf, 'r', label='高')
        axs[0, 1].set_title('H Membership Functions')
        axs[0, 1].legend()

        axs[0, 2].plot(self.L.universe, self.L['small'].mf, 'b', label='低')
        axs[0, 2].plot(self.L.universe, self.L['medium'].mf, 'g', label='中')
        axs[0, 2].plot(self.L.universe, self.L['large'].mf, 'r', label='高')
        axs[0, 2].set_title('L Membership Functions')
        axs[0, 2].legend()

        axs[1, 0].plot(self.W.universe, self.W['small'].mf, 'b', label='低')
        axs[1, 0].plot(self.W.universe, self.W['medium'].mf, 'g', label='中')
        axs[1, 0].plot(self.W.universe, self.W['large'].mf, 'r', label='高')
        axs[1, 0].set_title('W Membership Functions')
        axs[1, 0].legend()

        axs[1, 1].plot(self.MT.universe, self.MT['low_low'].mf, 'c', label='极低')
        axs[1, 1].plot(self.MT.universe, self.MT['low'].mf, 'b', label='低')
        axs[1, 1].plot(self.MT.universe, self.MT['moderate'].mf, 'g', label='中')
        axs[1, 1].plot(self.MT.universe, self.MT['high'].mf, 'r', label='高')
        axs[1, 1].set_title('MT Membership Functions')
        axs[1, 1].legend()

        plt.tight_layout()
        plt.show()

        input_variables = {"Z": self.Z.universe, "H": self.H.universe, "L": self.L.universe,"W":self.W.universe}
        input_variables_name = ["Z", "H", "L","W"]

        combination = itertools.combinations(input_variables_name, 2)

        for select_show in combination:
            params_dict = {}
            var1 = select_show[0]
            var2 = select_show[1]
            var3 = None
            var4 = None
            for var in input_variables_name:
                if var not in [var1, var2] and var3==None:
                    var3 = var
                    continue
                if var not in [var1, var2] and var4==None:
                    var4 = var


            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            X, Y = np.meshgrid(input_variables[var1][5::input_variables[var1].shape[0]//60], input_variables[var2][5::input_variables[var2].shape[0]//60])
            Z = np.zeros_like(X).astype(np.float32)
            params_dict[var1] = X
            params_dict[var2] = Y
            params_dict[var3] = np.ones_like(X)*400
            params_dict[var4] = np.ones_like(X)*400
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i, j] = self.calculate_MT(params_dict['Z'][i, j], params_dict['L'][i, j],
                                                   params_dict['W'][i, j],params_dict['H'][i, j])  # 假设L和W为1

            ax.plot_surface(X, Y, Z, cmap='viridis')
            ax.set_xlabel(var1)
            ax.set_ylabel(var2)
            ax.set_zlabel('MT')
            ax.set_title(f'在输入变量{var1}和{var2}影响下的模糊规则结果')

            plt.show()


# 示例使用
# fuzzy_system = FuzzySystemMT(12000,9000,3000,3000)
# z = 150
# h = 400
# l = 2900
# w = 400
#
# MT = fuzzy_system.calculate_MT(z, l, w, h)
# print(f"计算得到的MT: {MT}")
# fuzzy_system.view()


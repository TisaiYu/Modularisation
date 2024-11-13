import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class MTP16:
    def __init__(self):
        # 定义输入变量
        self.Z = ctrl.Antecedent(np.arange(1, 7, 1), 'Z')
        self.H = ctrl.Antecedent(np.arange(1, 7, 1), 'H')
        self.L = ctrl.Antecedent(np.arange(1, 7, 1), 'L')

        # 定义输出变量
        self.MT = ctrl.Consequent(np.arange(1, 7, 1), 'MT')

        # 定义隶属度函数
        self.Z['low'] = fuzz.trimf(self.Z.universe, [1, 1, 3])
        self.Z['medium'] = fuzz.trimf(self.Z.universe, [1, 3, 5])
        self.Z['high'] = fuzz.trimf(self.Z.universe, [3, 6, 6])

        self.H['low'] = fuzz.trimf(self.H.universe, [1, 1, 3])
        self.H['medium'] = fuzz.trimf(self.H.universe, [1, 3, 5])
        self.H['high'] = fuzz.trimf(self.H.universe, [3, 6, 6])

        self.L['low'] = fuzz.trimf(self.L.universe, [1, 1, 3])
        self.L['medium'] = fuzz.trimf(self.L.universe, [1, 3, 5])
        self.L['high'] = fuzz.trimf(self.L.universe, [3, 6, 6])

        self.MT['low'] = fuzz.trimf(self.MT.universe, [1, 1, 3])
        self.MT['medium'] = fuzz.trimf(self.MT.universe, [1, 3, 5])
        self.MT['high'] = fuzz.trimf(self.MT.universe, [3, 6, 6])

        # 定义规则
        self.rules = [
            ctrl.Rule(self.Z['low'] & self.H['low'] & self.L['low'], self.MT['low']),
            ctrl.Rule(self.Z['medium'] & self.H['medium'] & self.L['medium'], self.MT['medium']),
            ctrl.Rule(self.Z['high'] & self.H['high'] & self.L['high'], self.MT['high']),
            # 添加更多规则以提高分辨率
        ]

        self.system = ctrl.ControlSystem(self.rules)
        self.simulation = ctrl.ControlSystemSimulation(self.system)

    def calculate_MT(self, z, h, l):
        self.simulation.input['Z'] = z
        self.simulation.input['H'] = h
        self.simulation.input['L'] = l
        self.simulation.compute()
        return self.simulation.output['MT']

# 示例使用
mtp16 = MTP16()
output1 = mtp16.calculate_MT(6, 5, 4)
output2 = mtp16.calculate_MT(5, 5, 4)
print(f"Output for (6, 5, 4): {output1}")
print(f"Output for (5, 5, 4): {output2}")

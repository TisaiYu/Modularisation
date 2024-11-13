from utils.utils import *
'''
因为层次聚类是高度”局部“的，每次聚类都是从最近的聚合为一个类，自底向上。所以用遗传模拟退火来找 更好的。但是这也是有些局部，因为是在层次聚类的一个结
果上进行遗传模拟退火。比如层次聚类结果是4个类，则遗传模拟退火的搜索解空间就是4个类的组合，如果层次聚类结果是8个类，那么搜索空间就是8个类的组合
理论上可以对层次聚类的所有聚类树的节点都进行一次模拟遗传退火，但是那样其实层次聚类就相当于没用了（仅仅是产生初始解而已），而且计算也很庞大。



遗传算法：通过“生物“上的仿照吧，通过对“染色体”进行遗传模拟和筛选。每轮的染色体集合是一个分类方案。在染色体交叉、变异、复制遗传的情况下，计算评估指标，
然后数值较大的有大概率遗传下去，经过迭代直至最终终止。每次迭代的染色体个数是population_size的变量控制的。generations表示迭代轮数。
退火算法：通过模拟金属的退火过程。设置一个初始温度，寻找初始解的临近范围作为新的一轮输入，然后逐渐迭代降低温度，计算评估指标在各个温度下的值以此循环
直到终止条件。
遗传模拟退火HGSA原理：对每一轮的染色体执行模拟退火，增加其能够接受劣解的概率，来提高搜索。
具体流程和原理看代码吧。
'''

class GeneticSimulatedAnnealing:
    def __init__(self, initial_labels,DSM, compute_fitness,population_size=100, generations=500, temp_initial=100, cooling_rate=0.99, crossover_prob=0.75, mutation_prob=0.01,final_temp = 0.1):
        self.initial_labels = initial_labels
        self.compute_fitness = compute_fitness
        self.population_size = population_size
        self.DSM = DSM
        self.generations = generations
        self.temp_initial = temp_initial
        self.final_temp = final_temp
        self.temp = temp_initial
        self.cooling_rate = cooling_rate
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.init_best_fitness = self.compute_fitness(self.initial_labels,self.DSM)
        self.change_class = np.sqrt(self.initial_labels.shape[0])

    def initialize_population_with_cluster_labels(self):
        population = []
        num_original_classes = len(np.unique(self.initial_labels))
        min_classes = max(1, num_original_classes - self.change_class)
        max_classes = num_original_classes + self.change_class
        change_mask = np.random.rand(self.initial_labels.size) < 0.5
        while len(population) < self.population_size:
            new_labels = np.random.randint(-self.change_class, self.change_class+1, size=self.initial_labels.shape)
            new_labels[change_mask] = 0
            new_labels = self.initial_labels+new_labels
            new_labels[np.where(new_labels<1)] = 1
            unique_labels = np.unique(new_labels)
            label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels, 1)}
            new_labels = np.array([label_mapping[label] for label in new_labels])
            # if len(unique_labels) >= min_classes and len(unique_labels) <= max_classes:
            #     # Relabel to ensure continuity from 1 to len(unique_labels)
            #     population.append(new_labels)
            #     continue
            population.append(new_labels)
        return population

    def initialize_population_random(self):
        population = []
        while len(population) < self.population_size:
            new_labels = np.random.randint(1, int(np.sqrt(self.initial_labels.shape)), size=self.initial_labels.shape)
            unique_labels = np.unique(new_labels)
            label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels, 1)}
            new_labels = np.array([label_mapping[label] for label in new_labels])
            population.append(new_labels)
        return population

    def select_gambling(self, population):
        new_pop = []
        fitness_scores = np.array([self.compute_fitness(individual,self.DSM) for individual in population])
        probabilities = fitness_scores / fitness_scores.sum()
        for array, prob in zip(population, probabilities):
            if np.random.uniform() < prob:
                new_pop.append(array)
        # TisaiYu[2024/7/11] 剩下的按照轮盘赌，其实直接用randomchoice是一样的。
        remaining_size = self.population_size-len(new_pop)
        cumulative_probabilities = np.cumsum(probabilities)
        for _ in range(remaining_size):
            r = np.random.uniform()
            index = np.searchsorted(cumulative_probabilities, r)
            new_pop.append(population[index])
        return new_pop

    def crossover(self, population):
        newpop = []
        while len(newpop)<self.population_size:
            parents = np.random.choice(len(population), size=2, replace=False)
            parent1, parent2 = population[parents[0]], population[parents[1]]
            if np.random.rand() < self.crossover_prob:  # 以概率pc对染色体进行交叉
                crossover_point = np.random.randint(1, len(parent1) - 1)

                # TisaiYu[2024/6/28] 交叉片段
                child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))

                # TisaiYu[2024/6/28] 或者交叉对应位置，而不是切片
                # temp_vari = parent2[crossover_point]
                # parent2[crossover_point] = parent1[crossover_point]
                # parent1[crossover_point] = temp_vari
                # child1 = parent1
                # child2 = parent2

                num_original_classes = len(np.unique(self.initial_labels))
                min_classes = max(1, num_original_classes - self.change_class)
                max_classes = num_original_classes + self.change_class
                unique_labels1 = np.unique(child1)
                unique_labels2 = np.unique(child2)
                label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels1, 1)}
                child1 = np.array([label_mapping[label] for label in child1])
                label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels2, 1)}
                child2 = np.array([label_mapping[label] for label in child2])
                # if len(unique_labels1) >= min_classes and len(unique_labels1) <= max_classes:
                #     newpop.append(child1)
                # if len(unique_labels2) >= min_classes and len(unique_labels2) <= max_classes:
                #     newpop.append(child2)
                newpop.append(child1)
                newpop.append(child2)
            else:
                newpop.append(parent1)
                newpop.append(parent2)
        return newpop


    def mutate(self, population): # TisaiYu[2024/7/11] 按照论文里的，概率选择编译个体，然后个体的位置编码随机即打乱，然后变异值小于等于零件数的开方
        new_pop = []
        while len(new_pop) < self.population_size:  # x为种群个数
            individual = population[np.random.choice(range(len(population)))]
            change_range = np.sqrt(individual.shape[0])
            if np.random.rand() < self.mutation_prob:
                np.random.shuffle(individual) # TisaiYu[2024/6/28] 每次种群加入了self.init_labels注意如果恰好shuffle到这个了，会导致结果混乱。因为shuffle相当于引用了，而不是值的拷贝，所以传入self.init_labels传入它的拷贝
                mutation_value = np.random.randint(-change_range//2,change_range//2+1)
                individual = individual + mutation_value
                individual[np.where(individual < 1)] = 1
                unique_labels = np.unique(individual)
                num_original_classes = len(np.unique(self.initial_labels))
                min_classes = max(1, num_original_classes - self.change_class)
                max_classes = num_original_classes + self.change_class
                label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels, 1)}
                individual = np.array([label_mapping[label] for label in individual])
                # if len(unique_labels) >= min_classes and len(unique_labels) <= max_classes:
                #     new_pop.append(individual)
                #     continue
                new_pop.append(individual)
            else:
                new_pop.append(individual)
                # Ensure labels are continuous after mutation
        return new_pop

    def mutate_genes(self, population): # TisaiYu[2024/7/11] 按照一些博客的，对每个个体的每个基因，概率变异，变异为1到类别数-1的随机值
        new_pop = []
        for i in range(len(population)):
            individual = population[i]
            change_range = individual.shape[0]
            for j in range(individual.shape[0]):
                if np.random.rand() < self.mutation_prob:
                    mutation_value = np.random.randint(1,change_range)
                    individual[j] = mutation_value
                    continue
            unique_labels = np.unique(individual)
            num_original_classes = len(np.unique(self.initial_labels))
            min_classes = max(1, num_original_classes - self.change_class)
            max_classes = num_original_classes + self.change_class
            label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels, 1)}
            individual = np.array([label_mapping[label] for label in individual])
            # if len(unique_labels) >= min_classes and len(unique_labels) <= max_classes: # TisaiYu[2024/7/11] 保证在输入初始解的周围
            #     new_pop.append(individual)
            #     continue
            new_pop.append(individual)
                # Ensure labels are continuous after mutation
        return new_pop

    def simulated_annealing(self, old_population,new_population,temp):
        old_fitness = [self.compute_fitness(individual,self.DSM) for individual in old_population]
        new_fitness = [self.compute_fitness(individual,self.DSM) for individual in new_population]
        for i in range(len(old_fitness)):
            if new_fitness[i] > old_fitness[i] or np.random.rand() < np.exp((new_fitness[i] - old_fitness[i]) / temp):
                continue
            else:
                new_population[i] = old_population[i]
        return new_population

    def optimize(self):
        # population = self.initialize_population_with_cluster_labels()
        population = self.initialize_population_random()
        self.temp = self.temp_initial
        # best_solution = self.initial_labels
        # best_fitness = self.init_best_fitness
        best_solution = []
        best_fitness = 0

        for individual in population:
            fitness = self.compute_fitness(individual,self.DSM)
            if fitness > best_fitness:
                best_solution = individual
                best_fitness = fitness

        for generation in range(self.generations):

            new_population = self.select_gambling(population)
            new_population = self.crossover(new_population)
            # new_population = self.mutate(new_population)
            new_population = self.mutate_genes(new_population)
            new_population = self.simulated_annealing(population,new_population,self.temp)
            population = new_population
            # population.pop()# TisaiYu[2024/6/27] 每轮都保证层次聚类的结果在里面，也就是结果不会坏于层次聚类的
            # population.append(np.copy(self.initial_labels)) # TisaiYu[2024/6/27] 每轮都保证层次聚类的结果在里面，也就是结果不会坏于层次聚类的，必须传入拷贝，不然shuffle会导致奇怪的结果
            self.temp *= self.cooling_rate
            # Evaluate the best solution
            for individual in population:
                fitness = self.compute_fitness(individual,self.DSM)
                if fitness > best_fitness:
                    best_solution = individual
                    best_fitness = fitness
            print(population)
            print(f"generation{generation},fitness:{best_fitness},label:{best_solution}")
        # if best_fitness < self.init_best_fitness:
        #     best_solution = self.initial_labels
        #     best_fitness = self.init_best_fitness
        return best_solution, best_fitness

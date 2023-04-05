import pandas as pd
import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import random

data = pd.read_csv("Bank_Personal_Loan_Modelling.csv")
data.drop(['ID'],axis=1 ,inplace=True)
x = data.drop(['Personal Loan'],axis=1).values
y = data['Personal Loan'].values
x = torch.tensor(x,dtype=torch.float64)
y = torch.tensor(y,dtype=torch.float64)
y = y.to(torch.float64)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)


class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(12,10)
        self.layer2 = torch.nn.Linear(10,20)
        self.layer3 = torch.nn.Linear(20,1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x.float())
        x = self.relu(x.float())
        x = self.layer2(x.float())
        x = self.layer3(x.float())
        x = self.relu(x.float())
        x = self.sigmoid(x.float())
        return x

model = NeuralNetwork()
lossfun = torch.nn.MSELoss()

class CulturalAlgorithm:
    def __init__(self, model, population_size, mutation_prob, mutation_sd, inputs, target, k, n):
        self.model = model
        self.population_size = population_size
        self.mutation_prob = mutation_prob
        self.mutation_sd = mutation_sd
        self.population = self.population()
        self.inputs = inputs
        self.target = target
        self.k = k
        self.n = n
        self.current_culture = self.population[:k]
        self.culture_fitness = [0]
        self.probability = 0.95
        self.decay = 0.95
        self.culture = []

    def population(self):
        popl = []
        for i in range(self.population_size):
            weights = []
            for weight in self.model.parameters():
                weights.append(weight.data.numpy())
            popl.append(weights)
        return popl

    def fitness(self, weights):
        for i, param in enumerate(self.model.parameters()):
            param.data = torch.Tensor(weights[i])
        outputs = self.model(self.inputs)
        loss = lossfun(outputs.float(), self.target.reshape([len(self.inputs) , 1]).float())
        return 1 / (1 + loss.item())

    def recombine(self, parent1, parent2):
        parent1 = np.array(parent1)
        parent2 = np.array(parent2)
        alpha = np.random.uniform(low=0.1, high=0.9, size=parent1.shape)
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = alpha * parent2 + (1 - alpha) * parent1
        return child1, child2


    def mutation(self, weight):
        new_weight = weight + np.random.normal(0, self.mutation_sd, size=weight.shape)
        return new_weight

    def select(self, generate_solution):
        fitnesses = [generate_solution(solution) for solution in self.population]
        total_fitness = sum(fitnesses)
        prob = [f/total_fitness for f in fitnesses]
        parents_idx = np.random.choice(len(self.population), size=2, replace=False, p=prob)
        return parents_idx

    def offsprings(self):
        new_popl = []
        while len(new_popl) < self.population_size:
            parent1_idx, parent2_idx = self.select(self.culture_fitness)
            parent1 = self.current_culture[parent1_idx]
            parent2 = self.current_culture[parent2_idx]
            if np.random.uniform() < self.mutation_prob:
                child1, child2 = self.recombine(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                new_popl.append(child1)
                new_popl.append(child2)
            else:
                new_popl.append(parent1)
                new_popl.append(parent2)
        self.population = new_popl

    def update_culture(self):
        all_fitness = [self.fitness(weights) for weights in self.population]
        sorted_indices = np.argsort(all_fitness)[::-1]
        self.current_culture = [self.population[idx] for idx in sorted_indices[:self.k]]
        self.culture_fitness = [self.fitness(weights) for weights in self.current_culture]

   
    def initialize(self):
        self.population = [self.generate_solution() for _ in range(self.population_size)]
        self.culture = []
        self.culture_fitness = []

    def generate_solution(self):
        return [random.randint(0, 1) for _ in range(self.population_size)]

    def run(self, num_generations=50):
        self.initialize()
        for generation in range(num_generations):
            self.evolve()
            self.update_culture()
            if generation % 10 == 0:
                print(f"Generation: {generation}, Best fitness: {max(self.fitnesses)}")
        return max(self.fitness)

    def evolve(self):
        children = []
        while len(children) < self.population_size:
            parent1, parent2 = self.select(self.fitness)
            child = self.recombine(parent1, parent2)
            child = self.mutate(child)
            children.append(child)
        self.population = children


ca = CulturalAlgorithm(model=model,population_size=100,mutation_prob=0.1,mutation_sd=0.1,inputs=x_train,target=y_train,k=20,n=10)

ca.run(num_generations=50)

with torch.no_grad():
    outputs = model(x_test)
    predicted = torch.round(outputs)
    correct = (predicted == y_test.reshape([len(x_test), 1]))
    accuracy = torch.mean(correct.float())
    print('Accuracy:', accuracy.item())

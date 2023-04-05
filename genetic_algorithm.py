import pandas as pd
import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv("Bank_Personal_Loan_Modelling.csv")
data.drop(['ID'],axis=1 ,inplace=True)
x = data.drop(['Personal Loan'],axis=1).values
y = data['Personal Loan'].values
x = torch.tensor(x,dtype=torch.float64)
y = torch.tensor(y,dtype=torch.float64)
y = y.to(torch.float64)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)


class NeuralNetwork(nn.Module):
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

class GeneticAlgorithm:
    def __init__(self, model, population_size,mutation,decay,inputs,target):
        self.model = model
        self.population_size = population_size
        self.mutation = mutation
        self.population = self.population()
        self.decay = decay
        self.inputs = inputs
        self.target = target

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

    def mutate(self, weights):
        new_weights = []
        for w in weights:
            new_w = w + np.random.normal(0, 0.1, size=w.shape)
            new_weights.append(new_w)
        return new_weights

    def crossover(self, weights1, weights2):
        cross = np.random.randint(1, len(weights1))
        new_weights1 = weights1[:cross] + weights2[cross:]
        new_weights2 = weights1[:cross] + weights2[cross:]
        return new_weights2, new_weights2

    def select(self, fitness):
        cumsum = np.cumsum(fitness)
        total_fitness = sum(fitness)
        rand = np.random.uniform(total_fitness)
        parents_idx = np.searchsorted(cumsum, rand)
        return parents_idx
        
    
    def decayRate(self):
        self.mutation -= (self.decay*self.mutation)

    
    def offsprings(self, fitnesses):
        new_popl = []
        for i in range(self.population_size):
            parent1_idx = self.select(fitnesses)
            parent2_idx = self.select(fitnesses)
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            new_popl.append(child1)
            new_popl.append(child2)
        self.population = new_popl

    def modifyWeight(self):
        fitnesses = [self.fitness(weights) for weights in self.population]
        best_idx = np.argmax(fitnesses)
        best_weights = self.population[best_idx]
        for i, param in enumerate(self.model.parameters()):
            param.data = torch.Tensor(best_weights[i])

ga = GeneticAlgorithm(model, population_size=20, mutation=0.3,decay = 0.05, inputs = x_train, target = y_train)

num_epochs = 1000

loss_list = []
with tf.device('/gpu:0'):
    for epoch in range(num_epochs):
        ga.offsprings([])
        ga.modifyWeight()
        outputs = model(x_train)
        loss = lossfun(outputs, y_train.reshape([len(x_train) , 1]).float())
        loss_list.append(loss.item())
        loss.backward()
        ga.offsprings([])
        ga.modifyWeight()
        if (epoch%10 == 0):
            print("Epoch" , epoch , " : " , loss.item());
            ga.decayRate()

with torch.no_grad():
    outputs = model(x_test)
    predicted = torch.round(outputs)
    correct = (predicted == y_test.reshape([len(x_test), 1]))
    accuracy = torch.mean(correct.float())
    print('Accuracy:', accuracy.item())

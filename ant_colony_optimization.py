import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random


class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(12, 10 , bias = False)
        self.linear2 = torch.nn.Linear(10, 20 , bias = False)
        self.linear3 = torch.nn.Linear(20 , 1 , bias = False)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x.float())
        x = self.relu(x.float())
        x = self.linear2(x.float())
        x = self.linear3(x.float())
        x = self.relu(x.float())
        x = self.sigmoid(x.float())
        return x

model = NN()
loss_function = torch.nn.MSELoss()


class AntColonyOptimizer:
    def _init_(self, num_ants, num_iterations, alpha, beta, rho, Q, bounds, fitness_fn):
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.bounds = bounds
        self.fitness_fn = fitness_fn
        
    def optimize(self):
        pheromone_matrix = torch.ones(self.bounds.shape)
        best_fitness = float('-inf')
        best_weights = None
        
        for i in range(self.num_iterations):
            ant_positions = torch.zeros((self.num_ants, self.bounds.shape[0]))
            for j in range(self.num_ants):
                current_node = np.random.randint(0, self.bounds.shape[0])
                ant_positions[j, current_node] = 1
                
                while not torch.all(ant_positions[j]):
                    probabilities = torch.zeros(self.bounds.shape[0])
                    unvisited_nodes = torch.where(ant_positions[j] == 0)[0]
                    pheromone_sum = torch.sum(pheromone_matrix[current_node, unvisited_nodes])
                    for k in range(self.bounds.shape[0]):
                        if ant_positions[j, k] == 0:
                            probabilities[k] = pheromone_matrix[current_node, k] / pheromone_sum if pheromone_sum > 0 else 0
                    probabilities = probabilities ** self.alpha
                    heuristic_info = self.fitness_fn(self.bounds).reshape(-1)
                    probabilities = heuristic_info * self.beta
                    probabilities = probabilities / torch.sum(probabilities)
                    
                    next_node = np.random.choice(np.arange(self.bounds.shape[0]), p=probabilities.numpy())
                    ant_positions[j, next_node] = 1
                    current_node = next_node
            
            fitness_values = self.fitness_fn(ant_positions)
            pheromone_matrix *= (1 - self.rho)
            for j in range(self.num_ants):
                for k in range(self.bounds.shape[0]):
                    pheromone_matrix[k, int(torch.where(ant_positions[j])[0])] += self.Q * fitness_values[j]
            
            if torch.max(fitness_values) > best_fitness:
                best_fitness = torch.max(fitness_values)
                best_weights = self.bounds[int(torch.argmax(fitness_values))]
        
        return best_weights

def evaluate_fitness(weights):
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 2)
    )
    model.load_state_dict({'0.weight': torch.tensor(weights[:128]).reshape(16, 8),
                           '0.bias': torch.tensor(weights[128:144]),
                           '2.weight': torch.tensor(weights[144:208]).reshape(8, 16),
        '2.bias': torch.tensor(weights[208:])
    })

predictions = []
with torch.no_grad():
    for inputs, _ in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.numpy())
predictions = np.array(predictions)
accuracy = np.mean(predictions == y_test.numpy())


return accuracy


data = pd.read_csv('Bank_Personal_Loan_Modelling.csv')
X = data.drop(['Personal Loan'],axis=1).values
y = data['Personal Loan'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_loader = DataLoader(list(zip(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))), batch_size=32)
test_loader = DataLoader(list(zip(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))), batch_size=32)

bounds = np.concatenate([layer.weight.data.numpy().flatten() for layer in model])
bounds = np.tile(bounds, (10, 1))

aco = AntColonyOptimizer(num_ants=10, num_iterations=100, alpha=1, beta=2, rho=0.1, Q=1, bounds=bounds, fitness_fn=evaluate_fitness)
best_weights = aco.optimize()

model.load_state_dict({'0.weight': torch.tensor(best_weights[:128]).reshape(16, 8),
'0.bias': torch.tensor(best_weights[128:144]),
'2.weight': torch.tensor(best_weights[144:208]).reshape(8, 16),
'2.bias': torch.tensor(best_weights[208:])
})
predictions = []
with torch.no_grad():
for inputs, _ in test_loader:
outputs = model(inputs)
_, predicted = torch.max(outputs, 1)
predictions.extend(predicted.numpy())
predictions = np.array(predictions)
print('Accuracy:', np.mean(predictions == y_test.numpy()))


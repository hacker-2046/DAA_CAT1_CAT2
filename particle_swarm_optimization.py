import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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

class Particle:
    def _init_(self, num_weights, bounds):
        self.position = torch.FloatTensor(num_weights).uniform_(*bounds)
        self.velocity = torch.zeros(num_weights)
        self.best_position = self.position.clone()
        self.best_fitness = float('-inf')
        
    def update_position(self):
        self.position += self.velocity
        
    def clip_position(self, bounds):
        self.position.clamp_(*bounds)
        
    def evaluate_fitness(self, evaluate_fn):
        fitness = evaluate_fn(self.position)
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_position = self.position.clone()
        return fitness
    
class ParticleSwarmOptimizer:
    def _init_(self, num_particles, bounds, evaluate_fn, learning_rate=0.01, inertia_weight=0.5, cognitive_weight=1.0, social_weight=1.0):
        self.num_particles = num_particles
        self.bounds = bounds
        self.evaluate_fn = evaluate_fn
        self.learning_rate = learning_rate
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        
        self.particles = [Particle(len(bounds), bounds) for _ in range(num_particles)]
        self.global_best_position = None
        self.global_best_fitness = float('-inf')
        
    def optimize(self, num_iterations):
        for i in range(num_iterations):
            for particle in self.particles:
                cognitive_component = self.cognitive_weight * (particle.best_position - particle.position)
                social_component = self.social_weight * (self.global_best_position - particle.position)
                particle.velocity = self.inertia_weight * particle.velocity + cognitive_component + social_component
                
                particle.update_position()
                particle.clip_position(self.bounds)
                
                fitness = particle.evaluate_fitness(self.evaluate_fn)
                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.clone()
                    
    def get_best_position(self):
        return self.global_best_position
    
    def get_best_fitness(self):
        return self.global_best_fitness

model = nn.Sequential(
    nn.Linear(8, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 2)
)

# Load the data
data = pd.read_csv("Bank_Personal_Loan_Modelling.csv")
data.drop(['ID'],axis=1 ,inplace=True)
x = data.drop(['Personal Loan'],axis=1).values
y = data['Personal Loan'].values
y = torch.tensor(y, dtype=torch.long)
train_size = int(0.8 * len(data))
x_train, y_train = x[:train_size], y[:train_size]
x_test, y_test = x[train_size:], y[train_size:]

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
'2.bias': torch.tensor(weights[208:216]),
'4.weight': torch.tensor(weights[216:]).reshape(2, 8),
'4.bias': torch.tensor(weights[-2:])})
model.eval()

with torch.no_grad():
    outputs = model(torch.tensor(x_test, dtype=torch.float32))
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == y_test).sum().item()
    accuracy = correct / y_test.size(0)
    
return accuracy

num_particles = 20
bounds = (-1, 1)
optimizer = ParticleSwarmOptimizer(num_particles, bounds, evaluate_fitness)

num_iterations = 100
optimizer.optimize(num_iterations)

best_position = optimizer.get_best_position()
model.load_state_dict({'0.weight': torch.tensor(best_position[:128]).reshape(16, 8),
'0.bias': torch.tensor(best_position[128:144]),
'2.weight': torch.tensor(best_position[144:208]).reshape(8, 16),
'2.bias': torch.tensor(best_position[208:216]),
'4.weight': torch.tensor(best_position[216:]).reshape(2, 8),
'4.bias': torch.tensor(best_position[-2:])})

with torch.no_grad():
outputs = model(torch.tensor(x_test, dtype=torch.float32))
_, predicted = torch.max(outputs, 1)
correct = (predicted == y_test).sum().item()
accuracy = correct / y_test.size(0)
print("Test accuracy:", accuracy)

new_data = torch.tensor([0, 1, 0, 0, 0, 1, 0, 1], dtype=torch.float32).unsqueeze(0)
outputs = model(new_data)
_, predicted = torch.max(outputs, 1)
print("Predicted class:", predicted.item())


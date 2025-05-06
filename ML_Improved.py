"""
Neural Network-Based Lennard-Jones Potential Prediction System.

This module performs a Monte Carlo simulation of particle interactions 
using the Lennard-Jones potential. It then trains a neural network 
to predict the system's potential energy based on particle positions.

An improved version of the system also explores using **pairwise distances (inter-particle distances)** 
as input features to achieve better symmetry properties and improved model performance.

Author: [Yinan HU]
Date: [06/05/2025]
"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Constants
L = 8  # box size
Np = 20  # number of particles
kT = 0.5  # temperature (thermal energy)
dmax = 3  # maximum distance to move a particle in one MC step
epsilon = 1  # LJ interaction prefactor
sigma = 1  # LJ interaction particle size
sigma12 = sigma**12
sigma6 = sigma**6
MCsteps = 20000  # number of MC steps

# Function to calculate inter-particle distances
def calculate_distances(positions, L):
    """
    Calculate sorted inter-particle distances for a given configuration.
    :param positions: Array of particle positions (N, 2).
    :param L: Box size for periodic boundary conditions.
    :return: Sorted array of inter-particle distances.
    """
    distances = []
    N = positions.shape[0]
    for i in range(N):
        for j in range(i + 1, N):
            dist_vec = positions[j, :] - positions[i, :]
            dist_vec -= L * np.rint(dist_vec / L)  # Nearest periodic image
            dist = np.linalg.norm(dist_vec)
            distances.append(dist)
    return np.sort(distances)

# Save distances and target energies
distances = []
target_energies = []

# Initialize the system configuration
pos = np.zeros((Np, 2))
NperL = int(np.ceil(np.sqrt(Np)))  # number of rows, columns
a = L / NperL  # initial lattice spacing
ipart = 0  # particle counter

for x in range(NperL):
    for y in range(NperL):
        pos[ipart, 0] = x * a + a / 2  # x position
        pos[ipart, 1] = y * a + a / 2  # y position
        ipart += 1
        if ipart >= Np:
            break
    if ipart >= Np:
        break

# Arrays to save configurations for plotting
Epot_save = []  # save potential energy

# Calculate the initial potential energy of the system
Epot = 0
for i in range(Np):
    for j in range(i + 1, Np):
        distij = pos[j, :] - pos[i, :]
        distij[0] -= L * np.rint(distij[0] / L)
        distij[1] -= L * np.rint(distij[1] / L)
        distij2 = np.dot(distij, distij)
        if distij2 < (2.5 * sigma)**2:
            Epot += 4 * epsilon * ((sigma12 / distij2**6) - (sigma6 / distij2**3))

# Monte Carlo loop
for istep in range(MCsteps):
    # Randomly select a particle
    j = np.random.randint(0, Np)

    # Propose a new position
    jposnew = pos[j, :] + np.random.uniform(-dmax, dmax, 2)
    jposnew %= L

    Eold = 0
    Enew = 0

    # Calculate old and new energies
    for k in range(Np):
        if k == j:
            continue

        distij = pos[j, :] - pos[k, :]
        distij -= L * np.rint(distij / L)
        distij2 = np.dot(distij, distij)
        if distij2 < (2.5 * sigma)**2:
            Eold += 4 * epsilon * ((sigma12 / distij2**6) - (sigma6 / distij2**3))

        distij = jposnew - pos[k, :]
        distij -= L * np.rint(distij / L)
        distij2 = np.dot(distij, distij)
        if distij2 < (2.5 * sigma)**2:
            Enew += 4 * epsilon * ((sigma12 / distij2**6) - (sigma6 / distij2**3))

    # Metropolis acceptance
    DeltaE = Enew - Eold
    if DeltaE <= 0 or np.random.rand() < np.exp(-DeltaE / kT):
        pos[j, :] = jposnew
        Epot += DeltaE

    # Save distances and energies for training
    distances.append(calculate_distances(pos, L))
    target_energies.append(Epot)

# Convert distances and energies to arrays
distances = np.array(distances)
target_energies = np.array(target_energies)

# Normalize data
dist_mean = distances.mean(axis=0)
dist_std = distances.std(axis=0)
norm_distances = (distances - dist_mean) / dist_std
norm_energies = (target_energies - target_energies.mean()) / target_energies.std()

# Split data into training and test sets
train_distances = norm_distances[:15000]
test_distances = norm_distances[15000:]
train_energies = norm_energies[:15000]
test_energies = norm_energies[15000:]

# Convert to PyTorch tensors
train_distances_tensor = torch.tensor(train_distances, dtype=torch.float32)
train_energies_tensor = torch.tensor(train_energies, dtype=torch.float32).view(-1, 1)
test_distances_tensor = torch.tensor(test_distances, dtype=torch.float32)
test_energies_tensor = torch.tensor(test_energies, dtype=torch.float32).view(-1, 1)

# Define neural network
class PhysicsInformedNet(nn.Module):
    def __init__(self):
        super(PhysicsInformedNet, self).__init__()
        self.fc1 = nn.Linear(int(Np * (Np - 1) / 2), 40)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(40, 40)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(40, 1)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

model = PhysicsInformedNet()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Train the model
losses = []
epochs = 5000
for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(train_distances_tensor)
    loss = criterion(predictions, train_energies_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# Plot training loss
plt.plot(range(epochs), losses)
plt.xlabel("Epochs")
plt.ylabel("Log(mean squared error)")
plt.title("Training Loss")
plt.show()

# Test the model
model.eval()
with torch.no_grad():
    test_predictions = model(test_distances_tensor).numpy()

# Scatter plot of predicted vs true energy
plt.scatter(test_energies, test_predictions, alpha=0.5, label="Predicted", color="green")
plt.plot([test_energies.min(), test_energies.max()], [test_energies.min(), test_energies.max()], 'r--', label="Actual")
plt.xlabel("Initial Energy")
plt.ylabel("Final Energy")
plt.title("Predicted vs Actual Energy")
plt.legend(loc="upper left")  
plt.show()
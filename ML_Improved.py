"""
Neural Network-Based Lennard-Jones Potential Prediction System.

This module performs a Monte Carlo simulation of particle interactions 
using the Lennard-Jones potential. It then trains a neural network 
to predict the system's potential energy based on inter-particle distances.

This improves upon earlier versions that used absolute particle positions, 
by introducing pairwise distances which are invariant to translation, 
rotation, and permutation.

Author: Yinan HU
Date: 06/05/2025
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------- CONSTANTS --------------------
BOX_SIZE = 8                  # Box size
NUM_PARTICLES = 20            # Number of particles
KT = 0.5                      # Temperature (thermal energy)
DMAX = 3                      # Max displacement in one MC step
EPSILON = 1                   # LJ interaction prefactor
SIGMA = 1                     # LJ particle size
SIGMA12 = SIGMA**12
SIGMA6 = SIGMA**6
MC_STEPS = 20000              # Number of Monte Carlo steps

# -------------------- FUNCTIONS --------------------

def calculate_distances(positions, box_size):
    """
    Calculate sorted inter-particle distances for a given configuration.

    Args:
        positions (array): Particle positions (N, 2).
        box_size (float): Simulation box size.

    Returns:
        array: Sorted array of pairwise distances.
    """
    distances = []
    num_particles = positions.shape[0]
    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            dist_vec = positions[j, :] - positions[i, :]
            dist_vec -= box_size * np.rint(dist_vec / box_size)
            distance = np.linalg.norm(dist_vec)
            distances.append(distance)
    return np.sort(distances)

# -------------------- MONTE CARLO SIMULATION --------------------

distances_list = []
target_energies = []

# Initialize particle positions
positions = np.zeros((NUM_PARTICLES, 2))
n_per_side = int(np.ceil(np.sqrt(NUM_PARTICLES)))
lattice_spacing = BOX_SIZE / n_per_side
particle_idx = 0

for x in range(n_per_side):
    for y in range(n_per_side):
        positions[particle_idx, 0] = x * lattice_spacing + lattice_spacing / 2
        positions[particle_idx, 1] = y * lattice_spacing + lattice_spacing / 2
        particle_idx += 1
        if particle_idx >= NUM_PARTICLES:
            break
    if particle_idx >= NUM_PARTICLES:
        break

# Calculate initial potential energy
epot = 0
for i in range(NUM_PARTICLES):
    for j in range(i + 1, NUM_PARTICLES):
        dist_ij = positions[j, :] - positions[i, :]
        dist_ij -= BOX_SIZE * np.rint(dist_ij / BOX_SIZE)
        dist2 = np.dot(dist_ij, dist_ij)
        if dist2 < (2.5 * SIGMA) ** 2:
            epot += 4 * EPSILON * ((SIGMA12 / dist2 ** 6) - (SIGMA6 / dist2 ** 3))

# Monte Carlo sampling loop
for step in range(MC_STEPS):
    # Random particle move
    j = np.random.randint(0, NUM_PARTICLES)
    new_pos = positions[j, :] + np.random.uniform(-DMAX, DMAX, 2)
    new_pos %= BOX_SIZE

    e_old = 0
    e_new = 0

    for k in range(NUM_PARTICLES):
        if k == j:
            continue

        dist_ij = positions[j, :] - positions[k, :]
        dist_ij -= BOX_SIZE * np.rint(dist_ij / BOX_SIZE)
        dist2_old = np.dot(dist_ij, dist_ij)

        if dist2_old < (2.5 * SIGMA) ** 2:
            e_old += 4 * EPSILON * ((SIGMA12 / dist2_old ** 6) - (SIGMA6 / dist2_old ** 3))

        dist_ij_new = new_pos - positions[k, :]
        dist_ij_new -= BOX_SIZE * np.rint(dist_ij_new / BOX_SIZE)
        dist2_new = np.dot(dist_ij_new, dist_ij_new)

        if dist2_new < (2.5 * SIGMA) ** 2:
            e_new += 4 * EPSILON * ((SIGMA12 / dist2_new ** 6) - (SIGMA6 / dist2_new ** 3))

    delta_e = e_new - e_old

    if delta_e <= 0 or np.random.rand() < np.exp(-delta_e / KT):
        positions[j, :] = new_pos
        epot += delta_e

    # Save data for training
    distances_list.append(calculate_distances(positions, BOX_SIZE))
    target_energies.append(epot)

# -------------------- DATA PREPARATION --------------------

distances_array = np.array(distances_list)
energies_array = np.array(target_energies)

# Normalize data
dist_mean = distances_array.mean(axis=0)
dist_std = distances_array.std(axis=0)
norm_distances = (distances_array - dist_mean) / dist_std
norm_energies = (energies_array - energies_array.mean()) / energies_array.std()

# Train-test split
train_distances = norm_distances[:15000]
test_distances = norm_distances[15000:]
train_energies = norm_energies[:15000]
test_energies = norm_energies[15000:]

# Convert to tensors
train_distances_tensor = torch.tensor(train_distances, dtype=torch.float32)
train_energies_tensor = torch.tensor(train_energies, dtype=torch.float32).view(-1, 1)
test_distances_tensor = torch.tensor(test_distances, dtype=torch.float32)
test_energies_tensor = torch.tensor(test_energies, dtype=torch.float32).view(-1, 1)

# -------------------- NEURAL NETWORK --------------------

class PhysicsInformedNet(nn.Module):
    """
    Neural network to predict Lennard-Jones potential based on 
    pairwise distances.
    """

    def __init__(self):
        super(PhysicsInformedNet, self).__init__()
        self.fc1 = nn.Linear(int(NUM_PARTICLES * (NUM_PARTICLES - 1) / 2), 40)
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

# -------------------- TRAINING --------------------

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
plt.ylabel("Mean Squared Error")
plt.title("Training Loss")
plt.show()

# -------------------- EVALUATION --------------------

model.eval()
with torch.no_grad():
    test_predictions = model(test_distances_tensor).numpy()

plt.scatter(test_energies, test_predictions, alpha=0.5, label="Predicted", color="green")
plt.plot(
    [test_energies.min(), test_energies.max()],
    [test_energies.min(), test_energies.max()],
    'r--',
    label="Actual"
)
plt.xlabel("True Energy (Normalized)")
plt.ylabel("Predicted Energy")
plt.title("Predicted vs Actual Energy")
plt.legend()
plt.show()

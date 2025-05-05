
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
pos_save = []  # save positions
Epot_save = []  # save potential energy

# Calculate the initial potential energy of the system
Epot = 0
for i in range(Np):
    for j in range(i + 1, Np):
        distij = pos[j, :] - pos[i, :]
        distij[0] = distij[0] - L * np.rint(distij[0] / L)  # look for nearest periodic image
        distij[1] = distij[1] - L * np.rint(distij[1] / L)  # look for nearest periodic image
        distij2 = distij[0] * distij[0] + distij[1] * distij[1]
        if distij2 < 2.5 * 2.5 * sigma * sigma:
            Epot += 4 * epsilon * ((sigma12 / distij2**6) - (sigma6 / distij2**3))

# Start the main Monte Carlo loop
positions = []  # Save all configurations for training
target_energies = []  # Save all potential energies for training

for istep in range(MCsteps):
    # Randomly select a particle
    j = np.random.randint(0, Np)

    # Propose a new position
    jposnew = pos[j, :] + np.random.uniform(-dmax, dmax, 2)  # new position
    jposnew %= L  # apply periodic boundary condition

    Eold = 0
    Enew = 0

    # Calculate the old and the new energies
    for k in range(Np):  # all neighbors of particle j
        if k == j:
            continue

        # Calculate the old distance between particles k and j
        distij = pos[j, :] - pos[k, :]
        distij[0] = distij[0] - L * np.rint(distij[0] / L)  # look for nearest periodic image
        distij[1] = distij[1] - L * np.rint(distij[1] / L)  # look for nearest periodic image
        distij2 = distij[0] * distij[0] + distij[1] * distij[1]  # distance squared

        if distij2 < 2.5 * 2.5 * sigma * sigma:
            Eold += 4 * epsilon * ((sigma12 / distij2**6) - (sigma6 / distij2**3))

        # Calculate the new distance between particles k and j
        distij = jposnew[:] - pos[k, :]
        distij[0] = distij[0] - L * np.rint(distij[0] / L)  # look for nearest periodic image
        distij[1] = distij[1] - L * np.rint(distij[1] / L)  # look for nearest periodic image
        distij2 = distij[0] * distij[0] + distij[1] * distij[1]  # distance squared

        if distij2 < 2.5 * 2.5 * sigma * sigma:
            Enew += 4 * epsilon * ((sigma12 / distij2**6) - (sigma6 / distij2**3))

    # Metropolis criterion
    DeltaE = Enew - Eold
    if DeltaE <= 0 or np.random.rand() < np.exp(-DeltaE / kT):
        # Accept the move
        pos[j, :] = jposnew
        Epot += DeltaE

    # Save data for training
    positions.append(pos.copy().flatten())
    target_energies.append(Epot)

    # Save potential energy and configuration for plotting
    Epot_save.append(Epot / Np)
    if istep % 100 == 0:
        pos_save.append(pos.copy())

# Plot the potential energy
plt.plot(Epot_save, label="Epot", linestyle="-")
plt.legend()
plt.xlabel("MC Steps")
plt.ylabel("Potential energy per particle")
plt.title("Monte Carlo Simulation")
plt.show()

# Save positions and energies for neural network training
positions = np.array(positions)
target_energies = np.array(target_energies)
np.save("positions.npy", positions)
np.save("target_energies.npy", target_energies)

# Load data for NN
positions = np.load("positions.npy")
target_energies = np.load("target_energies.npy")

# Normalize data
data_mean = positions.mean(axis=0)
data_std = positions.std(axis=0)
norm_positions = (positions - data_mean) / data_std
norm_energies = (target_energies - target_energies.mean()) / target_energies.std()

# Split into training and test sets
train_positions = norm_positions[:10000]
test_positions = norm_positions[10000:]
train_energies = norm_energies[:10000]
test_energies = norm_energies[10000:]

# Convert to PyTorch tensors
train_positions_tensor = torch.tensor(train_positions, dtype=torch.float32)
train_energies_tensor = torch.tensor(train_energies, dtype=torch.float32).view(-1, 1)
test_positions_tensor = torch.tensor(test_positions, dtype=torch.float32)
test_energies_tensor = torch.tensor(test_energies, dtype=torch.float32).view(-1, 1)

# Define neural network
class LennardJonesNet(nn.Module):
    def __init__(self):
        super(LennardJonesNet, self).__init__()
        self.fc1 = nn.Linear(2 * Np, 40)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(40, 40)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(40, 1)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

model = LennardJonesNet()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Train the model
losses = []
epochs = 10000
checkpoints = [100, 1000, 10000]
predictions_at_checkpoints = {}

for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(train_positions_tensor)
    loss = criterion(predictions, train_energies_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    # Save predictions at checkpoints
    if epoch + 1 in checkpoints:
        with torch.no_grad():
            predictions_at_checkpoints[epoch + 1] = model(test_positions_tensor).numpy()

# Plot training loss
plt.plot(range(epochs), np.log(losses))
plt.xlabel("Training time [epochs]")
plt.ylabel("Log (mean squared error)")
plt.title("Training Loss")
plt.show()

# Evaluate on test set
model.eval()
with torch.no_grad():
    test_predictions = model(test_positions_tensor).numpy()

# Scatter plot of predicted vs true energy
plt.scatter(test_energies, test_predictions, alpha=0.5, label="Predicted")
plt.plot([test_energies.min(), test_energies.max()], [test_energies.min(), test_energies.max()], 'r--', label="Actual")
plt.xlabel("Initial Energy")
plt.ylabel("Final Energy")
plt.title("Predicted vs LJ Energy")
plt.legend()
plt.show()

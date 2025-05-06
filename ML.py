
"""
Neural Network implementation for predicting Lennard-Jones potential energy.

This module performs a Monte Carlo simulation of particle interactions 
using the Lennard-Jones potential and trains a neural network to predict 
the system's potential energy based on particle positions.

Author: [Yinan Hu]
Date: [06/05/2025]
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

L = 8            # Box size
NP = 20          # Number of particles
KT = 0.5         # Temperature (thermal energy)
DMAX = 3         # Maximum displacement per MC step
EPSILON = 1      # LJ interaction prefactor
SIGMA = 1        # LJ particle size
SIGMA12 = SIGMA**12
SIGMA6 = SIGMA**6
MC_STEPS = 20000 # Number of Monte Carlo steps

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

def initialize_system(num_particles, box_size):
    """
    Initialize the particle system on a lattice.

    Args:
        num_particles (int): Number of particles in the system.
        box_size (float): Size of the simulation box.

    Returns:
        numpy.ndarray: Initial positions of the particles.
    """
    pos = np.zeros((num_particles, 2))
    nper_l = int(np.ceil(np.sqrt(num_particles)))
    a = box_size / nper_l
    ipart = 0

    for x in range(nper_l):
        for y in range(nper_l):
            pos[ipart, 0] = x * a + a / 2
            pos[ipart, 1] = y * a + a / 2
            ipart += 1
            if ipart >= num_particles:
                break
        if ipart >= num_particles:
            break

    return pos

def calculate_energy(positions, box_size):
    """
    Calculate the system's Lennard-Jones potential energy.

    Args:
        positions (numpy.ndarray): Particle positions.
        box_size (float): Size of the simulation box.

    Returns:
        float: Potential energy of the system.
    """
    e_pot = 0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            distij = positions[j, :] - positions[i, :]
            distij[0] -= box_size * np.rint(distij[0] / box_size)
            distij[1] -= box_size * np.rint(distij[1] / box_size)
            distij2 = distij[0]**2 + distij[1]**2
            if distij2 < (2.5 * SIGMA)**2:
                e_pot += 4 * EPSILON * (
                    (SIGMA12 / distij2**6) - (SIGMA6 / distij2**3)
                )
    return e_pot

class LennardJonesNet(nn.Module):
    """Neural network for predicting Lennard-Jones potential energy."""

    def __init__(self, input_size):
        """
        Initialize the neural network.

        Args:
            input_size (int): Size of the input layer (2 * number of particles).
        """
        super(LennardJonesNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 40)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(40, 40)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(40, 1)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor containing particle positions.

        Returns:
            torch.Tensor: Predicted potential energy.
        """
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

# -----------------------------------------------------------------------------
# MAIN WORKFLOW
# -----------------------------------------------------------------------------

def main():
    """Main function to run the simulation and training process."""
    # Initialize the system
    pos = initialize_system(NP, L)
    pos_save = []
    e_pot_save = []
    positions = []
    target_energies = []

    # Calculate initial energy
    e_pot = calculate_energy(pos, L)

    # Main Monte Carlo loop
    for istep in range(MC_STEPS):
        j = np.random.randint(0, NP)
        jposnew = pos[j, :] + np.random.uniform(-DMAX, DMAX, 2)
        jposnew %= L

        # Calculate old and new energies
        e_old = calculate_energy(pos, L)
        pos_temp = pos.copy()
        pos_temp[j, :] = jposnew
        e_new = calculate_energy(pos_temp, L)

        # Metropolis acceptance criterion
        delta_e = e_new - e_old
        if delta_e <= 0 or np.random.rand() < np.exp(-delta_e / KT):
            pos[j, :] = jposnew
            e_pot = e_new

        # Save data for training
        positions.append(pos.copy().flatten())
        target_energies.append(e_pot)
        e_pot_save.append(e_pot / NP)
        if istep % 100 == 0:
            pos_save.append(pos.copy())

    # -----------------------------------------------------------------------------
    # DATA PROCESSING AND MODEL TRAINING
    # -----------------------------------------------------------------------------

    positions = np.array(positions)
    target_energies = np.array(target_energies)

    np.save("positions.npy", positions)
    np.save("target_energies.npy", target_energies)

    # Load and normalize data
    positions = np.load("positions.npy")
    target_energies = np.load("target_energies.npy")

    data_mean = positions.mean(axis=0)
    data_std = positions.std(axis=0)
    norm_positions = (positions - data_mean) / data_std
    norm_energies = (
        target_energies - target_energies.mean()
    ) / target_energies.std()

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

    # Create and train the neural network
    model = LennardJonesNet(2 * NP)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Training loop
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

        if (epoch + 1) in checkpoints:
            with torch.no_grad():
                predictions_at_checkpoints[epoch + 1] = (
                    model(test_positions_tensor).numpy()
                )

    # -----------------------------------------------------------------------------
    # VISUALIZATION
    # -----------------------------------------------------------------------------

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), np.log(losses))
    plt.xlabel("Training Time [epochs]")
    plt.ylabel("Log(MSE Loss)")
    plt.title("Training Loss")
    plt.show()

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        test_predictions = model(test_positions_tensor).numpy()

    # Plot predicted vs true energies
    plt.figure(figsize=(10, 6))
    plt.scatter(test_energies, test_predictions, alpha=0.5, label="Predicted")
    plt.plot(
        [test_energies.min(), test_energies.max()],
        [test_energies.min(), test_energies.max()],
        'r--', label="Actual"
    )
    plt.xlabel("True Energy")
    plt.ylabel("Predicted Energy")
    plt.title("Predicted vs True LJ Energy")
    plt.legend()
    plt.show()

# -----------------------------------------------------------------------------
# RUN MAIN FUNCTION
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    main()

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
"""
Initialize system parameters and constants
"""

L = 10             # Box size
nA = 20            # Number of type A particles
nB = 20            # Number of type B particles
Np = nA + nB       # Total number of particles

KB = 1             # Boltzmann constant
TEMP = 1           # Temperature
KT = KB * TEMP     # Thermal energy

SIGMA = 1          # Particle size
SIGMA12 = SIGMA**12
SIGMA6 = SIGMA**6

# Interaction strengths
EPSILON_AA = 0.1 * KT
EPSILON_BB = 0.1 * KT
EPSILON_AB = 0.1 * KT

DMAX = 0.8         # Maximum displacement for Monte Carlo moves
MCSTEPS = 10000    # Total number of Monte Carlo steps

BINSIZE = 0.1
NBINS = int(L / (2 * BINSIZE))  # Number of bins for radial distribution function (RDF)

"""
Initialize the particle positions on a 2D lattice grid within the box. 
The particles are placed in a square lattice with periodic boundary conditions.
"""
pos = np.zeros((Np, 2))
N_PER_L = int(np.ceil(np.sqrt(Np)))  # Number of particles per row
a = L / N_PER_L                      # Lattice spacing

ipart = 0
for x in range(N_PER_L):
    for y in range(N_PER_L):
        pos[ipart, 0] = x * a + a / 2
        pos[ipart, 1] = y * a + a / 2
        ipart += 1
        if ipart >= Np:
            break
    if ipart >= Np:
        break

"""
Assign particle types: 0 for A, and 1 for B. 
# The first nA particles are of type A, and the remaining nB particles are of type B.
"""
types = np.zeros(Np, dtype=int)
types[nA:] = 1  # The last nB particles are type B

# -----------------------------
# Lennard-Jones potential function
# -----------------------------
def lj_potential(r2: float, epsilon: float) -> float:
    """
    Computes the Lennard-Jones potential energy between two particles based on their squared distance.

    Args:
        r2 (float): The squared distance between two particles.
        epsilon (float): The interaction strength between the particles.

    Returns:
        float: The Lennard-Jones potential energy between the two particles.
    """
    if r2 < (2.5 * SIGMA)**2:
        r2_inv = 1.0 / r2
        r6_inv = r2_inv**3
        r12_inv = r6_inv**2
        return 4 * epsilon * (SIGMA12 * r12_inv - SIGMA6 * r6_inv)
    return 0

# -----------------------------
# Compute initial total potential energy
# -----------------------------
"""
Calculates the initial total potential energy of the system by summing up pairwise interactions 
between all particles using the Lennard-Jones potential function.
"""
Epot = 0
for i in range(Np):
    for j in range(i + 1, Np):
        rij = pos[j] - pos[i]
        rij -= L * np.rint(rij / L)  # Periodic boundary conditions
        r2 = np.dot(rij, rij)

        if types[i] == 0 and types[j] == 0:
            epsilon = EPSILON_AA
        elif types[i] == 1 and types[j] == 1:
            epsilon = EPSILON_BB
        else:
            epsilon = EPSILON_AB

        Epot += lj_potential(r2, epsilon)

# -----------------------------
# Monte Carlo loop
# -----------------------------
"""
Runs the Monte Carlo simulation for MCSTEPS iterations. 
In each iteration, a random particle is chosen, a new position is proposed, 
and the energy change is calculated. The move is accepted or rejected based on the Metropolis criterion.
"""
pos_save = []      # Save positions for animation
Epot_save = []     # Save energy per particle

for istep in range(MCSTEPS):

    # Select a random particle
    j = np.random.randint(0, Np)
    old_pos = pos[j].copy()

    # Propose a new position with random displacement
    new_pos = old_pos + np.random.uniform(-DMAX, DMAX, size=2)
    new_pos -= L * np.floor(new_pos / L)  # Periodic boundary conditions

    # Compute energy change ΔE
    DeltaE = 0
    for k in range(Np):
        if k == j:
            continue

        rij_old = old_pos - pos[k]
        rij_old -= L * np.rint(rij_old / L)
        r2_old = np.dot(rij_old, rij_old)

        rij_new = new_pos - pos[k]
        rij_new -= L * np.rint(rij_new / L)
        r2_new = np.dot(rij_new, rij_new)

        if types[j] == 0 and types[k] == 0:
            epsilon = EPSILON_AA
        elif types[j] == 1 and types[k] == 1:
            epsilon = EPSILON_BB
        else:
            epsilon = EPSILON_AB

        DeltaE += lj_potential(r2_new, epsilon) - lj_potential(r2_old, epsilon)

    # Metropolis acceptance criterion
    if DeltaE <= 0 or np.random.rand() < np.exp(-DeltaE / KT):
        pos[j] = new_pos
        Epot += DeltaE

    # Save energy and position snapshots
    Epot_save.append(Epot / Np)
    if istep % 80 == 0:
        pos_save.append(pos.copy())

# -----------------------------
# Radial distribution function (RDF)
# -----------------------------
def calculate_rdf(pos: np.ndarray, binsize: float, L: float, nbins: int) -> tuple:
    """
    Computes the radial distribution function (g(r)) to analyze the local ordering of particles.

    Args:
        pos (np.ndarray): The positions of particles in the system.
        binsize (float): The size of each radial bin.
        L (float): The size of the simulation box.
        nbins (int): The number of bins for RDF computation.

    Returns:
        tuple: A tuple containing two arrays: the radial distances and the corresponding RDF values.
    """
    g_r = np.zeros(nbins)
    dr = binsize

    for i in range(Np):
        for j in range(i + 1, Np):
            rij = pos[j] - pos[i]
            rij -= L * np.rint(rij / L)
            r = np.linalg.norm(rij)
            if r < L / 2:
                bin_index = int(r / dr)
                g_r[bin_index] += 2  # Each pair counted twice

    r = np.linspace(binsize / 2, L / 2, nbins)
    density = Np / L**2
    for i in range(nbins):
        shell_volume = np.pi * ((r[i] + dr / 2)**2 - (r[i] - dr / 2)**2)
        g_r[i] /= (density * shell_volume * Np)
    return r, g_r

# Compute RDF using the final saved position
final_pos = pos_save[-1]
r, g_r = calculate_rdf(final_pos, BINSIZE, L, NBINS)

"""
Plots the total potential energy per particle over the Monte Carlo steps to observe system's convergence.
"""
plt.plot(Epot_save)
plt.xlabel('MC Steps')
plt.ylabel('Potential Energy per Particle')
plt.title('Monte Carlo Simulation: Potential Energy')
plt.show()

"""
Plots the radial distribution function g(r) to analyze the local ordering and phase transitions.
"""
plt.plot(r, g_r, label="g(r)")
plt.xlabel("Radial Distance [σ]")
plt.ylabel("Radial Distribution Function g(r)")
plt.title("Radial Distribution Function")
plt.legend()
plt.show()

"""
Generates an animation of the particle configurations over time with different colors for particle 
types A and B.
"""
from matplotlib import animation, rc
import matplotlib

matplotlib.rcParams['animation.embed_limit'] = 2**32

fig, ax = plt.subplots()
ax.set(xlim=[0, L], ylim=[0, L], xlabel='x [σ]', ylabel='y [σ]')
ax.set_aspect('equal')

# Assign colors: blue for A particles, red for B particles
colors = np.array(['blue' if t == 0 else 'red' for t in types])

# Initialize the scatter plot using the first saved frame
scat = ax.scatter(pos_save[0][:, 0], pos_save[0][:, 1], c=colors, s=100)

def update(frame):
    """Update function for animation."""
    xf = pos_save[frame][:, 0] - L * np.floor(pos_save[frame][:, 0] / L)
    yf = pos_save[frame][:, 1] - L * np.floor(pos_save[frame][:, 1] / L)
    data = np.stack([xf, yf]).T
    scat.set_offsets(data)
    return (scat,)

# Create the animation
ani = animation.FuncAnimation(
    fig=fig,
    func=update,
    frames=len(pos_save),
    interval=30
)

# Display the animation in Jupyter Notebook or Colab
from IPython.display import HTML
HTML(ani.to_jshtml())

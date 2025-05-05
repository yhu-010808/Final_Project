# Final_Project
Monte Carlo Simulation and Physics-Informed Machine Learning Potentials for Binary Nanoparticle Systems

# 1. Introduction
This project investigates the self-assembly behavior of binary nanoparticle systems, which are composed of two types of nanoparticles (e.g., hydrophobic/hydrophilic or magnetic/non-magnetic particles). These systems exhibit complex behaviors such as clustering, partial ordering, and crystallization. By modifying the inter-particle attractions, such as electrostatic forces, van der Waals interactions, or ligand binding, we aim to control the self-assembly process and obtain ordered structures at different scales.

  ## 1.1 Key Motivations
The primary motivation behind this project is to understand how varying inter-particle interactions affect the energy and structural ordering in binary nanoparticle systems. This research has significant potential applications, particularly in fields such as drug delivery, materials science, and optoelectronics. By manipulating these interactions, we can design programmable nanomaterials and responsive assemblies with tunable properties.

  ## 1.2 Research Significance
This project aims to enhance our understanding of structural evolution and phase transitions in nanoparticle systems, focusing on how nanoparticles shift between gas-like, liquid-like, and crystalline phases. This research contributes to nanomaterial formation and self-assembly processes. Practical applications include designing nanomaterials for targeted drug delivery, where nanoparticles respond to stimuli for controlled release, as well as innovations in responsive nanoparticle assemblies for optoelectronics and other technologies.

  ## 1.3 Innovation
One of the innovative aspects of this project is the combination of Monte Carlo simulations and machine learning. By using machine learning to predict the system's energy, we aim to replace traditional Lennard-Jones potential calculations, speeding up the energy evaluation process. This approach can significantly improve the efficiency and scalability of simulations, especially in systems with many particles or complex interactions.

# 2. Methodology

  ## 2.1 Monte Carlo Simulation
  
  ### Research Method
This project uses a **Monte Carlo simulation** to study the self-assembly behavior of binary nanoparticle systems. The key goal is to understand how varying inter-particle attractions influence the system’s energy and structural ordering, transitioning between gas-like, liquid-like, and crystalline phases. 

The simulation employs the **Metropolis algorithm** for particle configuration updates, where random particle displacements are proposed, and the energy change (\(\Delta E\)) is evaluated. If the energy decreases, the move is accepted; if the energy increases, it is accepted with a probability given by the **Boltzmann factor** \(e^{-\Delta E / kT}\).

  ### Experimental Design
- **System Setup**: A 2D system of **40 particles** (20 type A and 20 type B) is initialized on a square lattice with periodic boundary conditions. The simulation box size is set to **L = 10**.
- **Interaction Model**: The particles interact through a **Lennard-Jones potential**, with different parameters for A-A, B-B, and A-B interactions.
- **Monte Carlo Steps**: A total of **10,000 MC steps** are performed, with random particle movements and energy calculations at each step.
- **Observation and Analysis**: Throughout the simulation, we track the system's **total energy** and calculate the **radial distribution function (RDF)** to analyze the phase behavior. We also visualize the particle configurations and the evolution of energy.

This approach allows us to study the system’s behavior under various interaction strengths and observe the transitions between different phases, providing insights into nanoparticle self-assembly.

  ## 2.2 Machine Learning


# 3. Code Logic

 ## 3.1 Monte Carlo Simulation

 ### Initialization
- **Particle positions** are initialized on a 2D lattice within a square simulation box.
- The system consists of **40 particles**, divided into **two types** (A and B). The particle types are assigned, and initial positions are saved.

### Lennard-Jones Potential Formula
- The **Lennard-Jones potential** function is used to calculate the interaction energy between pairs of particles.
- Interaction parameters (epsilon) are set for **A-A, B-B**, and **A-B** interactions.

### Monte Carlo Loop
1. **Random Particle Selection**: A particle is randomly selected, and a new position is proposed using a random displacement.
2. **Energy Calculation**: The **energy change** (\(\Delta E\)) between the new and old particle configurations is calculated using the Lennard-Jones potential.
3. **Metropolis Acceptance Criterion**: The proposed move is accepted if the energy decreases or with a probability of \( e^{-\Delta E / kT} \) if the energy increases.
4. **Data Saving**: Energy and particle positions are saved periodically for later analysis.

### Analysis and Visualization
- **Potential Energy**: The total energy per particle is tracked during the simulation to observe system convergence.
- **Radial Distribution Function (RDF)**: The RDF is computed to analyze local ordering and identify phase transitions (gas-like, liquid-like, crystalline).
- **Particle Configurations**: Snapshots of particle positions are saved and visualized as animations to show the evolution of the system (these gif plots can be obtained through Colab).


 ## 3.2 Machine Learning

# 4. Future Direction

# 5. Authors
Maxine Wang: https://github.com/Maxine-wang-7

Yinan Hu: https://github.com/yhu-010808

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

## Introduction to ML

To overcome the limitations of the traditional Monte Carlo method—specifically, the computational expense of recalculating the Lennard-Jones (LJ) energy for every particle move—we introduced **Machine Learning (ML)** to accelerate energy predictions.

**Machine Learning (ML)** refers to algorithms that can learn patterns from data. In this project, the ML model learns to predict the potential energy of particle configurations without explicitly recalculating the LJ energy each time. Once trained, the model can make rapid energy predictions, significantly speeding up the simulation.



##ML Model Using Particle Positions as Input

### Objective  
Train a neural network that can predict the potential energy of a particle configuration based on particle positions.

### Workflow

**Monte Carlo Sampling**  
- Perform a Monte Carlo simulation to generate various particle configurations by randomly perturbing particle positions.
- Each new configuration is accepted or rejected based on the Metropolis criterion.

**Saving Training Data**  
- For each accepted configuration:  
  - **Input** → Flattened particle positions (x, y for all particles).  
  - **Output (label)** → Corresponding LJ energy.

**Data Normalization and Splitting**  
- Normalize position and energy data to improve training stability.
- Split the dataset into **training** and **testing** sets.

**Neural Network Construction**  
- Define a **feedforward neural network** using PyTorch:  
  - **Input layer**: Particle positions (**2 × Np dimensions**).  
  - **Hidden layers**: Two layers with 40 neurons each, using ReLU activation.  
  - **Output layer**: Predicted potential energy.

**Training the Neural Network**  
- Loss function: **Mean Squared Error (MSE)**.  
- Optimizer: **Stochastic Gradient Descent (SGD)** with momentum.
- Train the network to minimize the difference between predicted and actual LJ energies.

**Model Evaluation**  
- Compare predicted energies against actual LJ energies using scatter plots.
- **Observations**:
  - Slow convergence.
  - Higher prediction error.
  - Sensitivity to input representation (absolute positions are sensitive to translation and rotation).



## Improved ML Model: Using Inter-Particle Distances 

### Motivation  
The limitations of using absolute positions motivated us to improve the model by using **inter-particle distances** as the input feature.

**Advantages of Distances as Input**:
- **Translation invariance**: Moving the entire system does not change the distances.
- **Rotation invariance**: Rotating the system does not affect distances.
- **Permutation invariance**: Swapping particle labels does not change the input.

### Key Modifications

1️⃣ **New Input Representation**  
- Compute and use **all pairwise inter-particle distances**.  
- Distances are **sorted** to maintain consistent input order.  
- Input dimension: \( Np \times (Np - 1) / 2 \).

2️⃣ **Updated Training Data**  
- **Input** → Sorted inter-particle distances.  
- **Output (label)** → Corresponding LJ energy.

3️⃣ **Normalization and Splitting**  
- Normalize distances and energies.
- Use larger training and testing datasets.

4️⃣ **Neural Network Construction**  
- Same architecture as Task 2(a), except:  
  - **Input layer** takes distance vectors instead of position vectors.

5️⃣ **Training and Evaluation**  
- The model converges **faster** and achieves **lower prediction errors** compared to the position-based model.
- Scatter plots confirm better alignment between predicted and actual energies.



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

Both tasks follow the same **general workflow**:

1. **Generate Configurations via Monte Carlo (MC)**
    - Initialize particles on a grid.
    - At each MC step:
        - Randomly select a particle and propose a new position.
        - Calculate the energy change (ΔE) using the Lennard-Jones (LJ) potential.
        - Accept or reject the new configuration based on the Metropolis criterion.

2. **Save Data for ML Training**
    - For each accepted configuration:
        - **ML**: Save flattened particle positions as input.
        - **ML_Improved**: Calculate all pairwise distances and save as input.
        - Save the LJ energy as the output (label).

 

# 4. Future Direction
**Physics-Informed Neural Networks (PINNs)**  
- Integrate physical laws (e.g., conservation, symmetries) directly into the network architecture or loss function.

**Active Learning**  
- Dynamically collect new training data during simulations to improve model generalization.

**Transfer Learning**  
- Transfer trained models to different systems (e.g., larger particle numbers, different interaction types).

**Graph Neural Networks (GNNs)**  
- Use graph-based architectures to capture complex spatial relationships between particles.

**ML-Accelerated MC Move Proposals**  
- Use reinforcement learning or generative models to suggest more efficient particle moves during MC sampling.

**Coupling with Quantum Calculations**  
- Train ML models on **DFT-calculated energies** to capture quantum-level accuracy.

**Applications to Biomolecular Systems**  
- Extend the ML-MC framework to study **protein folding** or **complex biomolecular assemblies**.

## Final Summary

*"Our project successfully demonstrated that combining Monte Carlo sampling with machine learning significantly accelerates energy prediction. By improving our input features and exploring advanced ML techniques, this approach can be extended to complex materials, quantum-level simulations, and biological systems in the future."*



# 5. Authors
Maxine Wang: https://github.com/Maxine-wang-7

Yinan Hu: https://github.com/yhu-010808

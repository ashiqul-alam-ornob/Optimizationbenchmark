import numpy as np


def pso(objective_function: callable, num_particles: int, num_dimensions: int, max_iterations: int) -> tuple:
    """
    Particle Swarm Optimization (PSO) algorithm.

    Parameters:
    - objective_function (callable): The objective function to be optimized.
    - num_particles (int): Number of particles in the swarm.
    - num_dimensions (int): Number of dimensions in the search space.
    - max_iterations (int): Maximum number of iterations.

    Returns:
    - tuple: Best solution found (numpy.ndarray) and its corresponding objective value (float).
    """
    # Initialize particles randomly in the search space
    particles_position = np.random.rand(num_particles, num_dimensions) * 5

    # Initialize particle velocities
    particles_velocity = np.random.rand(num_particles, num_dimensions) * 0.1

    # Initialize personal best positions and values
    personal_best_positions = particles_position.copy()
    personal_best_values = np.array([objective_function(point) for point in personal_best_positions])

    # Initialize global best position and value
    global_best_index = np.argmin(personal_best_values)
    global_best_position = personal_best_positions[global_best_index].copy()
    global_best_value = personal_best_values[global_best_index]

    # PSO main loop
    for _ in range(max_iterations):
        # Inertia weight to balance exploration and exploitation
        inertia_weight = 0.5 
        # Cognitive coefficient
        cognitive_coefficient = 1.5
         # Social coefficient
        social_coefficient = 2.0

        # Update velocities
        particles_velocity = (inertia_weight * particles_velocity +
                              cognitive_coefficient * np.random.rand() * (personal_best_positions - particles_position) +
                              social_coefficient * np.random.rand() * (global_best_position - particles_position))

        # Update positions
        particles_position += particles_velocity

        # Evaluate objective function at new positions
        current_values = np.array([objective_function(point) for point in particles_position])

        # Update personal best positions and values
        update_personal_best = current_values < personal_best_values
        personal_best_positions[update_personal_best] = particles_position[update_personal_best]
        personal_best_values[update_personal_best] = current_values[update_personal_best]

        # Update global best position and value
        global_best_index = np.argmin(personal_best_values)
        global_best_position = personal_best_positions[global_best_index].copy()
        global_best_value = personal_best_values[global_best_index]

    return global_best_position, global_best_value

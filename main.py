from algorithms import pso, pso_dynamic_weight
from benchmark_functions import (
    ackley_function,
    branin_rcos_function,
    goldstein_price_function,
    griewank_function,
    levy_function,
    michalewicz_function,
    rastrigin_function,
    rosenbrock_function,
    six_hump_camelback_function,
    sphere_function
)
from util.logger import setup_logging
from util.writer import write_to_csv

# Set up the logging configuration
logger = setup_logging(level="INFO")


if __name__ == "__main__":
    # Declare PSO constants
    num_particles = 100
    num_dimensions = 3
    max_iterations = 200

    # Results to write in a CSV file
    csv_results = []

    # Benchmark functions to test
    benchmark_functions = [ackley_function, branin_rcos_function, goldstein_price_function, 
                           griewank_function, levy_function, michalewicz_function, 
                           rastrigin_function, rosenbrock_function, six_hump_camelback_function, 
                           sphere_function]

    # Run tests
    for idx, objective_function in enumerate(benchmark_functions, start=1):
        logger.info(f"Testing PSO on Benchmark Function {idx} - {objective_function.__name__}")

        # Test base PSO
        best_position_static, best_value_static = pso(objective_function, num_particles, num_dimensions, max_iterations)
        logger.debug(f"Base PSO:")
        logger.debug(f"Best Position: {best_position_static}")
        logger.debug(f"Best Value: {best_value_static}")
        csv_results.append([objective_function.__name__, "Base PSO", *best_position_static, best_value_static])

        # Test PSO with dynamic weight
        best_position_dyn, best_value_dyn = pso_dynamic_weight(objective_function, num_particles, num_dimensions, max_iterations)
        logger.debug(f"Optimization Result with Dynamic Weight:")
        logger.debug(f"Best Position: {best_position_dyn}")
        logger.debug(f"Best Value: {best_value_dyn}")
        csv_results.append([objective_function.__name__, "PSO with Dynamic Weight", *best_position_dyn, best_value_dyn])

    # Write results to CSV file
    csv_file_path = "results/optimization_results.csv"
    header = ["Benchmark", "Algorithm", "BestX", "BestY", "BestZ", "BestValue"]
    write_to_csv(csv_file_path, header, csv_results)
    logger.info(f"Results written to {csv_file_path}")

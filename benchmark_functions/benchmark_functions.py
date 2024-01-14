import numpy as np


def ackley_function(x: np.ndarray) -> float:
    """
    Ackley Function

    Parameters:
    - x (numpy.ndarray): Particle position in the search space.

    Returns:
    - float: Objective function value at the given position.
    """
    a = 20
    b = 0.2
    c = 2 * np.pi
    term1 = -a * np.exp(-b * np.sqrt(np.mean(x**2)))
    term2 = -np.exp(np.mean(np.cos(c * x)))
    return term1 + term2 + a + np.exp(1)


def branin_rcos_function(x: np.ndarray) -> float:
    """
    Branin RCOS (Rotated Complex Overlapping Squares) Function

    Parameters:
    - x (numpy.ndarray): Particle position in the search space.

    Returns:
    - float: Objective function value at the given position.
    """
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)

    x1, x2 = x[0], x[1]
    term1 = a * (x2 - b * x1**2 + c * x1 - r)**2
    term2 = s * (1 - t) * np.cos(x1)
    return term1 + term2 + s


def goldstein_price_function(x: np.ndarray) -> float:
    """
    Goldstein-Price Function

    Parameters:
    - x (numpy.ndarray): Particle position in the search space.

    Returns:
    - float: Objective function value at the given position.
    """
    x1, x2 = x[0], x[1]
    term1 = 1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3 * x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)
    term2 = 30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12 * x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2)
    return term1 * term2


def griewank_function(x: np.ndarray) -> float:
    """
    Griewank Function

    Parameters:
    - x (numpy.ndarray): Particle position in the search space.

    Returns:
    - float: Objective function value at the given position.
    """
    term1 = np.sum(x**2) / 4000
    term2 = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return 1 + term1 - term2


def levy_function(x: np.ndarray) -> float:
    """
    Levy Function

    Parameters:
    - x (numpy.ndarray): Particle position in the search space.

    Returns:
    - float: Objective function value at the given position.
    """
    w = 1 + (x - 1) / 4
    term1 = (np.sin(np.pi * w[0]))**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * (np.sin(np.pi * w[:-1] + 1))**2))
    term3 = (w[-1] - 1)**2 * (1 + (np.sin(2 * np.pi * w[-1]))**2)
    return term1 + term2 + term3


def michalewicz_function(x: np.ndarray, m: int = 10) -> float:
    """
    Michalewicz Function

    Parameters:
    - x (numpy.ndarray): Particle position in the search space.
    - m (int): Parameter controlling the shape of the function.

    Returns:
    - float: Objective function value at the given position.
    """
    return -np.sum(np.sin(x) * np.sin((np.arange(1, len(x) + 1) * x**2) / np.pi)**(2 * m))


def rastrigin_function(x: np.ndarray) -> float:
    """
    Rastrigin Function

    Parameters:
    - x (numpy.ndarray): Particle position in the search space.

    Returns:
    - float: Objective function value at the given position.
    """
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def rosenbrock_function(x: np.ndarray) -> float:
    """
    Rosenbrock Function

    Parameters:
    - x (numpy.ndarray): Particle position in the search space.

    Returns:
    - float: Objective function value at the given position.
    """
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def six_hump_camelback_function(x: np.ndarray) -> float:
    """
    Six-hump Camelback Function

    Parameters:
    - x (numpy.ndarray): Particle position in the search space.

    Returns:
    - float: Objective function value at the given position.
    """
    x1, x2 = x[0], x[1]
    term1 = (4 - 2.1 * x1**2 + (x1**4)/3) * x1**2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2**2) * x2**2
    return term1 + term2 + term3


def sphere_function(x: np.ndarray) -> float:
    """
    Sphere Function

    Parameters:
    - x (numpy.ndarray): Particle position in the search space.

    Returns:
    - float: Objective function value at the given position.
    """
    return np.sum(x**2)

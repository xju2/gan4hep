import itertools

import numpy as np

"""All four vector representaion is assumed to be [E, px, py, pz]"""


def calculate_mass(lorentz_vector):
    sum_p2 = sum([lorentz_vector[idx]**2 for idx in range(1,4)])
    return np.sqrt(lorentz_vector[0]**2 - sum_p2)


def create_boost_fn(cluster_4vec: np.ndarray):
    mass = calculate_mass(cluster_4vec)
    E0, p0 = cluster_4vec[0], cluster_4vec[1:]
    gamma = E0 / mass

    velocity = p0 / gamma / mass
    v_mag = np.sqrt(sum([velocity[idx]**2 for idx in range(3)]))
    n = velocity / v_mag

    def boost_fn(lab_4vec: np.ndarray):
        """4vector [E, px, py, pz] in lab frame"""
        E = lab_4vec[0]
        p = lab_4vec[1:]
        n_dot_p = np.sum((n * p))
        E_prime = gamma * (E - v_mag * n_dot_p)
        P_prime = p + (gamma - 1) * n_dot_p * n - gamma * E * v_mag * n
        return np.array([E_prime]+ P_prime.tolist())
    
    def inv_boost_fn(boost_4vec: np.ndarray):
        """4vecot [E, px, py, pz] in boost frame (aka cluster frame)"""
        E_prime = boost_4vec[0]
        P_prime = boost_4vec[1:]
        n_dot_p = np.sum((n * P_prime))
        E = gamma * (E_prime + v_mag * n_dot_p)
        p = P_prime + (gamma - 1) * n_dot_p * n + gamma * E_prime * v_mag * n
        return np.array([E]+ p.tolist())

    return boost_fn, inv_boost_fn


def boost(a_row: np.ndarray):
    """boost all particles to the rest frame of the first particle in the list"""

    assert a_row.shape[0] % 4 == 0, "a_row should be a 4-vector"
    boost_fn, _ = create_boost_fn(a_row[:4])
    n_particles = len(a_row) // 4
    results = [boost_fn(a_row[4*x: 4*(x+1)]) for x in range(n_particles)]
    return list(itertools.chain(*[x.tolist() for x in results]))
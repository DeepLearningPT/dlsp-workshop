"""Utilities to import simulation data."""

import os

import numpy as np

def read_simulation(sim_dir: str, sim_id: int) -> np.ndarray:
    """Reads simulation info to numpy array.
    
    Args:
      sim_dir: directory where the simulations are stored.
      sim_id: ID of the simulation to read.

    Returns:
      Numpy array with shape (time_steps, 6, num_particles) with the data 
      form the simulation.
    """

    list_simulations = os.listdir(sim_dir)

    sim_folder_name = f"sim_{sim_id:04}"

    if sim_folder_name not in list_simulations:
        raise ValueError(f"Simulation id {sim_id} is not available in {sim_dir}")
    
    sim_data_path = os.path.join(sim_dir, sim_folder_name, "particle_data.npy")
    
    sim_data = np.load(sim_data_path)

    return sim_data

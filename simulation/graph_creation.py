"""Function to create graphs"""
import os

import torch
from torch_geometric.data import Data


def make_graph_edges_from_particle_positions(particle_positions,
                                             connectivity_radius):
    """Makes undirected edges between particles with distance equal
    or less than the connectivity radiius.

    Args:

      particle_posisitons: Numpy array with shape (3, N) where 3 is
      the number of dimensions and N is the number of particles.

      connectivity_radius: Distance between particles that should be
      connected.

    Returns:

    edges_list: Numpy array with shape (2, M) where M is the number of
    nodes in the graph.

    edge_features: Numpy array with shape (M, 4) where 4 corresponds
    to the features of each edge.

    """
    edges_list = []
    edge_features = []

    for i in range(particle_positions.shape[1]):
        for j in range(i + 1, particle_positions.shape[1]):
            dist = np.linalg.norm(particle_positions[:, i] -
                                  particle_positions[:, j])
            if dist <= connectivity_radius:
                edges_list.append([i, j])
                edge_features.append(
                    np.concatenate(
                        (particle_positions[:, i] - particle_positions[:, j],
                         np.array([dist]))))

                edges_list.append([j, i])
                edge_features.append(
                    np.concatenate(
                        (particle_positions[:, j] - particle_positions[:, i],
                         np.array([dist]))))

    return np.array(edges_list).T, np.array(edge_features)


def make_node_features(particle_velocities, particle_positions,
                       cube_wall_positions):
    """Args:

    particle_velocities: Array with velocities of shape [t, 3] where t
    is the number of time steps to be considered.

    particle_positions: Array with positions of particles of shape [3]
    with the x, y, z corrdinates of particles.

    cube_all_positions: A list of size 6 containing the position of
    the 6 walls of the cube. [x_front, x_back, y_left, y_right, z_top,
    z_bottom]

    Returns:

    features: An array with the particle velocities for the t time
    steps and the distances to each of the cube walls.

    """
    t, _ = particle_velocities.shape
    velocities = particle_velocities.reshape(t * 3)
    distance_to_walls = np.array([
        abs(cube_wall_positions[0] - particle_positions[0]),
        abs(cube_wall_positions[1] - particle_positions[0]),
        abs(cube_wall_positions[2] - particle_positions[1]),
        abs(cube_wall_positions[3] - particle_positions[1]),
        abs(cube_wall_positions[4] - particle_positions[2]),
        abs(cube_wall_positions[5] - particle_positions[2]),
    ])
    return np.concatenate((velocities, distance_to_walls))


def make_graph_node_features(particle_velocities, particle_positions,
                             cube_wall_positions):
    """Args:

    particle_velocities: List of [(t, 3)]. That is, a list with the
    velocities at the last t time steps.

    particle_positions: The positions of the nodes. That is, a list
    [(x, y, z)]

    """
    return np.array([
        make_node_features(v, p, cube_wall_positions)
        for v, p in zip(particle_velocities, particle_positions)
    ])


def make_graphs_from_simulation_data(simulation_data, cube_wall_positions,
                                     past_context, connectivity_radius,
                                     save_dir, time_between_steps):
    """Makes the graph and target predictions from the sim data.

    Args:

    simualtion_data: Numpy arra with shape [time_steps, 6, num_particles]

    
    cube_all_positions: A list of size 6 containing the position of
    the 6 walls of the cube. [x_front, x_back, y_left, y_right, z_top,
    z_bottom]

    past_context: Number of previous time steps to use.

    connectivity_radius: The radius to use for the connectivity

    time_between_steps: The time between simulation intervals
    """
    # Creates a lsit with [(time_step_0, ...,
    # time_stpe_past_context+1), (time_step_1, ...,
    # time_step_past_context + 2), ...]. Where each time step has
    # shape [6, num_particles]. The last element in the tuples
    # corresponds to the next time step used to compute the accelarion
    # target.
    time_steps_to_zip = [simulation_data[i + 1:] for i in range(past_context)]
    time_steps_to_zip = list(zip(*time_steps_to_zip))

    for graph_num, context_and_next_time_steps in enumerate(time_steps_to_zip):
        *time_steps, next_time_step = context_and_next_time_steps
        edges, edge_features = make_graph_edges_from_particle_positions(
            time_steps[-1][3:6, :], connectivity_radius)

        # shape [num_particles, t, 3]
        particle_velocities = np.array([
            time_step[0:3, :] for time_step in time_steps
        ]).transpose(2, 0, 1)

        # shape [num_particles, 3]
        particle_positions = time_steps[-1][3:6, :].T

        features = make_graph_node_features(particle_velocities,
                                            particle_positions,
                                            cube_wall_positions)

        data = Data(x=torch.tensor(features),
                    edge_index=torch.tensor(edges),
                    edge_attr=edge_features)

        # Computing the target
        current_velocities = particle_velocities[:, -1:].reshape(
            particle_velocities.shape[0], particle_velocities.shape[-1])
        next_velocities = next_time_step[0:3, :].T
        acc = (next_velocities - current_velocities) / time_between_steps

        # save the graph to disk
        torch.save([data, torch.tensor(acc)],
                   os.path.join(save_dir, f'graph_and_target_{graph_num}.pt'))

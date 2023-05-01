"""Converts the raw data to graphs and their prediction targets."""

import os

from absl import app
from absl import flags

import simulation

FLAGS = flags.FLAGS

flags.DEFINE_string('path_to_simulation_data', None,
                    'The path to the simualtion data.')
flags.DEFINE_string('output_path', None, 'Path to save the graph data to.')

flags.DEFINE_integer(
    'context_size', 5,
    'Number of previous time steps to use for the predictions')
flags.DEFINE_float('connectivity_radius', 0.1,
                   'The radius of the connectivity graph.')
flags.DEFINE_float('time_between_time_steps', 0.003,
                   'The time between time steps.')

flags.DEFINE_list('sims_to_read', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                  'The simulations to read.')


def main(_):
    # Create the output directory if it doesn't exist.
    if not os.path.exists(FLAGS.output_path):
        os.makedirs(FLAGS.output_path)

    for id in FLAGS.sims_to_read:
        data = simulation.read_simulation(FLAGS.path_to_simulation_data, id)

        # Make and save the graphs:
        save_dir = os.path.join(FLAGS.output_path, 'sim_{}'.format(id))
        # If save directory doesn't exist, create it.
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        simulation.graphs_from_simulation_data(data, [0, 1, 0, 1, 0, 1],
                                               FLAGS.context_size,
                                               FLAGS.connectivity_radius,
                                               save_dir,
                                               FLAGS.time_between_time_steps)


if __name__ == '__main__':
    app.run(main)

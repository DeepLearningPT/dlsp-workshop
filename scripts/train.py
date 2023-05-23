"""Trains the model"""
import random

from absl import app
from absl import flags

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

import simulation

FLAGS = flags.FLAGS

flags.DEFINE_string('train_data', None, 'Path to the train data')
flags.DEFINE_string('validation_data', None, 'Path to the validation data')

flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_integer('max_epochs', 10, 'Number of epochs')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')

flags.DEFINE_integer('n_processors', 1, 'Number of processors in the model')
flags.DEFINE_integer('n_hidden_layers', 1, 'Number of hidden layers')
flags.DEFINE_integer('hidden_size', 32, 'Size of the hidden layers')
flags.DEFINE_integer('latent_size', 32, 'Size of the latent space')
flags.DEFINE_enum('aggr', 'add', ['add', 'mean', 'max'], 'Aggregation method')

flags.DEFINE_enum('accelerator', 'gpu', ['gpu', 'cpu'], 'Accelerator')

flags.DEFINE_integer('random_seed', None, 'Random seed')


def main(_):
    if FLAGS.random_seed is not None:
        random.seed(FLAGS.random_seed)

    train_dataset = simulation.datasets.SimulationDataset(FLAGS.train_data)
    validation_dataset = simulation.datasets.SimulationDataset(
        FLAGS.validation_data)

    # Make the dataloaders
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=FLAGS.batch_size,
                                  shuffle=True)
    validation_dataloader = DataLoader(validation_dataset,
                                       batch_size=FLAGS.batch_size,
                                       shuffle=False)

    model = simulation.models.EncoderProcessorDecoder(FLAGS.n_processors,
                                                      FLAGS.n_hidden_layers,
                                                      FLAGS.hidden_size,
                                                      FLAGS.latent_size,
                                                      FLAGS.aggr)

    lightning_wrapper = simulation.lightning_wrapper.GraphWrapper(
        model, batch_size=FLAGS.batch_size, learning_rate=FLAGS.learning_rate)

    trainer = pl.Trainer(max_epochs=FLAGS.max_epochs,
                         accelerator=FLAGS.accelerator)

    trainer.fit(lightning_wrapper, train_dataloader, validation_dataloader)


if __name__ == '__main__':
    app.run(main)

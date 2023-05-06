"""Trains the model"""
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

flags.DEFINE_boolean('use_gpu', True, 'Use GPU')

flags.DEFINE_float('train_split', 0.9, 'Train split')

flags.DEFINE_integer('random_seed', 42, 'Random seed')


def main(_):
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

    for batch in train_dataloader:
        print(batch)
        break

    # TODO(Pedro): Insert the code for the model here
    model = None
    lightning_wrapper = simulation_lightning_wrapper.GraphWrapper(
        model, batch_size=FLAGS.batch_size, learning_rate=FLAGS.learning_rate)

    accelerator = 'gpu' if FLAGS.use_gpu else 'cpu'
    trainer = pl.Trainer(max_epochs=FLAGS.max_epochs, accelerator=accelerator)

    trainer.fit(lightning_wrapper, train_dataloader, validation_dataloader)


if __name__ == '__main__':
    app.run(main)

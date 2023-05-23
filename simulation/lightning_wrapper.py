"""Training logic"""
import torch
import pytorch_lightning as pl


class GraphWrapper(pl.LightningModule):

    def __init__(self, model, batch_size, learning_rate=1e-3):
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def training_step(self, batch, _):
        loss = self._compute_loss(batch)
        self.log('loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 batch_size=self.batch_size,
                 logger=True)
        return loss

    def validation_step(self, batch, _):
        loss = self._compute_loss(batch)
        self.log('val_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 batch_size=self.batch_size,
                 logger=True)
        return loss

    def _compute_loss(self, batch):
        input_graph, target = batch
        preds = self.model(input_graph.x, input_graph.edge_attr,
                           input_graph.edge_index)
        return torch.nn.functional.mse_loss(preds, torch.flatten(target, 0, 1))

    def configure_optimizers(self):
        """Wrap the optimizer into a pl optimizer. """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

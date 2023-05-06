"""Graph Network Simulator and its building blocks."""

from typing import Tuple

import torch
import torch_geometric


class MLP(torch.nn.Module):
    """Creates a Multi-Layer Perceptorn to serve as a building block.
    
    Creates a Multi-Layer Perceptorn with a given number of hidden layers, 
    number of neurons per layer and output dimension. Useful to serve as a 
    building block for larger models. The ReLU function is used as activation 
    for the hidden layers.
    
    Args:
        n_hidden_layers: Number of hidden layers in the MLP., 
        hidden_size: Number of neurons in each hidden layer.
        out_size: Size of the output vector."""

    def __init__(self, n_hidden_layers: int, hidden_size: int,
                 out_size: int) -> None:
        """Initializes an MLP object."""

        super().__init__()

        self._hidden_layers = [
            torch.nn.LazyLinear(n_hidden_layers)
            for _ in range(n_hidden_layers)
        ]
        self._activations = [torch.nn.ReLU() for _ in range(n_hidden_layers)]

        self._out_layer = torch.nn.Linear(hidden_size, out_size)

    def forward(self, *args: torch.Tensor):
        """Forward pass of the MLP.
        
        Args:
            args: Any number of torch.Tensor objects representing the inputs to
            the MLP. The inputs are expected to have size (batch, n_features_i)
            for i in inputs. Inputs are concatenated along the features axis, 
            assumed to be the last one (axis -1).
            
        Returns:
            out: The output of the MLP after its forward pass, with size 
            (batch, out_size)."""

        out = torch.cat(args, dim=-1)

        for layer, activation in zip(self._hidden_layers, self._activations):
            out = layer(out)
            out = activation(out)

        out = self._out_layer(out)

        return out


class Encoder(torch.nn.Module):
    """Encoder layer of the Graph Network Simulator.
    
    The Encoder updates the node and edge features of a graph into latent 
    embeddings. It does so by applying two MLPs, one for nodes and another for 
    edges, independently and without message passing.
    
    Args:
        n_hidden_layers: Number of hidden layers of the two MLPs.
        hidden_size: Size of the hidden layers of the two MLPs.
        latent_size: Size of the latent embeddings of the nodes and edges, i.e. 
          the output size of the MLPs."""

    def __init__(self, n_hidden_layers: int, hidden_size: int,
                 latent_size: int) -> None:
        """Initializes an Encoder object."""

        super().__init__()

        self._nodes_mlp = MLP(n_hidden_layers, hidden_size, latent_size)
        self._edges_mlp = MLP(n_hidden_layers, hidden_size, latent_size)

    def forward(self, node_features: torch.Tensor,
                edge_features: torch.Tensor) -> Tuple(torch.Tensor):
        """Forward pass of the Encoder layer.
        
        Args:
            node_features: Tensor of shape (n_nodes, n_node_features) with the
              features of the nodes of the graph.
            edge_features: Tensor of shape (n_edges, n_edge_features) with the
              features of the edges of the graph.
              
        Returns:
            node_embeddings: Tensor of shape (n_nodes, latent_size) with the
              latent embeddings of the nodes of the graph.
            edge_embeddings: Tensor of shape (n_edges, latent_size) with the
              latent embeddings of the edges of the graph."""

        node_embeddings = self._nodes_mlp(node_features)
        edge_embeddings = self._edge_mlp(edge_features)

        return node_embeddings, edge_embeddings


class Decoder(torch.nn.Module):
    """Decoder layer of the Graph Network Simulator.
    
    The Decoder decodes the relevant information of the nodes 
    (i.e. acceleration) by applying an MLP to each one, independently and 
    without message passing.
    
    Args:
        n_hidden_layers: Number of hidden layers of the MLP.
        hidden_size: Size of the hidden layers of the MLP."""

    def __init__(self, n_hidden_layers: int, hidden_size: int) -> None:
        """Initializes a Decoder layer object."""

        super().__init__()

        # out_size=3 because 3 components of acceleration
        self._nodes_mlp = MLP(n_hidden_layers, hidden_size, out_size=3)

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Decoder layer.
        
        Args:
            node_embeddings: Tensor of shape (n_nodes, latent_size) with the 
              latent node embeddings after message passing steps of the 
              Processor.
              
        Returns:
            accelerations: Tensor of shape (n_nodes, 3) with the three 
              components of the predicted node accelerations."""

        accelerations = self._nodes_mlp(node_embeddings)

        return accelerations


class Processor(torch_geometric.nn.MessagePasssing):
    """Processor layer of the Graph Network Simulator."""

"""Datasets"""
import os

import torch
from torch_geometric.data import Dataset


class SimulationDataset(Dataset):
    """Reads the graphs and prediction targets from disk.

    The files in disk are stored as follows:

    data_dir/
        sim_0000/
            graph_and_target_0000.pt
            graph_and_target_0001.pt
        sim_0001/
            graph_and_target_0000.pt
            graph_and_target_0001.pt

    """

    def __init__(self, data_dir):
        super().__init__(None, None, None)
        # Creates a list with all paths to graphs and targets
        self.graph_and_target_paths = []
        for sim_dir in os.listdir(data_dir):
            sim_dir = os.path.join(data_dir, sim_dir)
            for file in os.listdir(sim_dir):
                file = os.path.join(sim_dir, file)
                self.graph_and_target_paths.append(file)

    def len(self):
        return len(self.graph_and_target_paths)

    def get(self, idx):
        graph_and_target_path = self.graph_and_target_paths[idx]
        graph, target = torch.load(graph_and_target_path)
        return graph, target

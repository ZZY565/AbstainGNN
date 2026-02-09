import torch
from torch.utils.data import Dataset

class GraphListDataset(Dataset):
    def __init__(self, graph_list):
        self.graphs = graph_list

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        return g, int(g.label), idx



def collate_graph_batch(batch):

    graphs, labels, indices = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long)
    indices = torch.tensor(indices, dtype=torch.long)

    return list(graphs), labels, indices


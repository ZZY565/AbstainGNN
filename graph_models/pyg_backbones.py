import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, MessagePassing




def graph_pool(batch_graph, device):
    start_idx = [0]
    for g in batch_graph:
        start_idx.append(start_idx[-1] + len(g.g.nodes()))
    total_nodes = start_idx[-1]

    idx = []
    elem = []
    for i, g in enumerate(batch_graph):
        num_nodes = len(g.g.nodes())
        elem.extend([1] * num_nodes)
        idx.extend([[i, j] for j in range(start_idx[i], start_idx[i + 1])])

    idx = torch.LongTensor(idx).t().to(device)
    elem = torch.FloatTensor(elem).to(device)
    return torch.sparse.FloatTensor(
        idx, elem, torch.Size([len(batch_graph), total_nodes])
    )


def build_edge_index(batch_graph, device):
    edge_list = []
    start = 0
    for g in batch_graph:
        num_nodes = len(g.g.nodes())
        e = g.edge_mat + start
        edge_list.append(e)
        start += num_nodes

    edge_index = torch.cat(edge_list, dim=1).long().to(device)
    return edge_index



class PYG_GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_layers=3, dropout=0.5, device="cuda"):
        super().__init__()

        self.device = torch.device(device)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.linear = nn.Linear(hidden_dim, output_dim)

        self.cached_hidden = None

    def forward(self, batch_graph):
        X = torch.cat([g.node_features for g in batch_graph], dim=0).to(self.device)

        edge_index = build_edge_index(batch_graph, self.device)

        h = X
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        pool = graph_pool(batch_graph, self.device)
        graph_embeddings = torch.spmm(pool, h) 

        self.cached_hidden = graph_embeddings

       
        logits = self.linear(graph_embeddings)
        return logits


class MPNNLayer(MessagePassing):

    def __init__(self, in_dim, out_dim):
        super().__init__(aggr="add")
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        m = torch.cat([x_i, x_j], dim=-1) 
        return self.msg_mlp(m)

    def update(self, aggr_out, x):
     
        h = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(h)


class PYG_MPNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_layers=3, dropout=0.5, device="cuda"):
        super().__init__()

        self.device = torch.device(device)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.ModuleList()
        self.layers.append(MPNNLayer(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(MPNNLayer(hidden_dim, hidden_dim))

        self.linear = nn.Linear(hidden_dim, output_dim)

        self.cached_hidden = None

    def forward(self, batch_graph):
        X = torch.cat([g.node_features for g in batch_graph], dim=0).to(self.device)

        edge_index = build_edge_index(batch_graph, self.device)

        h = X
        for layer in self.layers:
            h = layer(h, edge_index)
            h = F.dropout(h, p=self.dropout, training=self.training)

        pool = graph_pool(batch_graph, self.device)
        graph_embeddings = torch.spmm(pool, h) 

        self.cached_hidden = graph_embeddings

        logits = self.linear(graph_embeddings)
        return logits


from torch_geometric.nn import TransformerConv


from torch_geometric.nn import TransformerConv

class PYG_GraphTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.5, device="cuda", heads=1):
        super().__init__()

        self.device = torch.device(device)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads

        assert hidden_dim % heads == 0, f"hidden_dim ({hidden_dim}) should be divisible by heads ({heads})"
        
        self.transformer_convs = nn.ModuleList()
        self.transformer_convs.append(TransformerConv(input_dim, hidden_dim // self.heads, heads=self.heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.transformer_convs.append(TransformerConv(hidden_dim // self.heads, hidden_dim // self.heads, heads=self.heads, dropout=dropout))
        self.transformer_convs.append(TransformerConv(hidden_dim // self.heads, hidden_dim // self.heads, heads=self.heads, dropout=dropout))

        self.linear = nn.Linear(hidden_dim, output_dim)

        self.cached_hidden = None

    def forward(self, batch_graph):
        X = torch.cat([g.node_features for g in batch_graph], dim=0).to(self.device)

        edge_index = build_edge_index(batch_graph, self.device)

        h = X
        for conv in self.transformer_convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

      
        pool = graph_pool(batch_graph, self.device)
        graph_embeddings = torch.spmm(pool, h)

        self.cached_hidden = graph_embeddings

        logits = self.linear(graph_embeddings)
        return logits

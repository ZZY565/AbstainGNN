import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv


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
    return torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), total_nodes]))


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


class GCNConfidenceHead(nn.Module):

    def __init__(
        self,
        input_dim: int,        
        hidden_dim: int = 64,   
        num_layers: int = 3,   
        dropout: float = 0.5,
        mlp_hidden: int = 64, 
        mlp_layers: int = 2,   
        device: str = "cuda",
    ):
        super().__init__()
        assert num_layers >= 1
        assert mlp_layers >= 1

        self.device = torch.device(device)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        mlp = []
        in_dim = hidden_dim
        if mlp_layers == 1:
            mlp.append(nn.Linear(in_dim, 1))   
        else:
         
            mlp.append(nn.Linear(in_dim, mlp_hidden))
            mlp.append(nn.ReLU(inplace=True))
            for _ in range(mlp_layers - 2):
                mlp.append(nn.Linear(mlp_hidden, mlp_hidden))
                mlp.append(nn.ReLU(inplace=True))
            mlp.append(nn.Linear(mlp_hidden, 1))

        self.mlp = nn.Sequential(*mlp)
        self.sigmoid = nn.Sigmoid()



    def forward(self, batch_graph):
        single_graph = False
        if not isinstance(batch_graph, (list, tuple)):
            batch_graph = [batch_graph]
            single_graph = True

        device = self.device

    
        X = torch.cat([g.node_features for g in batch_graph], dim=0).to(device)

        edge_index = build_edge_index(batch_graph, device)

        h = X
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)

        pool = graph_pool(batch_graph, device)      
        g_emb = torch.spmm(pool, h)               
    
        logit = self.mlp(g_emb)                   
        score = self.sigmoid(logit).squeeze(-1)   

        if single_graph:
            score = score.squeeze(0)
        return score



class GATConfidenceHead(nn.Module):
  

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,    
        dropout: float = 0.0,     
        mlp_hidden: int = 64,
        mlp_layers: int = 2,       
        heads: int = 4,          
        attn_dropout: float = 0.0, 
        concat: bool = False,   
        device: str = "cuda",
    ):
        super().__init__()
        assert num_layers >= 1
        assert mlp_layers >= 1

        self.device = torch.device(device)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(input_dim, hidden_dim, heads=heads, concat=concat, dropout=attn_dropout)
        )

        in_dim = hidden_dim if (not concat) else hidden_dim * heads
        for _ in range(num_layers - 1):
            self.convs.append(
                GATConv(in_dim, hidden_dim, heads=heads, concat=concat, dropout=attn_dropout)
            )
            in_dim = hidden_dim if (not concat) else hidden_dim * heads

        mlp = []
        in_dim = hidden_dim if (not concat) else hidden_dim * heads 
        if mlp_layers == 1:
            mlp.append(nn.Linear(in_dim, 1))
        else:
            mlp.append(nn.Linear(in_dim, mlp_hidden))
            mlp.append(nn.ReLU(inplace=True))
            for _ in range(mlp_layers - 2):
                mlp.append(nn.Linear(mlp_hidden, mlp_hidden))
                mlp.append(nn.ReLU(inplace=True))
            mlp.append(nn.Linear(mlp_hidden, 1))

        self.mlp = nn.Sequential(*mlp)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch_graph):
        single_graph = False
        if not isinstance(batch_graph, (list, tuple)):
            batch_graph = [batch_graph]
            single_graph = True

        device = self.device

        X = torch.cat([g.node_features for g in batch_graph], dim=0).to(device)

        edge_index = build_edge_index(batch_graph, device)

        h = X
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.elu(h) 
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)

        pool = graph_pool(batch_graph, device)
        g_emb = torch.spmm(pool, h)


        logit = self.mlp(g_emb)            
        score = self.sigmoid(logit).squeeze(-1)

        if single_graph:
            score = score.squeeze(0)
        return score







class ReadoutConfidenceHead(nn.Module):

    def __init__(
        self,
        input_dim: int,        
        mlp_hidden: int = 64,
        mlp_layers: int = 2,    
        dropout: float = 0.0,
        readout: str = "mean",  
        device: str = "cuda",
    ):
        super().__init__()
        assert mlp_layers >= 1
        assert readout in ["sum", "mean"]

        self.device = torch.device(device)
        self.input_dim = input_dim
        self.dropout = dropout
        self.readout = readout

        mlp = []
        in_dim = input_dim
        if mlp_layers == 1:
            mlp.append(nn.Linear(in_dim, 1))
        else:
            mlp.append(nn.Linear(in_dim, mlp_hidden))
            mlp.append(nn.ReLU(inplace=True))
            for _ in range(mlp_layers - 2):
                mlp.append(nn.Linear(mlp_hidden, mlp_hidden))
                mlp.append(nn.ReLU(inplace=True))
            mlp.append(nn.Linear(mlp_hidden, 1))

        self.mlp = nn.Sequential(*mlp)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch_graph):
        single_graph = False
        if not isinstance(batch_graph, (list, tuple)):
            batch_graph = [batch_graph]
            single_graph = True

        device = self.device

        X = torch.cat([g.node_features for g in batch_graph], dim=0).to(device)

    
        pool = graph_pool(batch_graph, device)     
        g_emb = torch.spmm(pool, X)                  

        if self.readout == "mean":

            sizes = torch.tensor([len(g.g.nodes()) for g in batch_graph],
                                 device=device, dtype=g_emb.dtype).clamp(min=1.0) 
            g_emb = g_emb / sizes.unsqueeze(-1)

        if self.dropout > 0:
            g_emb = F.dropout(g_emb, p=self.dropout, training=self.training)


        logit = self.mlp(g_emb)                    
        score = self.sigmoid(logit).squeeze(-1)    

        if single_graph:
            score = score.squeeze(0)
        return score

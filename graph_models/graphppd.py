import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphPPD(nn.Module):
    def __init__(self, dx: int, num_classes: int,
                 dh: int = 64, heads: int = 1,
                 mlp_hidden: int = 128, mlp_layers: int = 2,
                 dropout: float = 0.0):
        super().__init__()

        assert heads >= 1
        self.dx = dx
        self.C = num_classes
        self.dh = dh
        self.heads = heads
        self.dropout = dropout

        self.Wq = nn.Linear(dx, heads * dh, bias=False)
        self.Wk = nn.Linear(dx, heads * dh, bias=False)
        self.Wv = nn.Linear(dx + num_classes, heads * dh, bias=False)

  
        in_dim = dx + heads * dh
        mlp = []
        cur = in_dim
        for li in range(mlp_layers - 1):
            mlp += [nn.Linear(cur, mlp_hidden), nn.ReLU(inplace=True)]
            if dropout > 0:
                mlp += [nn.Dropout(dropout)]
            cur = mlp_hidden
        mlp += [nn.Linear(cur, num_classes)]
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x_t: torch.Tensor, x_c: torch.Tensor, y_c_onehot: torch.Tensor) -> torch.Tensor:

        B = x_t.size(0)
        K = x_c.size(0)

   
        if K == 0:
            raise ValueError("GraphPPD forward got empty context set (K=0).")

        Q = self.Wq(x_t)
        Kproj = self.Wk(x_c)

      
        ctx = torch.cat([x_c, y_c_onehot], dim=-1) 
        Vproj = self.Wv(ctx)                     


        H = self.heads
        dh = self.dh
        Qh = Q.view(B, H, dh)
        Kh = Kproj.view(K, H, dh)
        Vh = Vproj.view(K, H, dh)

        scores = torch.einsum("bhd,khd->bhk", Qh, Kh) / (dh ** 0.5)

      
        attn = torch.softmax(scores, dim=-1) 
        if self.dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout)

      
        r = torch.einsum("bhk,khd->bhd", attn, Vh).contiguous() 
        r = r.view(B, H * dh) 

        xtilde = torch.cat([x_t, r], dim=-1) 

        logits = self.mlp(xtilde) 
        return logits

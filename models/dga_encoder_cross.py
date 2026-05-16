import torch
import torch.nn as nn
from einops import rearrange

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
    
class GaussianWeightedSE(nn.Module):
    def __init__(self, in_channels, reduction=16, SEsigma_init=1.0):
        super(GaussianWeightedSE, self).__init__()
        self.in_channels = in_channels
        self.log_sigma_se = nn.Parameter(torch.log(torch.tensor(SEsigma_init)))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, mu):
        batch_size, c, h, w = x.size()
        sigma_se = torch.exp(self.log_sigma_se)

        # Gaussian pooling based on mu
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(0, 1, h, device=x.device),
            torch.linspace(0, 1, w, device=x.device),
            indexing='ij'
        )
        grid_y = grid_y.expand(batch_size, -1, -1) 
        grid_x = grid_x.expand(batch_size, -1, -1)

        mu_h = mu[:, 0].view(batch_size, 1, 1)
        mu_w = mu[:, 1].view(batch_size, 1, 1)

        dx = grid_x - mu_h
        dy = grid_y - mu_w

        kernel = torch.exp(-(dx**2 + dy**2) / (2 * sigma_se))
        
        kernel = kernel.view(batch_size, -1)
        kernel = kernel / kernel.sum(dim=1, keepdim=True)
        kernel = kernel.view(batch_size, h, w)

        squeezed = torch.einsum('bchw,bhw->bc', x, kernel)
        
        weights = self.fc(squeezed).view(batch_size, c, 1, 1)
        return x * weights


class GaussianAttention(nn.Module):
    def __init__(self, in_dim, sigma_init=1, SEsigma_init=1, dropout=0.1):
        super(GaussianAttention, self).__init__()
        self.query_linear = nn.Linear(in_dim, in_dim//4)
        self.key_linear = nn.Linear(in_dim, in_dim//4)
        self.value_linear = nn.Linear(in_dim, in_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.log_sigma_sq = nn.Parameter(torch.log(torch.tensor(sigma_init))) 
        self.dropout = nn.Dropout(dropout)
        self.scale = (in_dim//4) ** -0.5

        
        self.se = GaussianWeightedSE(in_dim, reduction=16, SEsigma_init=SEsigma_init)

    def forward(self, x_Q, x_KV):
        m_batchsize, c, height, width = x_Q.shape

        # Reshape and QKV
        x_Q_reshaped = x_Q.view(m_batchsize, c, height * width).permute(0, 2, 1)  # B, HW, C
        x_KV_reshaped = x_KV.view(m_batchsize, c, height * width).permute(0, 2, 1)  # (B, HW, C)
        
        queries = self.query_linear(x_Q_reshaped)  # B, HW, C//4
        keys = self.key_linear(x_KV_reshaped).permute(0, 2, 1)  # B, C//4, HW
        values = self.value_linear(x_KV_reshaped)  # B, HW, C
       
        # Gaussian distance prior
        grid_x, grid_y = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        positions = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=1).to(x_Q.device).float()
       
        dx = positions[:,0].unsqueeze(1) - positions[:,0].unsqueeze(0)  # (HW, HW)
        dy = positions[:,1].unsqueeze(1) - positions[:,1].unsqueeze(0)
        distances_sq = dx**2 + dy**2
        
        max_distance_sq = ((height-1)**2 + (width-1)**2)
        normalized_distances_sq = distances_sq / max_distance_sq
        
        sigma_sq = torch.exp(self.log_sigma_sq)  
        D_2d = torch.exp(-normalized_distances_sq / (2 * sigma_sq))
        D_2d = D_2d.unsqueeze(0).expand(m_batchsize, -1, -1)  # (B, HW, HW)
        
        # Gaussian attention
        attention_scores = torch.bmm(queries, keys) * self.scale
        attention_scores = self.softmax(attention_scores)* D_2d
        attention_scores = self.dropout(attention_scores)
        
        out = torch.bmm(attention_scores, values).permute(0,2,1).view(m_batchsize, c, height, width)

        # Calculate mu
        attention_sums = attention_scores.sum(dim=1)  # (B, HW)
        weights = attention_sums / (attention_sums.sum(dim=1, keepdim=True) + 1e-8)  
        
        mu_h = torch.sum(positions[:, 0] * weights, dim=1) / (height - 1)
        mu_w = torch.sum(positions[:, 1] * weights, dim=1) / (width - 1)
        mu = torch.stack([mu_h, mu_w], dim=1).to(x_Q.device)
        out = self.se(out, mu)  
        return out

class Transformer(nn.Module):
    def __init__(self, dim_q, hidden_dim, dropout=0.1, sigma_init=1, SEsigma_init=1):
        super(Transformer, self).__init__()
        self.att = GaussianAttention(dim_q, sigma_init=sigma_init, SEsigma_init=SEsigma_init, dropout=dropout)
        self.feedforward = FeedForward(dim_q, hidden_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim_q)
        self.norm2 = nn.LayerNorm(dim_q)

    
    def forward(self, x_Q, x_KV_normalized_from_other_branch):
        b, c, h, w = x_Q.shape

        residual1 = x_Q
        
        x_Q_norm = x_Q.permute(0, 2, 3, 1)  # B, H, W, C
        x_Q_norm = self.norm1(x_Q_norm)
        x_Q_norm = x_Q_norm.permute(0, 3, 1, 2)  
        
        x_att = self.att(x_Q_norm, x_KV_normalized_from_other_branch)

        x_att_res = x_att + residual1

        x_norm2 = x_att_res.permute(0, 2, 3, 1)  # B, H, W, C
        x_norm2 = self.norm2(x_norm2)
        x_norm2 = rearrange(x_norm2, 'b h w c -> (b h w) c')
        x_ff = self.feedforward(x_norm2)
        x_ff = rearrange(x_ff, '(b h w) c -> b h w c', b=b, h=h, w=w).permute(0, 3, 1, 2)

        x_out = x_ff + x_att_res
        return x_out

class DifferenceEncoder(nn.Module):
    def __init__(self, n_layers, feature_size, hidden_dim, dropout=0.1, sigma_init=1, SEsigma_init=1):
        super().__init__()
        h_feat, w_feat, channels = feature_size

        self.embedding_h = nn.Embedding(h_feat, int(channels/2))
        self.embedding_w = nn.Embedding(w_feat, int(channels/2))
        self.DGA = nn.ModuleList([])
        for _ in range(n_layers):
            self.DGA.append(nn.ModuleList([
                Transformer(channels, hidden_dim, dropout, sigma_init, SEsigma_init),
                Transformer(channels, hidden_dim, dropout, sigma_init, SEsigma_init)
            ]))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, imgA, imgB):
        batch, c, h, w = imgA.shape
        
        pos_h = torch.arange(h, device=imgA.device)
        pos_w = torch.arange(w, device=imgA.device)

        embed_h = self.embedding_h(pos_h)
        embed_w = self.embedding_w(pos_w)
        position_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1),
                                       embed_h.unsqueeze(1).repeat(1, w, 1)],
                                       dim=-1)
        position_embedding = position_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1, 1)

        feat_A = imgA + position_embedding
        feat_B = imgB + position_embedding

        for i in range(len(self.DGA)):
            transformer_A = self.DGA[i][0]
            transformer_B = self.DGA[i][1]

            norm_feat_A_for_B_KV = feat_A.permute(0, 2, 3, 1)
            norm_feat_A_for_B_KV = transformer_A.norm1(norm_feat_A_for_B_KV) # Use transformer_A's norm1
            norm_feat_A_for_B_KV = norm_feat_A_for_B_KV.permute(0, 3, 1, 2)

            norm_feat_B_for_A_KV = feat_B.permute(0, 2, 3, 1)
            norm_feat_B_for_A_KV = transformer_B.norm1(norm_feat_B_for_A_KV) # Using B's norm
            norm_feat_B_for_A_KV = norm_feat_B_for_A_KV.permute(0, 3, 1, 2)

            out_A = transformer_A(feat_A, norm_feat_B_for_A_KV)
            out_B = transformer_B(feat_B, norm_feat_A_for_B_KV)

            feat_A = out_A
            feat_B = out_B

        return feat_A, feat_B
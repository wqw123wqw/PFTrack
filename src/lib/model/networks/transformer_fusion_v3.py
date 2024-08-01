# --------------------------------------------------------------------*/
# This file includes code from https://github.com/facebookresearch/detr/blob/main/models/detr.py
# --------------------------------------------------------------------*/
#


import copy
import torch
import torch.nn.functional as F
from torch import nn
import pdb


class Transformer_Fusion(nn.Module):
    def __init__(self, d_model=64, nhead=8, num_fusion_encoder_layers=2, dim_feedforward=2048,
                 dropout=0.1, activation="relu", normalize_before=False, return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_fusion_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead       
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    def forward(self, src_v,src_t, pos_embed):

        output_v,output_t = self.encoder(src_v, src_t, pos=pos_embed)
        output=output_v+output_t
        return output


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm = norm
    def forward(self, src_v, src_t, pos=None):
        output_v = src_v
        output_t = src_t
        for layer in self.layers:
            output_v,output_t= layer(output_v,output_t, pos=pos)
        return output_v,output_t

        
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        #self.self_attn = nn.MultiheadAttention(d_model*2, nhead, dropout=dropout)
        self.multihead_attn_fv = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_ft = nn.MultiheadAttention(d_model, nhead, dropout=dropout)         
        self.multihead_attn_v = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn_t = nn.MultiheadAttention(d_model, nhead, dropout=dropout)        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(4*d_model, dim_feedforward)       
        self.linear2 = nn.Linear(dim_feedforward, d_model)        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model) 
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm5 = nn.LayerNorm(d_model)        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)                           
        self.dropout5 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src_v, src_t, pos=None,c=1,n=2):
        src=src_v+src_t
        
        q_f = k_f = self.with_pos_embed(src, pos)
        q_v = k_v = self.with_pos_embed(src_v, pos)        
        q_t = k_t = self.with_pos_embed(src_t, pos)        
        
        src_fv1 = self.multihead_attn_fv(q_f, k_v, value=src_v)[0]
        src_fv = src + self.dropout1(src_fv1)
        src_fv = self.norm1(src_fv)

        src_ft1 = self.multihead_attn_ft(q_f, k_t, value=src_t)[0]
        src_ft = src + self.dropout2(src_ft1)
        src_ft = self.norm2(src_ft)

        #k_f=self.with_pos_embed(src, pos)
        src_v1 = self.multihead_attn_v(q_v, k_f, value=src)[0]
        src_v = src_v + self.dropout3(src_v1)
        src_v = self.norm3(src_v)

        q_t = self.with_pos_embed(src_t, pos)
        #k_f=self.with_pos_embed(src, pos)
        src_t1= self.multihead_attn_t(q_t, k_f, value=src)[0]
        src_t = src_t + self.dropout4(src_t1)
        src_t = self.norm4(src_t)
        
        
        src_f=torch.cat((src_fv,src_ft,src_v,src_t),2)        
        src_f = self.linear2(self.dropout5(self.activation(self.linear1(src_f))))
        src_f = self.norm5(src_f) 
                        
        return src_v,src_t



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer_fusion(args):
    return Transformer_Fusion(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu/glu, not {activation}.")
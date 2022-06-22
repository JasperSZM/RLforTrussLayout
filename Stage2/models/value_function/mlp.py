import torch, os, math
import numpy as np
from torch import nn
from torch.nn import functional as F
from rlkit.policies.base import Policy
from rlkit.pythonplusplus import identity
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule, eval_np
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.networks import LayerNorm, ConcatMlp, Mlp
from rlkit.torch.pytorch_util import activation_from_string

class MLP(ConcatMlp):
    def __init__(self, input_dim, hidden_dims):
        super().__init__(input_size=input_dim, output_size=1, hidden_sizes=hidden_dims)

class _Mlp(Mlp):
    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

class TRANSFORMEREMBED(_Mlp):
    """
        represented by edge sequences --> MLP embed to high dim --> transformer --> MLP to dim 1
    """
    def __init__(self, input_dims, hidden_dims):
        super().__init__(input_size=input_dims[-1], output_size=1, hidden_sizes=hidden_dims)
        self.embed_dim = input_dims[-1]
        #print(input_dims[-1])
        #print(input_dims[:-1])
        self.embed_node = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=3)
        self.embed_edge = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=5)
        self.embed_id = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=2)
        self.embed_act = Mlp(hidden_sizes=input_dims[:-1], output_size=self.embed_dim, input_size=3)
        self.transformer = nn.Transformer(d_model=self.embed_dim, nhead=4, num_encoder_layers=2)

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self.dim).cuda()
        #print(flat_inputs.shape)
        flat_dim = flat_inputs.shape[1]
        act_inputs = flat_inputs[..., -3:]
        id_inputs = flat_inputs[..., -5: -3]
        num_points = int((math.sqrt(25 + 8 * (flat_dim - 5)) - 5) / 2)
        #print(num_points)
        pos_inputs = flat_inputs[..., :2 * num_points]
        force_inputs = flat_inputs[..., -5 - num_points: -5]
        edge_inputs = flat_inputs[..., 2 * num_points: -5 - num_points]
        node_outputs = NodeEmbedding(pos_inputs, force_inputs, num_points).cuda()
        embed_node_outputs = self.embed_node(node_outputs)
        #print(node_outputs.shape, embed_node_outputs.shape)
        edge_outputs = EdgeEmbedding(pos_inputs, edge_inputs, num_points).cuda()
        embed_edge_outputs = self.embed_edge(edge_outputs)
        #print(edge_outputs.shape, embed_edge_outputs.shape)
        embed_id_outputs = self.embed_id(id_inputs).unsqueeze(1)
        embed_act_outputs = self.embed_act(act_inputs).unsqueeze(1)
        #print(embed_id_outputs.shape, embed_act_outputs.shape)

        src = torch.cat([embed_node_outputs, embed_edge_outputs, embed_id_outputs], dim=1).transpose(0, 1)
        #print(src.shape)
        tgt = embed_act_outputs.transpose(0, 1)
        #print(tgt.shape)
        outs = self.transformer(src, tgt).transpose(0, 1).squeeze(dim=1)
        #print(outs.shape)
        return super().forward(outs, **kwargs)

def EdgeEmbedding(pos_inputs, edge_inputs, num_points):
    _pos_inputs = pos_inputs.reshape(pos_inputs.shape[0], num_points, 2)
    outputs = []
    for k in range(_pos_inputs.shape[0]):
        one_output = []
        idx = 0
        i = 0
        j = 1
        while idx < num_points * (num_points - 1) / 2:
            one_edge = []
            v_i = [_pos_inputs[k][i][0], _pos_inputs[k][i][1]]
            v_j = [_pos_inputs[k][j][0], _pos_inputs[k][j][1]]
            area_ij = edge_inputs[k][idx]
            one_edge += v_i
            one_edge += v_j
            one_edge.append(area_ij)
            idx += 1
            i += 1
            if i == j:
                i = 0
                j += 1
            one_output.append(one_edge)
        outputs.append(one_output)
    return torch.Tensor(outputs)

def NodeEmbedding(pos_inputs, force_inputs, num_points):
    _pos_inputs = pos_inputs.reshape(pos_inputs.shape[0], num_points, 2)
    _force_inputs = force_inputs.reshape(force_inputs.shape[0], num_points, 1)
    return torch.cat([_pos_inputs, _force_inputs], dim=-1)


# for flat_input, output is a sequence of tuple with size 5, each one is (P1.x, P1.y, P2.x, P2.y, area)
def Dim2EdgeEmbedding(flat_input, num_points):
    UseNormalizationEdge = False
    if UseNormalizationEdge:
        NormalizationEdge = 1000
    else:
        NormalizationEdge = 1
    output = []
    obs_action_dim = flat_input.size()[1]
    num_edges = int(num_points * (num_points - 1) / 2)
    obs_dim = num_points * 2 + num_edges
    act_dim = obs_action_dim - obs_dim
    fixed_points = int((obs_dim - act_dim) / 2)
    for one_input in flat_input:
        points = []
        changed_points = []
        for i in range(num_points):
            points.append([one_input[2 * i], one_input[2 * i + 1]])
        for i in range(num_points):
            if i < fixed_points:
                changed_points.append(points[i])
            else:
                changed_points.append([points[i][0] + one_input[(i - fixed_points) * 2 + obs_dim], points[i][1] + one_input[(i - fixed_points) * 2 + obs_dim + 1]])

        together_edges = []
        edges = []
        changed_edges = []
        idx = 2 * num_points
        changed_idx = obs_dim + 2 * (num_points - fixed_points)
        i = 0
        j = 1
        while idx < obs_dim:
            one_edge = []
            one_edge += points[i]
            one_edge += points[j]
            one_edge.append(one_input[idx] * NormalizationEdge)
            edges.append(one_edge)
            one_changed_edge = []
            one_changed_edge += changed_points[i]
            one_changed_edge += changed_points[j]
            one_changed_edge.append(one_input[changed_idx] * NormalizationEdge + one_input[idx] * NormalizationEdge)
            changed_edges.append(one_changed_edge)
            together_edges.append(one_edge)
            together_edges.append(one_changed_edge)
            idx += 1
            changed_idx += 1
            i += 1
            if i >= j:
                j += 1
                i = 0
        output.append(edges + [[-1, -1, -1, -1, -1],] + changed_edges)
        #output.append(together_edges)
    return torch.Tensor(output)

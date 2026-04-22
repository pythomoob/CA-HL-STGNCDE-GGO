import torch
import torch.nn.functional as F
import torch.nn as nn
import controldiffeq
from vector_fields import *


class InputFeatureAttention(nn.Module):
    def __init__(self, input_dim, reduction=2):
        super(InputFeatureAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // reduction, input_dim, bias=False),
            nn.Sigmoid()
        )

        self.weight_history = []
        self.record_weights = False

    def forward(self, coeffs):
        if isinstance(coeffs, (list, tuple)):
            coeffs_tensor = torch.stack(coeffs, dim=0)
        else:
            coeffs_tensor = coeffs.unsqueeze(0)

        all_dims = range(coeffs_tensor.dim())
        dims_to_pool = [d for d in all_dims if d != 1 and d != coeffs_tensor.dim() - 1]

        y = torch.mean(coeffs_tensor, dim=dims_to_pool)
        weights = self.fc(y)

        if self.record_weights:
            self.weight_history.append(weights.detach().cpu().numpy())

        sample_coeff = coeffs[0] if isinstance(coeffs, (list, tuple)) else coeffs

        view_shape = [sample_coeff.shape[0]] + [1] * (sample_coeff.dim() - 2) + [sample_coeff.shape[-1]]
        weights_reshaped = weights.view(*view_shape)

        if isinstance(coeffs, list):
            return [c * weights_reshaped for c in coeffs]
        elif isinstance(coeffs, tuple):
            return tuple(c * weights_reshaped for c in coeffs)
        else:
            return coeffs * weights_reshaped

class NeuralGCDE(nn.Module):
    def __init__(self, args, func_f, func_g, input_channels, hidden_channels, output_channels, initial, device, atol, rtol, solver):
        super(NeuralGCDE, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = input_channels
        self.feature_attention = InputFeatureAttention(self.input_dim, reduction=2)
        self.hidden_dim = hidden_channels
        self.output_dim = output_channels
        self.horizon = args.horizon
        self.num_layers = args.num_layers

        self.default_graph = args.default_graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)
        
        self.func_f = func_f
        self.func_g = func_g
        self.solver = solver
        self.atol = atol
        self.rtol = rtol

        #predictor
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

        self.init_type = 'fc'
        if self.init_type == 'fc':
            self.initial_h = torch.nn.Linear(self.input_dim, self.hidden_dim)
            self.initial_z = torch.nn.Linear(self.input_dim, self.hidden_dim)
        elif self.init_type == 'conv':
            self.start_conv_h = nn.Conv2d(in_channels=input_channels,
                                            out_channels=hidden_channels,
                                            kernel_size=(1,1))
            self.start_conv_z = nn.Conv2d(in_channels=input_channels,
                                            out_channels=hidden_channels,
                                            kernel_size=(1,1))

    def forward(self, times, coeffs):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)
        coeffs = self.feature_attention(coeffs)
        spline = controldiffeq.NaturalCubicSpline(times, coeffs)
        if self.init_type == 'fc':
            h0 = self.initial_h(spline.evaluate(times[0]))
            z0 = self.initial_z(spline.evaluate(times[0]))
        elif self.init_type == 'conv':
            h0 = self.start_conv_h(spline.evaluate(times[0]).transpose(1,2).unsqueeze(-1)).transpose(1,2).squeeze()
            z0 = self.start_conv_z(spline.evaluate(times[0]).transpose(1,2).unsqueeze(-1)).transpose(1,2).squeeze()

        z_t = controldiffeq.cdeint_gde_dev(dX_dt=spline.derivative, #dh_dt
                                   h0=h0,
                                   z0=z0,
                                   func_f=self.func_f,
                                   func_g=self.func_g,
                                   t=times,
                                   method=self.solver,
                                   atol=self.atol,
                                   rtol=self.rtol,
                                   adjoint=False)

        # init_state = self.encoder.init_hidden(source.shape[0])
        # output, _ = self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden
        # output = output[:, -1:, :, :]                                   #B, 1, N, hidden
        z_T = z_t[-1:,...].transpose(0,1)

        #CNN based predictor
        output = self.end_conv(z_T)                         #B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)                             #B, T, N, C

        return output
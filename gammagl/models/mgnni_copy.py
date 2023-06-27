import numpy as np
import tensorlayerx as tlx
import tensorlayerx.nn as nn

from gammagl.utils import degree
from gammagl.layers.conv import MessagePassing


class MGNNI(tlx.nn.Module):
    def __init__(self, m, m_y, nhid, ks, threshold, max_iter, gamma, fp_layer='MGNNI_m_att', dropout=0.5,
                 batch_norm=False):
        super(MGNNI, self).__init__()
        self.fc1 = tlx.layers.Linear(out_features=nhid,
                                     in_features=m,
                                     # W_init='xavier_uniform',
                                     b_init=None)
        self.fc2 = tlx.layers.Linear(out_features=nhid,
                                     in_features=nhid,
                                     # W_init='xavier_uniform',
                                     )

        self.dropout = dropout
        self.MGNNI_layer = eval(fp_layer)(nhid, m_y, ks, threshold, max_iter, gamma, dropout=self.dropout,
                                          batch_norm=batch_norm)

    def forward(self, X, edge_index, edge_weight=None, num_nodes=None):
        # print('I am coming!')
        X = nn.Dropout(p=self.dropout)(tlx.ops.transpose(X))
        X = nn.ReLU()(self.fc1(X))
        X = nn.Dropout(p=self.dropout)(X)
        X = self.fc2(X)
        output = self.MGNNI_layer(tlx.ops.transpose(X), edge_index, edge_weight, num_nodes)

        return output


class MGNNI_m_att(nn.Module):
    def __init__(self, m, m_y, ks, threshold, max_iter, gamma, dropout=0.5,
                 layer_norm=False, batch_norm=False):
        super(MGNNI_m_att, self).__init__()
        self.dropout = tlx.layers.Dropout(p=dropout)
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.MGNNIs = nn.ModuleList()
        self.att = Attention(in_size=m)
        for k in ks:
            self.MGNNIs.append(MGNNI_m_iter(m, k, threshold, max_iter, gamma, layer_norm=layer_norm))
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(num_features=m, momentum=0.8)

        # self.B = nn.Parameter(1. / np.sqrt(m))
        tmp = tlx.convert_to_numpy(tlx.random_uniform((m_y, m)))
        # m = tlx.convert_to_tensor(m)
        tmp = tlx.convert_to_tensor(1. / np.sqrt(m) * tmp)

        self.B = nn.Parameter(tmp)
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.MGNNIs)):
            self.MGNNIs[i].reset_parameters()

    def get_att_vals(self, x, edge_index, edge_weight, num_nodes):
        outputs = []
        for idx, model in enumerate(self.MGNNIs):
            tmp = model(x, edge_index, edge_weight, num_nodes)
            outputs.append(tlx.ops.transpose(tmp))
        outputs = tlx.stack(outputs, axis=1)
        att_vals = self.att(outputs)
        return att_vals

    def forward(self, X, edge_index, edge_weight, num_nodes):
        outputs = []
        for idx, model in enumerate(self.MGNNIs):
            tmp = model(X, edge_index=edge_index, edge_weight=edge_weight, num_nodes=num_nodes)
            # tmp = model(X, adj).t()
            outputs.append(tlx.ops.transpose(tmp))
        outputs = tlx.stack(outputs, axis=1)
        att_vals = self.att(outputs)
        outputs = tlx.reduce_sum(outputs * att_vals, 1)

        if self.batch_norm:
            outputs = self.bn1(outputs)
        outputs = self.dropout(outputs)
        outputs = outputs @ tlx.ops.transpose(self.B)
        return outputs

# global cnt = 1

class MGNNI_m_iter(MessagePassing):
    def __init__(self, m, k, threshold, max_iter, gamma, layer_norm=False):
        super(MGNNI_m_iter, self).__init__()
        self.F = nn.Parameter(tlx.convert_to_tensor(np.zeros((m, m)), dtype=tlx.float32))
        self.layer_norm = layer_norm
        self.gamma = gamma
        # self.gamma = nn.Parameter(tlx.convert_to_tensor(gamma, dtype=tlx.float32), name="gammaGL")
        self.k = k
        self.max_iter = max_iter
        self.threshold = threshold
        self.f_solver = fwd_solver
        self.b_solver = fwd_solver

        # self.reset_parameters()

    def reset_parameters(self):
        initor = tlx.initializers.Zeros()
        self.F = self._get_weights("F", shape=self.F.shape, init=initor)


        # torch.nn.init.zeros_(self.F)
        # torch.nn.init.xavier_uniform(self.F) # 使用这种初始化方法就无法完成收敛
        # pass
        # x = tuple(self.F.shape)
        # self.F = nn.initializers.xavier_uniform()(shape=self.F.shape)


    def _inner_func(self, Z, X, edge_index, edge_weight, num_nodes):
        P = tlx.ops.transpose(Z)
        ei = tlx.ops.convert_to_tensor(tlx.ops.convert_to_numpy(edge_index))
        # ei = edge_index.requires_grad_(False)
        src, dst = ei[0], ei[1]
        if edge_weight is None:
            edge_weight = tlx.ones(shape=(ei.shape[1], 1))
        edge_weight = tlx.reshape(edge_weight, (-1,))
        weights = edge_weight

        deg = degree(src, num_nodes=num_nodes, dtype=tlx.float32)
        norm = tlx.pow(deg, -0.5)
        weights = tlx.ops.gather(norm, src) * tlx.reshape(edge_weight, (-1,))

        deg = degree(dst, num_nodes=num_nodes, dtype=tlx.float32)
        norm = tlx.pow(deg, -0.5)
        weights = tlx.reshape(weights, (-1,)) * tlx.ops.gather(norm, dst)

        for _ in range(self.k):
            P = self.propagate(P, ei, edge_weight=weights, num_nodes=num_nodes)

        Z = tlx.ops.transpose(P)

        Z_new = self.gamma * g(self.F) @ Z + X
        del Z, P, ei
        return Z_new

    # @profile
    # def _inner_func(self, Z, X, edge_index, edge_weight, num_nodes):
    #     P = Z.t()
    #     ei = edge_index.requires_grad_(False)
    #     src, dst = ei[0], ei[1]
    #     if edge_weight is None:
    #         edge_weight = tlx.ones(shape=(ei.shape[1], 1))
    #     edge_weight = tlx.reshape(edge_weight, (-1,))
    #     weights = edge_weight
    #
    #     deg = degree(src, num_nodes=num_nodes, dtype=tlx.float32)
    #     norm = tlx.pow(deg, -0.5)
    #     weights = tlx.ops.gather(norm, src) * tlx.reshape(edge_weight, (-1,))
    #
    #     deg = degree(dst, num_nodes=num_nodes, dtype=tlx.float32)
    #     norm = tlx.pow(deg, -0.5)
    #     weights = tlx.reshape(weights, (-1,)) * tlx.ops.gather(norm, dst)
    #
    #     for _ in range(self.k):
    #         P = self.propagate(P, ei, edge_weight=weights, num_nodes=num_nodes)
    #
    #     Z = P.t()
    #
    #     Z_new = self.gamma * g(self.F) @ Z + X
    #     del Z, P, ei
    #     return Z_new

    # # @profile
    # def _inner_func(self, Z, X, edge_index, edge_weight, num_nodes):
    #     P = Z.t()
    #     ei = edge_index.requires_grad_(False)
    #     for _ in range(self.k):
    #         P = self.propagate(P, ei, edge_weight=edge_weight, num_nodes=num_nodes)
    #     # gc.collect()
    #     Z = P.t()
    #
    #     Z_new = self.gamma * g(self.F) @ Z + X
    #     del Z, P, ei
    #     return Z_new

    # @profile
    def forward(self, X, edge_index, edge_weight, num_nodes):
        Z, abs_diff = self.f_solver(lambda Z: self._inner_func(Z, X, edge_index, edge_weight, num_nodes),
                                    z_init=tlx.zeros_like(X),
                                    threshold=self.threshold,
                                    max_iter=self.max_iter)
        Z = tlx.convert_to_tensor(tlx.convert_to_numpy(Z))

        new_Z = Z

        if self.is_train:
            if tlx.BACKEND != 'paddle':
                new_Z = self._inner_func(tlx.Variable(Z, 'Z'), X, edge_index, edge_weight, num_nodes)
            else:
                Z.stop_gradient = False
                new_Z = self._inner_func(Z, X, edge_index, edge_weight, num_nodes)

        # if tlx.BACKEND == 'torch':
        #     with torch.no_grad():
        #         Z, abs_diff = self.f_solver(lambda Z: self._inner_func(Z, X, edge_index, edge_weight, num_nodes),
        #                                     z_init=torch.zeros_like(X),
        #                                     threshold=self.threshold,
        #                                     max_iter=self.max_iter)
        #     new_Z = Z
        #     if self.training:
        #         new_Z = self._inner_func(Z.requires_grad_(), X, edge_index, edge_weight, num_nodes)
        #
        #     return new_Z
        # elif tlx.BACKEND == 'tensorflow':
        #     pass
        # elif tlx.BACKEND == 'paddle':
        #     pass

        # Z, abs_diff = self.f_solver(lambda Z: self._inner_func(Z, X, edge_index, edge_weight, num_nodes),
        #               z_init=tlx.zeros_like(X),
        #               threshold=self.threshold,
        #               max_iter=self.max_iter)
        # Z = tlx.convert_to_tensor(tlx.convert_to_numpy(Z))
        #
        # new_Z = Z
        # if self.is_train:
        #     if tlx.BACKEND == 'torch':
        #         new_Z = self._inner_func(Z.requires_grad_(), X, edge_index, edge_weight, num_nodes)
        #
        #         def backward_hook(grad):
        #             # print('backward_hook')
        #             if self.hook is not None:
        #                 self.hook.remove()
        #                 torch.cuda.synchronize()
        #             result, b_abs_diff = self.b_solver(lambda y: torch.autograd.grad(new_Z, Z, y, retain_graph=True)[0] + grad,
        #                                                z_init=torch.zeros_like(X),
        #                                                threshold=self.threshold,
        #                                                max_iter=self.max_iter)
        #             return result
        #         self.hook = new_Z.register_hook(backward_hook)
        #
        return new_Z


# @profile
def fwd_solver(f, z_init, threshold, max_iter, mode='abs'):
    z_prev, z = z_init, f(z_init)
    nstep = 0
    while nstep < max_iter:
        z_prev, z = z, f(z)
        # torch
        abs_diff = tlx.ops.convert_to_numpy(norm(z_prev - z)).item()
        # rel_diff = abs_diff / (torch.norm(z_prev).detach().numpy().item() + 1e-9)
        # abs_diff = norm(z_prev - z).numpy().item()
        # rel_diff = abs_diff / (torch.norm(z_prev).numpy().item() + 1e-9)
        # diff_dict = {'abs': abs_diff, 'rel': rel_diff}
        if abs_diff < threshold:
            break
        nstep += 1
        del z_prev
    if nstep == max_iter:
        print(f'step {nstep}, not converged, abs_diff: {abs_diff}')
    return z, abs_diff


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(out_features=hidden_size, in_features=in_size),
            nn.Tanh(),
            nn.Linear(out_features=1, in_features=hidden_size)
        )

    def forward(self, z):
        w = self.project(z)
        beta = tlx.softmax(w, axis=1)
        return beta


epsilon_F = 10 ** (-12)


# def norm(X, p='fro'):
#     res = X
#     if tlx.BACKEND == 'torch':
#         res = torch.norm(X, p=p)
#     elif tlx.BACKEND == 'tensorflow':
#         res = tf.norm(X)
#
#     return res

def norm(input, p="fro", dim=None, keepdim=False, out=None, dtype=None):
    if p == "fro":
        norm_np = np.linalg.norm(tlx.convert_to_numpy(input), ord="fro", axis=dim, keepdims=keepdim)
    elif p == "nuc":
        norm_np = np.linalg.norm(tlx.convert_to_numpy(input), ord="nuc", axis=dim, keepdims=keepdim)
    else:
        norm_np = np.linalg.norm(tlx.convert_to_numpy(input), ord=p, axis=dim, keepdims=keepdim)

    op = tlx.convert_to_tensor(norm_np)
    if (tlx.BACKEND == "paddle"):
        op.stop_gradient = False
    else:
        op = tlx.Variable(op, 'op')

    # print('2w22')
    # o = tlx.convert_to_tensor(norm_np)
    return op


def g(F):
    FF = tlx.ops.transpose(F) @ F
    FF_norm = norm(FF, p='fro')
    return (1 / (FF_norm + epsilon_F)) * FF


def get_G(Lambda_F, Lambda_S, gamma):
    G = 1.0 - gamma * Lambda_F @ tlx.ops.transpose(Lambda_S)
    G = 1 / G
    return G

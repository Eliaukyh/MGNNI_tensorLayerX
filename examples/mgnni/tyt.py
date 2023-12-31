import os
import numpy as np
# import tensorflow as tf

os.environ['TL_BACKEND'] = 'torch'
# os.environ['TL_BACKEND'] = 'tensorflow'

import tensorlayerx as tlx
import torch


# def norm(X):
#     res = X
#     if tlx.BACKEND == 'torch':
#         res = torch.norm(X)
#     elif tlx.BACKEND == 'tensorflow':
#         res = tf.norm(X)
#
#     return res


# def fwd_solver(f, z_init, threshold, max_iter, mode='abs'):
#     z_prev, z = z_init, f(z_init)
#     nstep = 0
#     while nstep < max_iter:
#         z_prev, z = z, f(z)
#         abs_diff = norm(z_prev - z).detach().numpy().item()
#         if abs_diff < threshold:
#             break
#         nstep += 1
#     if nstep == max_iter:
#         print(f'step {nstep}, not converged, abs_diff: {abs_diff}')
#     del z_prev
#     return z, abs_diff


def tesss(x, z):
    return x @ z


def torch_norm(input, p="fro", dim=None, keepdim=False, out=None, dtype=None):
    if p == "fro":
        norm_np = np.linalg.norm(input, ord="fro", axis=dim, keepdims=keepdim)
    elif p == "nuc":
        norm_np = np.linalg.norm(input, ord="nuc", axis=dim, keepdims=keepdim)
    else:
        norm_np = np.linalg.norm(input, ord=p, axis=dim, keepdims=keepdim)

    return norm_np


if __name__ == '__main__':
    import numpy as np

    x_t = torch.tensor([[1, 2, 3.], [2, 3, 4]])

    print(x_t.sum(1))
    print(tlx.reduce_sum(x_t, 1))

    # n = 100000
    # l = []
    # x = torch.randn(1500, 1500)
    # B = torch.randn(1500, 1500)
    # z, a = fwd_solver(lambda z: tesss(x, z), z_init=torch.nn.init.xavier_uniform(B), threshold=1e-6, max_iter=300)

    # set TL_BACKEND = 'torch'
    # net = tlx.nn.Input([10, 5], name='input')
    # linear = tlx.nn.Linear(out_features=8, act=tlx.ReLU, in_features=5, name='linear_1')
    # tensor = tlx.nn.Linear(out_features=800, act=tlx.ReLU, name='linear_2')(net)
    # print(linear.trainable_weights)
    # F = tlx.nn.Parameter(tlx.convert_to_tensor(np.random.randn(3, 3), dtype=tlx.float32))
    # F = np.array([0.991])
    # F_t = tlx.convert_to_tensor(F)
    # F_t.trainable=True
    # F_n = tlx.Variable(F_t)
    # F_ta = tlx.convert_to_tensor(F_n)
    # x = tlx.ops.constant(value=[[1, 2, 3], [2, 3, 4]])
    # x_t = torch.tensor([[1, 2, 3.], [2, 3, 4]])
    # x_tf = tlx.constant(value=[[1, 2., 3], [2, 3, 4]])

    # x_tf = tlx.convert_to_tensor(tlx.convert_to_numpy(x_tf).T)
    # x_tf = tlx.ops.transpose(x_tf)

    # y_t = torch.norm(x_t)
    # y = tlx.ops.l2_normalize(x, axis=1)
    # # y = tlx.ops.l2_normalize(y, axis=0)
    # y_tf = tf.norm(x_tf)
    # print(y_t.numpy().item())

    # print(torch_norm(x_t))
    # print(x_tf)
    # print('ssdsdssdsss')
    # print(y)
    # print('ssdsdssdsss')
    # print(y_tf.numpy().item())
    # print(F_n)
    # print(F_ta.trainable)
    # F = tlx.xavier_uniform(F.shape)
    # print(F)

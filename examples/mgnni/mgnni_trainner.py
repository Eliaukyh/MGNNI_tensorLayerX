import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TL_BACKEND'] = 'torch'
# os.environ['TL_BACKEND'] = 'mindspore'
# os.environ['TL_BACKEND'] = 'paddle'
# os.environ['TL_BACKEND'] = 'tensorflow'  # set your backend here, default `tensorflow`
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

sys.path.insert(0, os.path.abspath('../../'))  # adds path2gammagl to execute in command line.
import argparse
import tensorlayerx as tlx  # 导入tensorlayerx包
from gammagl.datasets import WebKB, WikipediaNetwork  # 专门做读取数据的类Planetoid
from gammagl.utils import add_self_loops, calc_gcn_norm, mask_to_index, set_device
from gammagl.models import MGNNI, MGNNI_m_att  #
from tensorlayerx.model import TrainOneStep, WithLoss
import gammagl.transforms as T
# from memory_profiler import profile

class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        logits = self.backbone_network(data['x'].t(), data['edge_index'], None, data['num_nodes'])
        # print(data['train_idx'])
        train_logits = tlx.gather(logits, data['train_idx'])
        # print(len(data['train_idx']))
        train_y = tlx.gather(data['y'], data['train_idx'])
        loss = self._loss_fn(train_logits, train_y)
        return loss

def calculate_acc(logits, y, metrics):
    """
    Args:
        logits: node logits
        y: node labels
        metrics: tensorlayerx.metrics

    Returns:
        rst
    """
    metrics.update(logits, y)
    rst = metrics.result()
    metrics.reset()
    return rst

# @profile
def main(args):
    # load datasets
    # set_device(5)
    # tlx.set_seed(args.seed)
    if str.lower(args.dataset) in ['cornell', 'texas', 'wisconsin']:
        # raise ValueError('Unknown dataset: {}'.format(args.dataset))
        dataset = WebKB(args.dataset_path, args.dataset, transform=T.NormalizeFeatures())
    elif str.lower(args.dataset) in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(args.dataset_path, args.dataset, transform=T.NormalizeFeatures())
    # dataset.process()
    graph = dataset[0]
    if args.dataset in ['cornell', 'texas', 'wisconsin']:
        edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes, n_loops=args.self_loops)
    else:
        edge_index = graph.edge_index
    edge_weight = tlx.convert_to_tensor(calc_gcn_norm(edge_index, graph.num_nodes))

    if not os.path.exists(args.path):
        os.mkdir(args.path)

    ks = eval(args.ks)
    ks_str = '-'.join(map(str, ks))
    result_name = '_'.join(
        [str(args.dataset), args.model, ks_str, str(args.epochs), str(args.lr), str(args.l2_coef),
         str(args.gamma),
         str(args.idx_split), str(int(time.time()))]) + '.txt'
    result_path = os.path.join(args.path, result_name)
    filep = open(result_path, 'w')
    filep.write(str(args) + '\n')

    # ti = graph.train_mask[:250]

    # tmp_train = []
    # tmp_test = []
    # tmp_val = []
    # for i in range(10):
    #     tmp_train.append(mask_to_index(graph.train_mask[i * 251: (i + 1) * 251]))
    #     tmp_test.append(mask_to_index(graph.test_mask[i * 251: (i + 1) * 251]))
    #     tmp_val.append(mask_to_index(graph.val_mask[i * 251: (i + 1) * 251]))

    train_idx = mask_to_index(graph.train_mask[args.idx_split])
    test_idx = mask_to_index(graph.test_mask[args.idx_split])
    val_idx = mask_to_index(graph.val_mask[args.idx_split])

    # train_idx = tlx.stack(tmp_train, axis=1)[:, args.idx_split]
    # test_idx = tlx.stack(tmp_test, axis=1)[:, args.idx_split]
    # val_idx = tlx.stack(tmp_val, axis=1)[:, args.idx_split]

    # train_idx = tlx.stack(tmp_train, axis=1)[:, args.idx_split]
    # test_idx = tlx.stack(tmp_test, axis=1)[:, args.idx_split]
    # val_idx = tlx.stack(tmp_val, axis=1)[:, args.idx_split]

    if args.model == 'MGNNI_m_MLP':
        net = MGNNI(
            dataset.num_node_features,
            dataset.num_classes,
            nhid=64,
            ks=ks,
            threshold=1e-6,
            max_iter=300,
            gamma=0.8,
            batch_norm=args.batch_norm
        )
    elif args.model == 'MGNNI_m_att':
        net = MGNNI_m_att(
            dataset.num_node_features,
            dataset.num_classes,
            ks=[1, 2],
            threshold=1e-6,
            max_iter=300,
            gamma=0.8,
        )

    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)
    metrics = tlx.metrics.Accuracy()
    train_weights = net.trainable_weights

    loss_func = SemiSpvzLoss(net, tlx.losses.softmax_cross_entropy_with_logits)

    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    data = {
        "x": graph.x,
        "y": graph.y,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "val_idx": val_idx,
        "num_nodes": graph.num_nodes,
    }

    best_val_acc = 0
    best_test_acc = 0
    for epoch in range(args.epochs):
        # optimizer.zero_grad()
        net.set_train()
        train_loss = train_one_step(data, graph.y)

        net.set_eval()
        logits = net(data['x'].t(), data['edge_index'], None, data['num_nodes'])
        val_logits = tlx.gather(logits, data['val_idx'])
        val_y = tlx.gather(data['y'], data['val_idx'])
        val_acc = calculate_acc(val_logits, val_y, metrics)

        test_logits = tlx.gather(logits, data['test_idx'])
        test_y = tlx.gather(data['y'], data['test_idx'])
        test_acc = calculate_acc(test_logits, test_y, metrics)

        outstr = "Epoch [{:0>3d}] ".format(epoch + 1) \
              + "  train loss: {:.4f}".format(train_loss.item()) \
              + "  val acc: {:.4f}".format(val_acc) \
              + "  test acc: {:.4f}".format(test_acc)

        print(outstr)

        filep.write(outstr + '\n')

        # print("Epoch [{:0>3d}] ".format(epoch + 1) \
        #       + "  train loss: {:.4f}".format(train_loss.item()) \
        #       + "  val acc: {:.4f}".format(val_acc)
        #       + "  test acc: {:.4f}".format(test_acc))

        # save best model on evaluation set
        if val_acc >= best_val_acc and test_acc > best_test_acc:
            best_val_acc = val_acc
            net.save_weights(args.best_model_path + net.name + ".npz", format='npz_dict')
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            # net.save_weights(args.best_model_path + net.name + ".npz", format='npz_dict')

    net.load_weights(args.best_model_path + net.name + ".npz", format='npz_dict')
    if tlx.BACKEND == 'torch':
        net.to(data['x'].device)
    net.set_eval()
    logits = net(data['x'].t(), data['edge_index'], data['edge_weight'], data['num_nodes'])
    test_logits = tlx.gather(logits, data['test_idx'])
    test_y = tlx.gather(data['y'], data['test_idx'])
    test_acc = calculate_acc(test_logits, test_y, metrics)

    finalstr = "the best val acc: {:.4f}".format(best_val_acc) \
               + '\n' + "Test acc:  {:.4f}".format(test_acc)

    print(finalstr)
    filep.write(finalstr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--no-cuda', action='store_true', default=False,
    #                     help='Disables CUDA training.')
    # parser.add_argument('--fastmode', action='store_true', default=False,
    #                     help='Validate during training pass.')
    # parser.add_argument('--seed', type=int, default=2023, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.5,
                        help='Initial learning rate.')
    parser.add_argument('--l2_coef', type=float, default=5e-6,
                        help='Weight decay (L2 loss on parameters).')
    # parser.add_argument('--hidden', type=int, default=64,
    #                     help='Number of hidden units.')
    # parser.add_argument('--dropout', type=float, default=0.5,
    #                     help='Dropout rate (1 - keep probability).')
    # parser.add_argument('--kappa', type=float, default=0.9,
    #                     help='Projection parameter. ||W|| <= kappa/lpf(A)')
    # ['cornell', 'texas', 'wisconsin', 'chameleon', 'squirrel']:
    parser.add_argument('--dataset', type=str, default="cornell",
                        help='Dataset to use.')
    parser.add_argument("--dataset_path", type=str, default=r'../', help="path to save dataset")
    parser.add_argument("--self_loops", type=int, default=1, help="number of graph self-loop")
    parser.add_argument("--best_model_path", type=str, default='./', help="best model path")
    # parser.add_argument('--feature', type=str, default="mul",
    #                     choices=['mul', 'cat', 'adj'],
    #                     help='feature-type')
    # parser.add_argument('--normalization', type=str, default='AugNormAdj',
    #                     choices=['AugNormAdj'],
    #                     help='Normalization method for the adjacency matrix.')
    # parser.add_argument('--degree', type=int, default=2,
    #                     help='degree of the approximation.')
    # parser.add_argument('--per', type=int, default=-1,
    #                     help='Number of each nodes so as to balance.')
    parser.add_argument('--model', type=str, default='MGNNI_m_att', choices=['MGNNI_m_MLP', 'MGNNI_m_att'],
                        help='model to use')
    # parser.add_argument('--experiment', type=str, default="base-experiment",
    #                     help='feature-type')
    # # IDM-SGC arguments
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--max_iter', type=int, default=300)
    parser.add_argument('--threshold', type=float, default=1e-6)
    parser.add_argument('--ks', type=str, default='[1,2]', help='a list of S^k, then concat for EIGNN_m_concat')
    # parser.add_argument('--layer_norm', type=int, default=0, choices=[0, 1],
    #                     help='whether to use layer norm within equilibrium')
    parser.add_argument('--batch_norm', type=int, default=0, choices=[0, 1], help='whether to use batch norm')
    # parser.add_argument('--learnable_alphas', type=int, default=0, choices=[0, 1],
    #                     help='whether to use learnable_alphas')
    # parser.add_argument('--spectral_radius_mode', type=int, default=0, choices=[0, 1],
    #                     help='whether to use spectral_radius mode')
    # parser.add_argument('--compute_jac_loss', type=int, default=0, choices=[0, 1], help='whether to compute jac loss')
    # parser.add_argument('--fp_layer', type=str, default='MGNNI_m_att')
    # # parser.add_argument('--fp_layer', type=str, default='EIGNN_m_att')
    # parser.add_argument('--momentum', type=float, default=0.8)
    parser.add_argument('--path', type=str, default='./results/')
    # parser.add_argument('--patience', type=int, default=200, help='early stop patience')
    parser.add_argument('--idx_split', type=int, default=5)
    # parser.add_argument('--undirected', type=int, default=0,
    #                     help='for heterophilic graphs, use undirected version or directed ')
    # parser.add_argument('--save_model', action='store_true', default=False,
    #                     help='Save to model')
    # parser.add_argument('--save_path', type=str, default='./saved_model_EIGNN_m',
    #                     help='path to save model')
    # parser.add_argument('--adj_preload_path', type=str, default=None,
    #                     help='the path to save adj')
    # parser.add_argument('--save_tensors', type=int, default=0, choices=[0, 1], help='whether to save Z* for debugging')
    # parser.add_argument('--num_layers', type=int, default=2, help='the number of layers for EIGNN_finite')

    args = parser.parse_args()

    main(args)

import numpy as np
import importlib

import torch
import torch.nn as nn

import networks.tiny.learner as Learner
import networks.tiny.modelfactory as mf
from networks.activation import ParametricSoftplus
from networks.blocks import ConvBNBlock
# from utils import CELoss, GradNormRegularizedLoss, CurvatureRegularizedLoss, CurvatureAndGradientRegularizedLoss

class BaseNet(torch.nn.Module):

    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(BaseNet, self).__init__()

        self.args = args
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_tasks = n_tasks
        self.cuda = args.cuda

        # setup network
        if args.dataset == 'tinyimagenet':
            config = mf.ModelFactory.get_model(sizes=[self.n_outputs], dataset=self.args.dataset, args=args)
            self.net = Learner.Learner(config, args)
            self.net.n_rep = 6
            self.net.multi_head = True
        else:
            if args.dataset in ['mnist_permutations', 'pmnist']:
                network = importlib.import_module('networks.mlp')
            elif args.dataset == 'cifar100':
                if args.activation_name is None:
                    network = importlib.import_module('networks.alexnet')
                else:
                    network = importlib.import_module('networks.alexnet_spectral')
            elif args.dataset == 'cifar100_superclass':
                if args.activation_name is None:
                    network = importlib.import_module('networks.lenet')
                else:
                    # print(args.activation)
                    network = importlib.import_module('networks.lenet_spectral')
            self.net = network.Learner(self.n_inputs, self.n_outputs, self.n_tasks, self.args)
            self.net_old = network.Learner(self.n_inputs, self.n_outputs, self.n_tasks, self.args)
            self.old_state = None
            self.prev_task = 0
            
        if self.cuda:
            self.net = self.net.cuda()
            self.net_old = self.net_old.cuda()

        if self.net.multi_head:
            self.nc_per_task = int(n_outputs / n_tasks)
        else:
            self.nc_per_task = n_outputs

        # setup losses
        self.loss_criterion = torch.nn.CrossEntropyLoss()
        # if args.regularized_loss is None:
        #     self.loss_ce = CELoss(model=self.net)
        # elif args.regularized_loss == 'grad_curvature':
        #     self.loss_ce = CurvatureAndGradientRegularizedLoss(model=self.net)
        # elif args.regularized_loss == 'grad':
        #     self.loss_ce = GradNormRegularizedLoss(model=self.net)
        # elif args.regularized_loss == 'curvature':
        #     self.loss_ce = CurvatureRegularizedLoss(model=self.net)
        # setup optimizer
        self.lr = args.lr
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=args.momentum)

        # define training params
        self.current_task = -1
        self.epoch = 0
        self.real_epoch = 0
        self.iter = 0
        self.glances = args.glances
        self.n_epochs = args.n_epochs
        self.batchSize = args.batch_size
        self.test_batch_size = args.test_batch_size

        # setup replay buffer (episodic memory)
        self.memories = args.memories  # mem_size of M
        self.age = 0  # total number of training samples
        self.M = []

        # setup GPM
        self.mem_batch_size = args.mem_batch_size
        self.M_vec = []  # bases of GPM
        self.M_val = []  # eigenvalues of each basis of GPM
        self.M_task = []  # the task id of each basis, only used to analysis GPM

    def compute_offsets(self, t):
        if self.net.multi_head:
            # mapping from classes to their idx within a task
            offset1 = t * self.nc_per_task
            offset2 = min((t + 1) * self.nc_per_task, self.n_outputs)
        else:
            offset1 = 0
            offset2 = self.nc_per_task

        return offset1, offset2
    
    @torch.no_grad()
    def _apply_hooks_and_init(self, device):
        # add forward hooks        
        self.fwd_handles = []
        for m in self.net.modules():
            if isinstance(m, ConvBNBlock):
                handle_f = m.register_forward_hook(self._extract_lipschitz)
                self.fwd_handles.append(handle_f) 
            elif isinstance(m, ParametricSoftplus):
                handle_f = m.register_forward_hook(self._extract_curvature)
                self.fwd_handles.append(handle_f)

        self.modules_visited = []
        self.betas = []
        self.gammas = []
        self.index = 0

    @torch.no_grad()
    def _remove_hooks(self):
        for f in self.fwd_handles:
            f.remove()

    def _extract_lipschitz(self, module, input, output):
        if module not in self.modules_visited:
            self.gammas.append(module.log_lipschitz)

    def _extract_curvature(self, module, input, output):
        self.betas.append(module.log_beta.exp())

    def take_loss(self, x, y, t, fast_weights=None):
        """
            Get loss of a task t
        """
        outputs = self.net.forward(x, fast_weights)

        if self.net.multi_head:
            # make sure we predict classes within the current task
            offset1, offset2 = self.compute_offsets(t)
            loss = self.loss_criterion(outputs[:, offset1:offset2], y - offset1)
        else:
            loss = self.loss_criterion(outputs, y)
        return loss
        

    def meta_loss(self, x, y, tasks, fast_weights=None):
        """
            Get loss of multiple tasks 

            Basically CE Loss 
        """
        # x = x.requires_grad_()
        # self._apply_hooks_and_init(x.device)
        outputs = self.net.forward(x, fast_weights)
        # self._remove_hooks()

        loss = 0.0

        grad_cur_task = 0.0
        grad_prev_tasks = 0.0
        # print(f"Type tasks: {type(tasks)}")
        for task in np.unique(tasks.data.cpu().numpy()):
            task = int(task)
            idx = torch.nonzero(tasks == task).view(-1)

            if self.net.multi_head:
                offset1, offset2 = self.compute_offsets(task)
                loss += self.loss_criterion(outputs[idx, offset1:offset2], y[idx] - offset1) * len(idx)
            else:
                loss += self.loss_criterion(outputs[idx], y[idx]) * len(idx)
            
            # Compute grad_norm of each task
        #     graph_required = self.args.second_order
        #     grads = list(torch.autograd.grad(loss, self.net.get_params(), create_graph=graph_required, retain_graph=graph_required,
        #                                     allow_unused=True))
            
        #     grad_norm = torch.norm(torch.stack(
        #         [
        #             torch.norm(p, p=2) for p in grads if p is not None
        #         ]
        #     ))
            
        #     if task != self.current_task:
        #         grad_prev_tasks += grad_norm.item() / len(idx)
        #     else:
        #         grad_cur_task = grad_norm.item() / len(idx)
        # if self.current_task > 0:
        #     grad_prev_tasks /= (len(np.unique(tasks.data.cpu().numpy())) - 1)
        
        
        raw_loss = loss / len(y)
        metadata = {
            'loss': raw_loss,
        }
        return metadata

    def update_optimizer(self, lr=None):
        if lr is None:
            lr = self.lr

        # if self.args.activation_name is not None:
        #     psoftplus_paramnames, bn_paramnames = [],[]
        #     for (name, layer) in self.net.named_modules():
        #         if type(layer) == ParametricSoftplus:
        #             for p in layer.named_parameters():
        #                 psoftplus_paramnames.append(name + '.' + p[0])
        #         if type(layer) == ConvBNBlock:
        #             for p in layer.named_parameters():
        #                 if 'log_lipschitz' in p[0]:
        #                     bn_paramnames.append(name + '.' + p[0])
        #     softplus_params = list(map(lambda x: x[1],
        #                             list(filter(lambda kv: kv[0] in psoftplus_paramnames, self.net.named_parameters()))))
        #     bn_thresh_params = list(map(lambda x: x[1],
        #                                 list(filter(lambda kv: kv[0] in bn_paramnames, self.net.named_parameters()))))
        #     base_params = list(map(lambda x: x[1],
        #                         list(filter(
        #                             lambda kv: (kv[0] not in psoftplus_paramnames) and (kv[0] not in bn_paramnames),
        #                             self.net.named_parameters()))))
        #     # print(f"base_params: {len(base_params)} elements, softplus_params: {len(softplus_params)} elements, bn_thresh_params: {len(bn_thresh_params)} elements")
        #     self.optimizer = torch.optim.SGD(
        #         [
        #             {"params": base_params, "weight_decay": self.args.weight_decay},
        #             {"params": softplus_params, "weight_decay": 0},
        #             {"params": bn_thresh_params, "weight_decay": 0}
        #         ],
        #         lr=lr,
        #         momentum=self.args.momentum
        #     )
        # else:
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=self.args.momentum)
        # milestone_fracs = [0.750, 0.875]
        # gamma = 0.1
        # milestones = [int(_ * self.args.n_epochs) for _ in milestone_fracs]
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
        #                                                 milestones=milestones,
        #                                                 gamma=gamma)

 
    def push_to_mem(self, batch_x, batch_y, t):
        """
            Reservoir sampling to push subsampled stream of data points to replay buffer
        """
        if self.real_epoch > 0:
            return

        batch_x = batch_x.cpu()
        batch_y = batch_y.cpu()
        t = t.cpu()

        for i in range(batch_x.shape[0]):
            self.age += 1
            if len(self.M) < self.memories:
                self.M.append([batch_x[i], batch_y[i], t])
            else:
                p = np.random.randint(0, self.age)
                if p < self.memories:
                    self.M[p] = [batch_x[i], batch_y[i], t]

    def get_batch(self, x, y, t):
        """
            Given the new data points, create a batch of old + new data,
            where old data is sampled from the replay buffer
            """
        t = (torch.ones(x.shape[0]).int() * t)

        if len(self.M) > 0:
            MEM = self.M
            order = np.arange(len(MEM))
            np.random.shuffle(order)
            index = order[:min(x.shape[0], len(MEM))]

            x = x.cpu()
            y = y.cpu()

            for k, idx in enumerate(index):
                ox, oy, ot = MEM[idx]
                x = torch.cat((x, ox.unsqueeze(0)), 0)
                y = torch.cat((y, oy.unsqueeze(0)), 0)
                t = torch.cat((t, ot.unsqueeze(0)), 0)

        # handle gpus if specified
        if self.cuda:
            x = x.cuda()
            y = y.cuda()
            t = t.cuda()

        return x, y, t

    def train_restgpm(self):
        """
            update grad to the projection on the rest of subspace spanned by GPM
            """
        j = 0

        for name, p in self.net.named_parameters():
            # print(f"name: {name}")
            # only handle conv weight and fc weight
            if p.grad is None:
                continue
            if p.grad.ndim != 2 and p.grad.ndim != 4:
                continue
            # print(f"name: {name}")

            if j < len(self.M_vec):
                if self.args.model in ['fsdgpm'] and self.args.method in ['dgpm', 'xdgpm']:
                    # lambdas = torch.sigmoid(self.args.tmp * self.lambdas[j]).reshape(-1)
                    lambdas = self.lambdas[j]
                else:
                    lambdas = torch.ones(self.M_vec[j].shape[1])

                if self.cuda:
                    self.M_vec[j] = self.M_vec[j].cuda()
                    lambdas = lambdas.cuda()

                if p.grad.ndim == 4:
                    # rep[i]: n_samples * n_features
                    grad = p.grad.reshape(p.grad.shape[0], -1)
                    grad -= torch.mm(torch.mm(torch.mm(grad, self.M_vec[j]), torch.diag(lambdas)), self.M_vec[j].T)
                    p.grad = grad.reshape(p.grad.shape).clone()
                else:
                    p.grad -= torch.mm(torch.mm(torch.mm(p.grad, self.M_vec[j]), torch.diag(lambdas)), self.M_vec[j].T)

                j += 1

    def set_gpm_by_svd(self, threshold):
        """
            Get the bases matrix (GPM) based on data sampled from replay buffer
            """
        assert len(self.M) > 0

        self.M_vec = []
        self.M_val = []

        index = np.arange(len(self.M))
        np.random.shuffle(index)

        for k, idx in enumerate(index):
            if k < min(self.mem_batch_size, len(self.M)):
                ox, oy, ot = self.M[idx]
                if k == 0:
                    mx = ox.unsqueeze(0)
                else:
                    mx = torch.cat((mx, ox.unsqueeze(0)), 0)

        # handle gpus if specified
        if self.cuda:
            mx = mx.cuda()

        with torch.no_grad():
            rep = self.net.forward(mx, svd=True)

        for i in range(len(rep)):
            # rep[i]: n_samples * n_features
            assert rep[i].ndim == 2

            u, s, v = torch.svd(rep[i].detach().cpu())

            if threshold[i] < 1:
                r = torch.cumsum(s ** 2, 0) / torch.cumsum(s ** 2, 0)[-1]
                k = torch.searchsorted(r, threshold[i]) + 1
            elif threshold[i] > 10:
                # differ with thres = 1
                k = int(threshold[i])
            else:
                k = len(s)

            if k > 0:
                self.M_vec.append(v[:, :k].cpu())
                self.M_val.append(s[:k].cpu())